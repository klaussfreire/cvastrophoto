# -*- coding: utf-8 -*-
import collections
import socket
import select
import logging
import json
import threading
import weakref
import itertools

logger = logging.getLogger('guiding.phd2.client')

class RPCError(Exception):
    pass

class RPCTimeoutError(RPCError):
    pass

class PHD2Client(object):

    def __init__(self, host, port=4400):
        self.host = host
        self.port = port

        self._socket = None
        self._iothread = None
        self._listeners = collections.defaultdict(list)
        self._stop = False
        self._rdbuf = []
        self._wrbuf = []
        self._idgen = itertools.count()

    def listen(self, listener, msgtype=None):
        self._listeners[msgtype].append(listener)

    def remove_listener(self, listener, msgtype=None):
        listeners = self._listeners[msgtype]
        if listener in listeners:
            listeners.remove(listener)

    def _invoke_listeners(self, listeners, msg):
        removed_listeners = False
        for i, listener in enumerate(listeners):
            if listener is None:
                continue

            continue_listening = listener(msg)
            if not continue_listening:
                listeners[i] = None
                removed_listeners = True

        if removed_listeners:
            listeners[:] = filter(None, listeners)

    def _handle_message(self, msg):
        logger.debug("PHD2 message: %r", msg)
        try:
            msg = json.loads(msg)
        except Exception:
            logger.warning("Ignoring malformed message: %r", msg)
            return

        self._invoke_listeners(self._listeners[None], msg)

        msgtype = msg.get('Event')
        if msgtype is not None:
            self._invoke_listeners(self._listeners[msgtype], msg)

    def _drain_buffer(self):
        while self._rdbuf and '\n' in self._rdbuf[-1]:
            msgbuf = []
            for i, block in enumerate(self._rdbuf):
                if '\n' in block:
                    msgbuf.append(block)
                    block = ''.join(msgbuf)
                    lines = block.splitlines()
                    if not block.endswith('\n'):
                        self._rdbuf[:i+1] = [lines[-1]]
                        del lines[-1:]
                    else:
                        del self._rdbuf[:i+1]
                    for line in lines:
                        self._handle_message(line)
                    break
                else:
                    msgbuf.append(block)
            else:
                break

    @staticmethod
    def _io_loop(wself):
        timeout = 1.0
        blocksize = 1 << 20
        while True:
            self = wself()
            if self is None or self._stop:
                break

            sock = self._socket
            rlist = [sock]
            wlist = [sock] if self._wrbuf else []
            xlist = [sock]
            del self

            rlist, wlist, xlist = select.select(rlist, wlist, xlist, timeout)
            self = wself()
            if self is None or self._stop:
                break

            if xlist:
                self.disconnect()

            if rlist:
                block = sock.recv(blocksize)
                logger.debug("Received %d bytes", len(block))
                if not block:
                    self.disconnect()
                    break
                else:
                    self._rdbuf.append(block)
                    if '\n' in block:
                        self._drain_buffer()

            if wlist:
                block = self._wrbuf[0]
                written = sock.send(block)
                logger.debug("Sent %d bytes", written)
                if written >= len(block):
                    del self._wrbuf[0]
                elif written > 0:
                    self._wrbuf[0] = block[written:]

            del self

    def connect(self):
        if self.is_connected():
            raise RuntimeError("Already connected")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        self._socket = sock
        self._stop = False
        self._iothread = iothread = threading.Thread(target=self._io_loop, args=(weakref.ref(self),))
        iothread.start()

    def is_connected(self):
        return self._socket is not None

    def disconnect(self):
        if self.is_connected():
            self._socket.close()
            self._socket = None
            self._stop = True
            if self._iothread is not None:
                self._iothread.join()
                self._iothread = None

    def _send_message(self, msg):
        self._wrbuf.append(json.dumps(msg) + '\r\n')

    def _call_method(self, method, *pparams, **kwparams):
        timeout = kwparams.pop('_timeout', 600)

        if pparams and kwparams:
            raise RuntimeError("Can only pass positional or keyword params, not both")

        msgid = next(self._idgen)
        msg = dict(method=method, id=msgid)

        if pparams:
            msg['params'] = list(pparams)
        elif kwparams:
            msg['params'] = kwparams

        respevent = threading.Event()
        resp = []

        def message_listener(msg):
            if 'jsonrpc' in msg and msg.get('id') == msgid:
                resp[:] = [msg]
                respevent.set()
                return False
            else:
                return True

        self.listen(message_listener)
        self._send_message(msg)
        respevent.wait(timeout)

        if not resp:
            self.remove_listener(message_listener)
            raise RPCTimeoutError()
        else:
            resp = resp[0]
            if 'error' in resp:
                raise RPCError(resp['error'])
            elif 'result' in resp:
                return resp['result']

    def dither(self, amount=5.0, raOnly=False, settle_pixels=2.0, settle_time=10, settle_timeout=120):
        settle_event = threading.Event()
        settle_result = []
        def on_settle_done(msg):
            settle_result[:] = [msg]
            settle_event.set()
            return False

        self.listen(on_settle_done, 'SettleDone')
        try:
            self._call_method(
                "dither",
                amount=float(amount), raOnly=bool(raOnly),
                settle=dict(
                    pixels=float(settle_pixels),
                    time=int(settle_time),
                    timeout=int(settle_timeout),
                )
            )
            if not settle_event.wait(settle_time + settle_timeout):
                raise RPCTimeoutError("Settle failed - timeout waiting for settle event")
        finally:
            self.remove_listener(on_settle_done, 'SettleDone')

        if not settle_result:
            raise RPCError("Settle failed - got no result")
        else:
            settle_result = settle_result[0]
            status = settle_result.get('Status')
            if status == 0:
                return True
            else:
                raise RPCError("Settle failed - %s" % (settle_result.get('Error'),))

    def get_app_state(self):
        return self._call_method("get_app_state")

    def get_calibrated(self):
        return self._call_method("get_calibrated")
