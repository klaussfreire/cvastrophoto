# -*- coding: utf-8 -*-
import time
import threading
import os.path
import collections
import queue

import PyIndi
from astropy.io import fits

import logging


logger = logging.getLogger(__name__)


class ConnectionError(Exception):
    pass


class NotSubscribedError(Exception):
    pass


class IndiDevice(object):

    def __init__(self, client, d):
        self.d = d
        self.client = client

    @property
    def properties(self):
        return self.client.properties[self.d.getDeviceName()]

    def waitConnect(self):
        logger.info("Connecting %r", self.d.getDeviceName())
        connect = self.waitSwitch("CONNECTION")

        if connect is None:
            raise ConnectionError("Can't find connect switch")

        connect[0].s = PyIndi.ISS_ON  # the "CONNECT" switch
        connect[1].s = PyIndi.ISS_OFF # the "DISCONNECT" switch
        self.client.sendNewSwitch(connect)

        if not self.waitCondition(lambda: self.properties["CONNECTION"][0]):
            raise ConnectionError("Could not connect")
        logger.info("Connected %r", self.d.getDeviceName())

    def waitDisonnect(self):
        logger.info("Disconnecting %r", self.d.getDeviceName())

        connect = self.waitSwitch("CONNECTION")

        if connect is None:
            raise ConnectionError("Can't find connect switch")

        connect[0].s = PyIndi.ISS_OFF # the "CONNECT" switch
        connect[1].s = PyIndi.ISS_ON  # the "DISCONNECT" switch
        self.client.sendNewSwitch(connect)

        if not self.waitCondition(lambda: not self.properties["CONNECTION"][0]):
            raise ConnectionError("Could not disconnect")
        logger.info("Disconnected %r", self.d.getDeviceName())

    def waitCondition(self, condition):
        deadline = time.time() + self.client.DEFAULT_TIMEOUT
        while not condition() and time.time() < deadline:
            self.client.any_event.wait(10)
            self.client.any_event.clear()
        return condition()

    def waitProperty(self, ptype, name):
        if ptype == PyIndi.INDI_NUMBER:
            get = self.d.getNumber
        elif ptype == PyIndi.INDI_TEXT:
            get = self.d.getText
        elif ptype == PyIndi.INDI_SWITCH:
            get = self.d.getSwitch
        elif ptype == PyIndi.INDI_BLOB:
            get = self.d.getBLOB
        else:
            raise NotImplementedError

        prop = None
        wait = False
        deadline = time.time() + self.client.DEFAULT_TIMEOUT
        while prop is None and time.time() < deadline:
            if wait:
                self.client.property_event.wait(10)
                self.client.property_event.clear()
            else:
                wait = True
            prop = get(name)

        return prop

    def waitSwitch(self, name):
        return self.waitProperty(PyIndi.INDI_SWITCH, name)

    def waitNumber(self, name):
        return self.waitProperty(PyIndi.INDI_NUMBER, name)

    def waitText(self, name):
        return self.waitProperty(PyIndi.INDI_TEXT, name)

    def waitBLOB(self, name):
        return self.waitProperty(PyIndi.INDI_BLOB, name)

    def setNumber(self, name, values):
        if isinstance(values, (int, float)):
            values = [values]

        nvp = self.waitNumber(name)
        if nvp is None:
            raise RuntimeError("Can't find property %r" % (name,))

        assert len(nvp) == len(values)
        for i in xrange(len(nvp)):
            nvp[i].value = values[i]

        self.client.sendNewNumber(nvp)

    def setSwitch(self, name, values):
        svp = self.waitSwitch(name)
        if svp is None:
            raise RuntimeError("Can't find property %r" % (name,))

        assert len(svp) == len(values)
        for i in xrange(len(svp)):
            svp[i].s = PyIndi.ISS_ON if values[i] else PyIndi.ISS_OFF

        self.client.sendNewSwitch(svp)

    def setNarySwitch(self, name, value):
        svp = self.waitSwitch(name)
        if svp is None:
            raise RuntimeError("Can't find property %r" % (name,))

        for i in xrange(len(svp)):
            svp[0].s = PyIndi.ISS_ON if value == i else PyIndi.ISS_OFF

        self.client.sendNewSwitch(svp)

    def setText(self, name, values):
        tvp = self.waitSwitch(name)
        if tvp is None:
            raise RuntimeError("Can't find property %r" % (name,))

        assert len(tvp) == len(values)
        for i in xrange(len(tvp)):
            tvp[i].text = values[i]

        self.client.sendNewText(tvp)


class IndiCCD(IndiDevice):

    DEFAULT_QUEUE = 10

    def __init__(self, client, d):
        super(IndiCCD, self).__init__(client, d)
        self.subscriptions = set()
        self.blob_queues = {}

    def subscribeBLOB(self, name="CCD1"):
        self.subscriptions.add(name)
        self.client.listenBLOB(self.d.getDeviceName(), name, self.newBLOB)

    def unsubscribeBLOB(self, name="CCD1"):
        self.client.unlistenBLOB(self.d.getDeviceName(), name)
        self.subscriptions.discard(name)

    def newBLOB(self, name, blob):
        logger.info("Got new blob for %r/%r (%r)", self.d.getDeviceName(), name, blob.label)
        if name not in self.subscriptions:
            return

        q = self.blob_queues.get(name)
        if q is None:
            q = self.blob_queues.setdefault(name, queue.Queue(self.DEFAULT_QUEUE))
            self.client.blob_event.set()

        try:
            q.put_nowait(blob)
        except queue.Full:
            logger.warning(
                "Queue overflow receiving BLOB from %r/%r, discarded",
                self.d.getDeviceName(), name)

    def pullBLOB(self, name, wait=True):
        if name not in self.subscriptions:
            raise NotSubscribedError()

        q = self.blob_queues.get(name)
        if wait and q is None:
            while q is None:
                self.client.blob_event.wait(self.client.DEFAULT_TIMEOUT)
                self.client.blob_event.clear()
                q = self.blob_queues.get(name)
        if wait:
            get = q.get
        else:
            get = q.get_nowait
        return get()

    def expose(self, exposure):
        self.setNumber("CCD_EXPOSURE", exposure)

    def setLight(self):
        self.setSwitch("CCD_FRAME_TYPE", [True, False, False, False])

    def setBias(self):
        self.setSwitch("CCD_FRAME_TYPE", [False, True, False, False])

    def setDark(self):
        self.setSwitch("CCD_FRAME_TYPE", [False, False, True, False])

    def setFlat(self):
        self.setSwitch("CCD_FRAME_TYPE", [False, False, False, True])

    def blob2FitsHDUL(self, blob):
        return fits.HDUList(file=bytes(blob.getblobdata()))

    def writeBLOB(self, blob, file_or_path, overwrite=False):
        closeobj = None
        if isinstance(file_or_path, basestring):
            if not overwrite and os.path.exists(file_or_path):
                raise ValueError("File already exists: %r" % file_or_path)
            fileobj = closeobj = open(file_or_path, "wb")
        else:
            fileobj = file_or_path

        try:
            fileobj.write(blob.getblobdata())
        finally:
            if closeobj is not None:
                closeobj.close()


class IndiST4(IndiDevice):

    def pulseNorth(self, ms):
        self.setNumber("TELESCOPE_TIMED_GUIDE_NS", [ms, 0])

    def pulseSouth(self, ms):
        self.setNumber("TELESCOPE_TIMED_GUIDE_NS", [0, ms])

    def pulseWest(self, ms):
        self.setNumber("TELESCOPE_TIMED_GUIDE_WE", [ms, 0])

    def pulseEast(self, ms):
        self.setNumber("TELESCOPE_TIMED_GUIDE_WE", [0, ms])


class IndiClient(PyIndi.BaseClient):

    DEFAULT_HOST = 'localhost'
    DEFAULT_PORT = 7624

    DEFAULT_TIMEOUT = 600

    def __init__(self):
        self.devices = {}
        self.properties = collections.defaultdict(dict)
        self.blob_listeners = collections.defaultdict(dict)

        self.connection_event = threading.Event()
        self.property_event = threading.Event()
        self.blob_event = threading.Event()
        self.device_event = threading.Event()
        self.any_event = threading.Event()

        super(IndiClient, self).__init__()

    def waitCCD(self, device_name):
        return self._waitWrappedDevice(device_name, IndiCCD)

    def waitST4(self, device_name):
        return self._waitWrappedDevice(device_name, IndiST4)

    def _waitWrappedDevice(self, device_name, device_class):
        d = self.waitDevice(device_name)
        if d is not None:
            d = device_class(self, d)
        return d

    def waitDevice(self, device_name):
        dev = None
        wait = False
        deadline = time.time() + self.DEFAULT_TIMEOUT
        while dev is None and time.time() < deadline:
            if wait:
                self.device_event.wait(10)
                self.device_event.clear()
            else:
                wait = True
            dev = self.getDevice(device_name)
        return dev

    def newDevice(self, d):
        logger.info("New device: %r", d.getDeviceName())
        self.devices[d.getDeviceName] = d
        self.device_event.set()
        self.any_event.set()

    def newProperty(self, p):
        ptype = p.getType()
        if ptype == PyIndi.INDI_NUMBER:
            val = [ np.value for np in p.getNumber() ]
        elif ptype == PyIndi.INDI_TEXT:
            val = [ tp.text for tp in p.getText() ]
        elif ptype == PyIndi.INDI_SWITCH:
            val = [ sp.s for sp in p.getSwitch() ]
        else:
            val = 'unk'
        self.properties[p.getDeviceName()][p.getName()] = val
        self.property_event.set()
        self.any_event.set()

    def removeProperty(self, p):
        props = self.properties[p.getDeviceName()]
        props.pop(p.getName(), None)
        if not props:
            self.properties.pop(p.getDeviceName(), None)
        self.property_event.set()
        self.any_event.set()

    def listenBLOB(self, device_name, ccd_name, callback):
        self.setBLOBMode(PyIndi.B_ALSO, device_name, ccd_name)
        self.blob_listeners[device_name][ccd_name] = callback

    def unlistenBLOB(self, device_name, ccd_name):
        self.blob_listeners[device_name].pop(ccd_name, None)
        self.setBLOBMode(PyIndi.B_NEVER, device_name, ccd_name)

    def newBLOB(self, bp):
        bvp = bp.bvp

        self.blob_event.set()
        self.any_event.set()

        dev_listeners = self.blob_listeners.get(bvp.device, {})
        ccd_listener = dev_listeners.get(bvp.name)
        if ccd_listener is not None:
            ccd_listener(bvp.name, bp)

    def newSwitch(self, svp):
        val = [ sp.s for sp in svp ]
        self.properties[svp.device][svp.name] = val
        self.property_event.set()
        self.any_event.set()

    def newNumber(self, nvp):
        val = [ np.value for np in nvp ]
        self.properties[nvp.device][nvp.name] = val
        self.property_event.set()
        self.any_event.set()

    def newText(self, tvp):
        val = [ tp.text for tp in tvp ]
        self.properties[tvp.device][tvp.name] = val
        self.property_event.set()
        self.any_event.set()

    def newLight(self, lvp):
        self.any_event.set()

    def newMessage(self, d, m):
        self.any_event.set()

    def serverConnected(self):
        logger.info("INDI server connected")
        self.connection_event.set()
        self.any_event.set()

    def serverDisconnected(self, code):
        logger.info("INDI server disconnected")
        self.connection_event.set()
        self.any_event.set()
