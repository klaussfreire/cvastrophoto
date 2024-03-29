# -*- coding: utf-8 -*-
from __future__ import absolute_import

import time
import threading
import os.path
import collections
import queue
import weakref
import functools
from past.builtins import xrange, basestring

import PyIndi
from astropy.io import fits

import logging

from cvastrophoto.image import rgb
from . import driver_info


logger = logging.getLogger(__name__)


class ConnectionError(Exception):
    pass


class TimeoutError(Exception):
    pass


class NotSubscribedError(Exception):
    pass


class PropertyWrapper(object):
    """ INDI Property wrapper

    Wrapper to unify INDI properrty interface across versions
    """

    _GETTERS = {
        'group': 'getGroupName',
        'label': 'getLabel',
        'name': 'getName',
        'permission': 'getPermission',
        'state': 'getState',
        'type': 'getType',
        'p': 'getPermission',
        's': 'getState',
        'r': 'getRule',
    }

    def __init__(self, prop):
        self._prop = prop

    def __getattr__(self, attr):
        prop = self._prop
        try:
            return getattr(prop, attr)
        except AttributeError:
            if attr not in self._GETTERS:
                raise
            return getattr(prop, self._GETTERS[attr])()

    def __getitem__(self, ix):
        return self._prop[ix]

    def __len__(self):
        return len(self._prop)


class IndiDevice(object):

    _MONITOR_PROPS = []

    onPropertyChange = None

    @staticmethod
    def _weak_onPropertyChange(wself, dname, pname, vp):
        self = wself()
        if self is None:
            return False

        self.onPropertyChange(dname, pname, vp)

        return True

    def __init__(self, client, d):
        self.d = d
        self.client = client

        if self.onPropertyChange is not None:
            wself = weakref.ref(self)
            onPropertyChange = functools.partial(self._weak_onPropertyChange, wself)

            dname = self.name
            for pname in self._MONITOR_PROPS:
                client.listenProperty(dname, pname, onPropertyChange)

    def config_load(self, quick=False, optional=True):
        """ Load device configuration

        If the device supports it, load driver configuration from its config file.
        """
        return self.setNarySwitch('CONFIG_PROCESS', 'Load', quick=quick, optional=optional)

    def config_save(self, quick=False, optional=True):
        """ Save device configuration

        If the device supports it, save driver configuration to its config file.
        """
        return self.setNarySwitch('CONFIG_PROCESS', 'Save', quick=quick, optional=optional)

    def setActiveDevices(self, telescope=None, focuser=None, cfw=None):
        mods = {}
        if telescope is not None:
            mods['Telescope'] = telescope.name
        if focuser is not None:
            mods['Focuser'] = focuser.name
        if cfw is not None:
            mods['Filter'] = cfw.name
        self.setText('ACTIVE_DEVICES', mods, optional=True, quick=True)

    def autoconf(self, quick=True):
        """ Autoconfigure connection settings

        Try to guess workable connection settings for this device, and
        set it up for connection
        """
        device_name = self.name
        device_port = self.getTextByName('DEVICE_PORT', quick=True, optional=True)
        if device_port:
            device_port = next(iter(device_port.values()))
        else:
            device_port = None

        # Check device port accessible
        if device_port is not None and device_port.startswith('/dev/tty'):
            if not os.path.exists(device_port) or not os.access(device_port, os.R_OK | os.W_OK):
                # Can't open that port, pick another one if any have been detected
                system_ports = self.getSwitchByName('SYSTEM_PORTS', quick=True, optional=True) or []
                for candidate_port in system_ports:
                    if os.path.exists(candidate_port) and os.access(candidate_port, os.R_OK | os.W_OK):
                        logging.warning(
                            "Currently selected port for %r not accessible, switching to %r",
                            device_name, candidate_port)
                        self.setNarySwitch('SYSTEM_PORTS', candidate_port)
                else:
                    logging.warning("No port accessible for %r, connection likely to fail", device_name)

            baud_rate = driver_info.guess_baud_rate(
                device_name,
                device_port,
                self.getTextByName('DRIVER_INFO', quick=True, optional=True))

            if baud_rate:
                logger.info("Setting baud rate for %r to %r", device_name, baud_rate)
                self.setNarySwitch('DEVICE_BAUD_RATE', str(baud_rate), quick=quick, optional=True)

    @property
    def name(self):
        return self.d.getDeviceName()

    @property
    def properties(self):
        return self.client.properties[self.d.getDeviceName()]

    @property
    def property_meta(self):
        return self.client.property_meta[self.d.getDeviceName()]

    @property
    def connected(self):
        return self.properties.get("CONNECTION", (0,))[0]

    def connect(self):
        logger.info("Connecting %r", self.d.getDeviceName())
        connect = self.waitSwitch("CONNECTION")

        if connect is None:
            raise ConnectionError("Can't find connect switch")

        connect[0].s = PyIndi.ISS_ON  # the "CONNECT" switch
        connect[1].s = PyIndi.ISS_OFF # the "DISCONNECT" switch
        self.client.sendNewSwitch(connect)

    def disconnect(self):
        logger.info("Disconnecting %r", self.d.getDeviceName())

        connect = self.waitSwitch("CONNECTION")

        if connect is None:
            raise ConnectionError("Can't find connect switch")

        connect[0].s = PyIndi.ISS_OFF # the "CONNECT" switch
        connect[1].s = PyIndi.ISS_ON  # the "DISCONNECT" switch
        self.client.sendNewSwitch(connect)

    def onReconnect(self):
        pass

    def waitConnect(self, connect=True):
        if connect:
            self.connect()

        if not self.waitCondition(lambda: self.connected):
            raise ConnectionError("Could not connect")
        logger.info("Connected %r", self.d.getDeviceName())

    def waitDisonnect(self, disconnect=True):
        if disconnect:
            self.disconnect()

        if not self.waitCondition(lambda: not self.connected):
            raise ConnectionError("Could not disconnect")
        logger.info("Disconnected %r", self.d.getDeviceName())

    def waitCondition(self, condition, timeout=None):
        deadline = time.time() + (timeout or self.client.DEFAULT_TIMEOUT)
        while not condition() and time.time() < deadline:
            self.client.any_event.wait(10)
            self.client.any_event.clear()
        return condition()

    def waitPropertiesReady(self):
        deadline = time.time() + self.client.DEFAULT_TIMEOUT
        steady_deadline = time.time() + 2
        propset = set(self.properties)
        while time.time() < min(steady_deadline, deadline):
            if not self.client.property_event.wait(2):
                break
            npropset = set(self.properties)
            if npropset != propset:
                steady_deadline = time.time() + 2
            self.client.property_event.clear()
            propset = npropset

    def getAnyProperty(self, name):
        for get in (self.d.getNumber, self.d.getText, self.d.getSwitch, self.d.getBLOB):
            prop = get(name)
            if prop is not None and prop.isValid():
                return PropertyWrapper(prop)

    def waitProperty(self, ptype, name, quick=False):
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
        timeout = self.client.DEFAULT_TIMEOUT
        if quick:
            timeout = min(timeout, 2)
        deadline = time.time() + timeout
        wait_time = 2 if quick else 10
        while prop is None and time.time() < deadline:
            if wait:
                self.client.property_event.wait(wait_time)
                self.client.property_event.clear()
            else:
                wait = True
            with self.client.socketlock:
                prop = get(name)
                if not prop.isValid():
                    prop = None

        return prop

    def waitSwitch(self, name, quick=False):
        return self.waitProperty(PyIndi.INDI_SWITCH, name, quick=quick)

    def waitNumber(self, name, quick=False):
        return self.waitProperty(PyIndi.INDI_NUMBER, name, quick=quick)

    def waitText(self, name, quick=False):
        return self.waitProperty(PyIndi.INDI_TEXT, name, quick=quick)

    def waitBLOB(self, name, quick=False):
        return self.waitProperty(PyIndi.INDI_BLOB, name, quick=quick)

    def setNumber(self, name, values, quick=False, optional=False):
        if isinstance(values, (int, float)):
            values = [values]

        nvp = self.waitNumber(name, quick=quick)
        if nvp is None:
            if optional:
                logger.warning("Property %s on %s missing", name, self.d.getDeviceName())
                return
            raise RuntimeError("Can't find property %r" % (name,))

        if isinstance(values, dict):
            # Merge new values with current values
            dvalues = {pname.lower(): pvalue for pname, pvalue in values.items()}
            values = [dvalues.get(p.label.strip().lower(), p.value) for p in nvp]

        assert len(nvp) == len(values)
        for i in xrange(len(nvp)):
            nvp[i].value = values[i]

        self.client.sendNewNumber(nvp)

    def setSwitch(self, name, values, quick=False, optional=False):
        svp = self.waitSwitch(name, quick=quick)
        if svp is None:
            if optional:
                logger.warning("Property %s on %s missing", name, self.d.getDeviceName())
                return
            raise RuntimeError("Can't find property %r" % (name,))

        assert len(svp) == len(values)
        for i in xrange(len(svp)):
            svp[i].s = PyIndi.ISS_ON if values[i] else PyIndi.ISS_OFF

        self.client.sendNewSwitch(svp)

    def setNarySwitch(self, name, value, quick=False, optional=False):
        svp = self.waitSwitch(name, quick=quick)
        if svp is None:
            if optional:
                logger.warning("Property %s on %s missing", name, self.d.getDeviceName())
                return
            raise RuntimeError("Can't find property %r" % (name,))

        if isinstance(value, basestring):
            value = value.strip().lower()
            for i in xrange(len(svp)):
                svp[i].s = (
                    PyIndi.ISS_ON
                    if value in (svp[i].label.strip().lower(), svp[i].name.strip().lower())
                    else PyIndi.ISS_OFF
                )
        else:
            for i in xrange(len(svp)):
                svp[i].s = PyIndi.ISS_ON if value == i else PyIndi.ISS_OFF

        self.client.sendNewSwitch(svp)

    def setText(self, name, values, quick=False, optional=False):
        if isinstance(values, basestring):
            values = [values]

        tvp = self.waitText(name, quick=quick)
        if tvp is None:
            if optional:
                logger.warning("Property %s on %s missing", name, self.d.getDeviceName())
                return
            raise RuntimeError("Can't find property %r" % (name,))

        if isinstance(values, dict):
            # Merge new values with current values
            dvalues = {pname.lower(): pvalue for pname, pvalue in values.items()}
            values = [dvalues.get(p.label.strip().lower(), p.text) for p in tvp]

        assert len(tvp) == len(values)
        for i in xrange(len(tvp)):
            tvp[i].text = values[i]

        self.client.sendNewText(tvp)

    def getProperty(self, ptype, name, quick=False, optional=False):
        vp = self.waitProperty(ptype, name, quick=quick)
        if vp is None:
            if optional:
                logger.warning("Property %s on %s missing", name, self.d.getDeviceName())
                return
            raise RuntimeError("Can't find property %r" % (name,))

        return vp

    def getPropertyLabels(self, ptype, name, quick=False, optional=False):
        vp = self.getProperty(ptype, name, quick=quick, optional=optional)
        return [p.label.strip().lower() for p in vp]

    def getPropertyNames(self, ptype, name, quick=False, optional=False):
        vp = self.getProperty(ptype, name, quick=quick, optional=optional)
        return [p.name.strip().lower() for p in vp]

    def getNumberByLabel(self, name, quick=False, optional=False):
        nvp = self.getProperty(PyIndi.INDI_NUMBER, name, quick=quick, optional=optional)
        if nvp is None:
            return {}

        return {p.label.strip().lower(): p.value for p in nvp}

    def getNumberByName(self, name, quick=False, optional=False):
        nvp = self.getProperty(PyIndi.INDI_NUMBER, name, quick=quick, optional=optional)
        if nvp is None:
            return {}

        return {p.name.strip().lower(): p.value for p in nvp}

    def getNumberLabels(self, name, quick=False, optional=False):
        return self.getPropertyLabels(PyIndi.INDI_NUMBER, name, quick=quick, optional=optional)

    def getNumberNames(self, name, quick=False, optional=False):
        return self.getPropertyNames(PyIndi.INDI_NUMBER, name, quick=quick, optional=optional)

    def getSwitchByLabel(self, name, quick=False, optional=False):
        svp = self.getProperty(PyIndi.INDI_SWITCH, name, quick=quick, optional=optional)
        if svp is None:
            return {}

        return {p.label.strip().lower(): p.s == PyIndi.ISS_ON for p in svp}

    def getSwitchByName(self, name, quick=False, optional=False):
        svp = self.getProperty(PyIndi.INDI_SWITCH, name, quick=quick, optional=optional)
        if svp is None:
            return {}

        return {p.name.strip().lower(): p.s == PyIndi.ISS_ON for p in svp}

    def getSwitchLabels(self, name, quick=False, optional=False):
        return self.getPropertyLabels(PyIndi.INDI_SWITCH, name, quick=quick, optional=optional)

    def getSwitchNames(self, name, quick=False, optional=False):
        return self.getPropertyLabels(PyIndi.INDI_SWITCH, name, quick=quick, optional=optional)

    def getNarySwitchByLabel(self, name, quick=False, optional=False):
        svp = self.getProperty(PyIndi.INDI_SWITCH, name, quick=quick, optional=optional)
        if svp is None:
            return None

        for p in svp:
            if p.s == PyIndi.ISS_ON:
                return p.label.strip().lower()

    def getNarySwitchByName(self, name, quick=False, optional=False):
        svp = self.getProperty(PyIndi.INDI_SWITCH, name, quick=quick, optional=optional)
        if svp is None:
            return None

        for p in svp:
            if p.s == PyIndi.ISS_ON:
                return p.name.strip().lower()

    def getTextByLabel(self, name, quick=False, optional=False):
        tvp = self.getProperty(PyIndi.INDI_TEXT, name, quick=quick, optional=optional)
        if tvp is None:
            return {}

        return {p.label.strip().lower(): p.text for p in tvp}

    def getTextByName(self, name, quick=False, optional=False):
        tvp = self.getProperty(PyIndi.INDI_TEXT, name, quick=quick, optional=optional)
        if tvp is None:
            return {}

        return {p.name.strip().lower(): p.text for p in tvp}

    def getTextLabels(self, name, quick=False, optional=False):
        return self.getPropertyLabels(PyIndi.INDI_TEXT, name, quick=quick, optional=optional)

    def getTextNames(self, name, quick=False, optional=False):
        return self.getPropertyNames(PyIndi.INDI_TEXT, name, quick=quick, optional=optional)


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

    def onReconnect(self):
        for name in self.subscriptions:
            self.client.listenBLOB(self.d.getDeviceName(), name, self.newBLOB)

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
            self.client.blob_event.set()
        except queue.Full:
            logger.warning(
                "Queue overflow receiving BLOB from %r/%r, discarded",
                self.d.getDeviceName(), name)

    def pullBLOB(self, name, wait=True):
        if name not in self.subscriptions:
            raise NotSubscribedError()

        q = self.blob_queues.get(name)
        if wait and q is None:
            deadline = time.time() + self.client.DEFAULT_TIMEOUT
            while q is None and time.time() < deadline:
                self.client.blob_event.wait(1)
                self.client.blob_event.clear()
                q = self.blob_queues.get(name)
            if q is None:
                raise TimeoutError()

        deadline = time.time() + self.client.DEFAULT_TIMEOUT
        while time.time() < deadline:
            try:
                return q.get_nowait()
            except queue.Empty:
                self.client.blob_event.wait(1)
                self.client.blob_event.clear()

        raise TimeoutError()

    def expose(self, exposure):
        self.setNumber("CCD_EXPOSURE", exposure)

    def setUploadSettings(self, upload_dir=None, image_prefix=None, image_type=None, image_suffix=None):
        if not upload_dir:
            upload_dir = self.properties["UPLOAD_SETTINGS"][0]
        if not image_prefix and not image_type:
            image_pattern = self.properties["UPLOAD_SETTINGS"][1]
        else:
            image_pattern = '_'.join(filter(bool, [image_prefix, image_type, image_suffix, 'XXX']))
        self.setText("UPLOAD_SETTINGS", [upload_dir, image_pattern])

    def setLight(self):
        self.setNarySwitch("CCD_FRAME_TYPE", 0)

    def setBias(self):
        self.setNarySwitch("CCD_FRAME_TYPE", 1)

    def setDark(self):
        self.setNarySwitch("CCD_FRAME_TYPE", 2)

    def setFlat(self):
        self.setNarySwitch("CCD_FRAME_TYPE", 3)

    def setTransferFormatNative(self, **kw):
        self.setTransferFormat('native', **kw)

    def setTransferFormatFits(self, **kw):
        self.setTransferFormat('fits', **kw)

    def setTransferFormat(self, fmt, **kw):
        self.setNarySwitch("CCD_TRANSFER_FORMAT", fmt, **kw)

    @property
    def transfer_format(self):
        return self.getNarySwitchByLabel("CCD_TRANSFER_FORMAT", quick=True, optional=True)

    @transfer_format.setter
    def transfer_format(self, fmt):
        self.setTransferFormat(fmt)

    def blob2FitsHDUL(self, blob):
        return fits.HDUList(file=bytes(blob.getblobdata()))

    def blob2Image(self, blob):
        hdul = self.blob2FitsHDUL(blob)
        img = rgb.RGB(None, img=hdul[0].data, linear=True, autoscale=False)
        img.fits_header = hdul[0].header
        return img

    def pullImage(self, name, wait=True):
        return self.blob2Image(self.pullBLOB(name, wait))

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

    def detectCCDInfo(self, name="CCD1"):
        ccd_info = [np.value for np in self.waitNumber("CCD_INFO")]
        if list(filter(None, ccd_info[:2])):
            logger.info("CCD info for %r already set (%r)", self.d.getDeviceName(), ccd_info)
            return

        logger.info("Deetecting CCD info for %r", self.d.getDeviceName())
        self.setNumber("CCD_INFO", [2, 2, 3.0, 3.0, 3.0, 16])
        self.setNumber("CCD_FRAME", [0, 0, 2, 2])

        self.setUploadClient()
        self.expose(0.01)
        self.pullBLOB(name)

        ccd_info = [np.value for np in self.waitNumber("CCD_INFO")]
        logger.info("CCD info for %r (%r)", self.d.getDeviceName(), ccd_info)

        self.setNumber("CCD_FRAME", [0, 0, ccd_info[0], ccd_info[1]])
        logger.info("CCD frame set to full frame for %r", self.d.getDeviceName())

    def setUploadMode(self, mode, **kw):
        self.setNarySwitch("UPLOAD_MODE", mode, **kw)

    def setUploadClient(self, **kw):
        self.setUploadMode("Client", **kw)

    def setUploadLocal(self, **kw):
        self.setUploadMode("Local", **kw)

    @property
    def upload_mode(self):
        return self.getNarySwitchByLabel("UPLOAD_MODE", quick=True, optional=True)

    @upload_mode.setter
    def upload_mode(self, mode):
        self.setUploadMode(mode)

    @property
    def supports_cooling(self):
        return "CCD_COOLER" in self.properties

    @property
    def cooling_enabled(self):
        return (
            self.properties.get("CCD_COOLER", (False,))[0]
            or self.properties.get("CCD_COOLER_POWER", [0])[0] != 0
        )

    _cached_cooling_iface = None

    @property
    def _cooling_interface(self):
        props = self.properties

        temp_nvp = None
        temp_writable = False
        iface = self._cached_cooling_iface

        if iface is None:
            if "CCD_COOLER" in props and "CCD_TEMPERATURE" in props:
                temp_nvp = self.waitNumber("CCD_TEMPERATURE", quick=True)
                temp_writable = temp_nvp is not None and temp_nvp.p != PyIndi.IP_RO

            if temp_writable:
                iface = 'write_temp'

            if iface:
                logger.info("Cooling interface for %r is %r", self.name, iface)
                self._cached_cooling_iface = iface

        return iface

    def _cooling_dispatch(self, method, *p, **kw):
        iface = self._cooling_interface

        if iface is None:
            if not kw.get('optional'):
                raise NotImplementedError("Unknown cooling interface for this device")
            logger.info("Requested cooling operation %r on %r but cooling interface unknown", method, self.name)
            return

        return getattr(self, method + '_' + iface)(*p, **kw)

    def enable_cooling(self, target_temperature, quick=False, optional=False):
        if not self.supports_cooling:
            if not optional:
                raise NotImplementedError("Cooling not supported by this device")
            return

        return self._cooling_dispatch('_enable_cooling', target_temperature, quick=quick, optional=optional)

    def _enable_cooling_write_temp(self, target_temperature, quick=False, optional=False):
        logger.info("Enabling cooling on %r", self.name)
        self._set_cooling_temp_write_temp(target_temperature, quick=quick, optional=optional)
        self.setNarySwitch("CCD_COOLER", 0, quick=quick, optional=optional)
        if not quick:
            self.waitCondition(lambda: self.properties.get('CCD_COOLER', [True])[0], 10)
            logger.info("Cooling enabled on %r", self.name)
        self._set_cooling_temp_write_temp(target_temperature, quick=quick, optional=optional)
        logger.info("Set target temperature for %r to %r", self.name, target_temperature)

    def set_cooling_temp(self, target_temperature, quick=False, optional=False):
        if not self.supports_cooling:
            if not optional:
                raise NotImplementedError("Cooling not supported by this device")
            return

        return self._cooling_dispatch('_set_cooling_temp', target_temperature, quick=quick, optional=optional)

    def _set_cooling_temp_write_temp(self, target_temperature, quick=False, optional=False):
        self.setNumber("CCD_TEMPERATURE", target_temperature, quick=quick, optional=optional)

    def disable_cooling(self, quick=False, optional=False):
        if not self.supports_cooling:
            return

        return self._cooling_dispatch('_disable_cooling', quick=quick, optional=optional)

    def _disable_cooling_write_temp(self, quick=False, optional=False):
        self.setNarySwitch("CCD_COOLER", 1, quick=quick, optional=optional)

    _gain_control_index = None

    @property
    def gain(self):
        gain = self.prpoerties.get('CCD_GAIN')
        if gain:
            return gain[0]

        controls = self.properties.get('CCD_CONTROLS')
        if controls:
            if self._gain_control_index is not None:
                labels = self.getNumberLabels('CCD_CONTROLS')
                try:
                    self._gain_control_index = labels.index('gain')
                except ValueError:
                    return None

            try:
                return controls[self._gain_control_index]
            except IndexError:
                return None

    @gain.setter
    def gain(self, gain):
        self.set_gain(gain, quick=True)

    def set_gain(self, gain, quick=False, optional=False):
        if 'CCD_GAIN' in self.properties:
            # Some drivers have a CCD_GAIN property
            self.setNumber('CCD_GAIN', gain, quick=quick, optional=optional)
        elif 'CCD_CONTROLS' in self.properties:
            # Some other drivers have a CCD_CONTROLS with multiple settings
            self.setNumber('CCD_CONTROLS', {'gain': gain}, quick=quick, optional=optional)
        else:
            # If none is present, it may not have arrived yet. Use the stadard-ish CCD_GAIN.
            self.setNumber('CCD_GAIN', gain, quick=quick, optional=optional)

    _offset_control_index = None

    @property
    def offset(self):
        offset = self.prpoerties.get('CCD_OFFSET')
        if offset:
            return offset[0]

        controls = self.properties.get('CCD_CONTROLS')
        if controls:
            if self._offset_control_index is not None:
                labels = self.getNumberLabels('CCD_CONTROLS')
                try:
                    self._offset_control_index = labels.index('offset')
                except ValueError:
                    return None

            try:
                return controls[self._offset_control_index]
            except IndexError:
                return None

    @offset.setter
    def offset(self, offset):
        self.set_offset(offset, quick=True)

    def set_offset(self, offset, quick=False, optional=False):
        if 'CCD_OFFSET' in self.properties:
            self.setNumber('CCD_OFFSET', {'offset': offset})
        elif 'CCD_CONTROLS' in self.properties:
            # Some other drivers have a CCD_CONTROLS with multiple settings
            self.setNumber('CCD_CONTROLS', {'offset': offset})
        elif not optional:
            raise RuntimeError("Cannot find offset property")


class IndiCFW(IndiDevice):

    _cached_wheel_iface = None

    _maxpos_property = 'MAX_FILTER'
    _curpos_property = 'FILTER_SLOT'
    _filter_name_property = 'FILTER_NAME'

    @property
    def maxpos(self):
        mxp = self.properties.get(self._maxpos_property, (None,))[0]
        if mxp is not None:
            mxp = int(mxp)
        return mxp

    @maxpos.setter
    def maxpos(self, value):
        self.set_maxpos(value, quick=True)

    @property
    def curpos(self):
        pos = self.properties.get(self._curpos_property, (None,))[0]
        if pos is not None:
            pos = int(pos)
        return pos

    @curpos.setter
    def curpos(self, value):
        self.set_curpos(value, quick=True)

    @property
    def curfilter(self):
        pos = self.curpos
        if pos:
            return self.filter_names[self.curpos - 1]

    @property
    def filter_names(self):
        return self.properties.get(self._filter_name_property)

    def set_curpos(self, value, quick=False, optional=False, wait=False, timeout=None):
        self.setNumber(self._curpos_property, value, quick=quick, optional=optional)
        if wait:
            self.waitCondition(lambda:self.curpos == value, timeout=timeout)

    def set_maxpos(self, value, quick=False, optional=False):
        self.setNumber(self._maxpos_property, value, quick=quick, optional=optional)


class IndiST4(IndiDevice):

    _MONITOR_PROPS = [
        "TELESCOPE_TIMED_GUIDE_NS",
        "TELESCOPE_TIMED_GUIDE_WE",
    ]

    # Whether this particular driver reflects pulse guiding status in the pulse guiding properties
    _dynamic_pulse_updates = None

    _pulse_started = False

    def onPropertyChange(self, dname, pname, vp):
        if not self._pulse_started and self.pulse_in_progress:
            self._pulse_started = True

    def pulseNorth(self, ms):
        self._pulse_started = False
        self.setNumber("TELESCOPE_TIMED_GUIDE_NS", [ms, 0])

    def pulseSouth(self, ms):
        self._pulse_started = False
        self.setNumber("TELESCOPE_TIMED_GUIDE_NS", [0, ms])

    def pulseWest(self, ms):
        self._pulse_started = False
        self.setNumber("TELESCOPE_TIMED_GUIDE_WE", [ms, 0])

    def pulseEast(self, ms):
        self._pulse_started = False
        self.setNumber("TELESCOPE_TIMED_GUIDE_WE", [0, ms])

    def pulseGuide(self, n_ms, w_ms):
        if n_ms > 0:
            self.pulseNorth(n_ms)
        elif n_ms < 0:
            self.pulseSouth(-n_ms)
        if w_ms > 0:
            self.pulseWest(w_ms)
        elif w_ms < 0:
            self.pulseEast(-w_ms)

    @property
    def pulse_in_progress(self):
        nss = self.property_meta.get("TELESCOPE_TIMED_GUIDE_NS")
        wes = self.property_meta.get("TELESCOPE_TIMED_GUIDE_WE")
        return (
            (nss is not None and nss.get('state') == PyIndi.IPS_BUSY)
            or (wes is not None and wes.get('state') == PyIndi.IPS_BUSY)
        )

    def waitPulseDone(self, timeout):
        self.waitCondition(lambda:self._pulse_started and not self.pulse_in_progress, timeout=timeout)


class IndiFocuser(IndiDevice):

    _MONITOR_PROPS = [
        "ABS_FOCUS_POSITION",
        "REL_FOCUS_POSITION",
    ]

    _move_started = False

    INWARD = "FOCUS_INWARD"
    OUTWARD = "FOCUS_OUTWARD"

    def onPropertyChange(self, dname, pname, vp):
        if not self._move_started and self.move_in_progress:
            self._move_started = True

    @property
    def absolute_position(self):
        return self.properties.get("ABS_FOCUS_POSITION", [None])[0]

    @absolute_position.setter
    def absolute_position(self, value):
        self.setAbsolutePosition(value, quick=True)

    def setAbsolutePosition(self, value, **kw):
        self._move_started = False
        self.setNumber("ABS_FOCUS_POSITION", [value], **kw)

    @property
    def move_direction(self):
        return self.getNarySwitchByLabel("FOCUS_MOTION", quick=True, optional=True)

    def setMoveDirection(self, value, **kw):
        self.setNarySwitch("FOCUS_MOTION", value, **kw)

    def sync(self, value, **kw):
        self.setNumber("FOCUS_SYNC", value, **kw)

    def moveRelative(self, value, **kw):
        self._move_started = False
        if value < 0:
            self.setMoveDirection(self.INWARD, **kw)
            value = -value
        elif value > 0:
            self.setMoveDirection(self.OUTWARD, **kw)
        if value:
            self.setNumber("REL_FOCUS_POSITION", [value], **kw)

    @property
    def move_in_progress(self):
        apos = self.property_meta.get("ABS_FOCUS_POSITION")
        rpos = self.property_meta.get("REL_FOCUS_POSITION")
        return (
            (apos is not None and apos.get('state') == PyIndi.IPS_BUSY)
            or (rpos is not None and rpos.get('state') == PyIndi.IPS_BUSY)
        )

    def waitMoveDone(self, timeout):
        return self.waitCondition(lambda:self._move_started and not self.move_in_progress, timeout=timeout)


class IndiTelescope(IndiDevice):

    SLEW_MODE_SLEW = 0
    SLEW_MODE_TRACK = 1
    SLEW_MODE_SYNC = 2

    COORD_EOD = "EQUATORIAL_EOD_COORD"
    COORD_J2000 = "EQUATORIAL_COORD"
    TARGET_EOD = "TARGET_EOD_COORD"

    @property
    def default_guide_flip(self):
        return driver_info.get_guide_flip(self.name, self.properties.get('DRIVER_INFO'))

    def setCoordMode(self, mode):
        self.setNarySwitch("ON_COORD_SET", mode)

    def coordTo(self, ra, dec, which=COORD_EOD):
        self.setNumber(which, [ra, dec])

    def trackTo(self, *p, **kw):
        self.setCoordMode(self.SLEW_MODE_TRACK)
        self.coordTo(*p, **kw)

    def slewTo(self, *p, **kw):
        self.setCoordMode(self.SLEW_MODE_SLEW)
        self.coordTo(*p, **kw)

    def syncTo(self, *p, **kw):
        self.setCoordMode(self.SLEW_MODE_SYNC)
        self.coordTo(*p, **kw)

    def waitSlew(self, tolerance=0.5, timeout=None):
        if timeout is None:
            timeout = self.client.DEFAULT_TIMEOUT
        deadline = time.time() + timeout
        while time.time() < deadline:
            target = self.properties[self.TARGET_EOD]
            cur = self.properties[self.COORD_EOD]
            if abs(target[0] - cur[0]) < tolerance and abs(target[1] - cur[1]) < tolerance:
                break
            time.sleep(1)

    def startTracking(self, which=COORD_EOD):
        self.setCoordMode(self.SLEW_MODE_TRACK)
        ra, dec = self.waitNumber(which)
        self.coordTo(ra, dec, which)

    def stopTracking(self, which=COORD_EOD):
        self.setCoordMode(self.SLEW_MODE_SLEW)
        ra, dec = self.waitNumber(which)
        self.coordTo(ra, dec, which)


class IndiClient(PyIndi.BaseClient):

    DEFAULT_HOST = 'localhost'
    DEFAULT_PORT = 7624

    DEFAULT_TIMEOUT = 600

    def __init__(self):
        self.devices = {}
        self.properties = collections.defaultdict(dict)
        self.property_meta = collections.defaultdict(dict)
        self.blob_listeners = collections.defaultdict(dict)
        self.prop_listeners = collections.defaultdict(functools.partial(collections.defaultdict, list))

        self.connection_event = threading.Event()
        self.property_event = threading.Event()
        self.blob_event = threading.Event()
        self.device_event = threading.Event()
        self.any_event = threading.Event()
        self.socketlock = threading.Lock()

        self._reconnect = False
        self._watchdog_thread = None
        self._autoreconnect = []

        super(IndiClient, self).__init__()

    def waitCCD(self, device_name):
        return self._waitWrappedDevice(device_name, IndiCCD)

    def waitST4(self, device_name):
        return self._waitWrappedDevice(device_name, IndiST4)

    def waitTelescope(self, device_name):
        return self._waitWrappedDevice(device_name, IndiTelescope)

    def waitCFW(self, device_name):
        return self._waitWrappedDevice(device_name, IndiCFW)

    def waitFocuser(self, device_name):
        return self._waitWrappedDevice(device_name, IndiFocuser)

    def waitDevice(self, device_name):
        return self._waitWrappedDevice(device_name, IndiDevice)

    def _waitWrappedDevice(self, device_name, device_class):
        d = self._waitDevice(device_name)
        if d is not None:
            d = device_class(self, d)
        return d

    def _waitDevice(self, device_name):
        dev = None
        wait = False
        deadline = time.time() + self.DEFAULT_TIMEOUT
        while dev is None and time.time() < deadline:
            if wait:
                self.device_event.wait(10)
                self.device_event.clear()
            else:
                wait = True
            with self.socketlock:
                dev = self.getDevice(device_name)
                if not dev.isValid():
                    dev = None
        return dev

    def newDevice(self, d):
        logger.info("New device: %r", d.getDeviceName())
        self.devices[d.getDeviceName()] = None
        self.device_event.set()
        self.any_event.set()

    def newProperty(self, p):
        ptype = p.getType()
        if ptype == PyIndi.INDI_NUMBER:
            vp = p.getNumber()
            val = [ np.value for np in vp ]
        elif ptype == PyIndi.INDI_TEXT:
            vp = p.getText()
            val = [ tp.text for tp in vp ]
        elif ptype == PyIndi.INDI_SWITCH:
            vp = p.getSwitch()
            val = [ sp.s for sp in vp ]
        else:
            vp = None
            val = 'unk'

        dname = p.getDeviceName()
        pname = p.getName()
        self.properties[dname][pname] = val
        pmeta = self.property_meta[dname].setdefault(pname, {})
        pmeta.update(dict(
            state=p.getState(),
            label=p.getLabel(),
            group=p.getGroupName(),
            type=ptype,
            perm=p.getPermission(),
        ))
        if vp is not None:
            pmeta['vmeta'] = [
                dict(
                    label=subp.label,
                    name=subp.name,
                )
                for subp in vp
            ]
        self.property_event.set()
        self.any_event.set()

    def removeProperty(self, p):
        dname = p.getDeviceName()
        pname = p.getName()
        props = self.properties[dname]
        props.pop(pname, None)
        if not props:
            self.properties.pop(dname, None)
        pmeta = self.property_meta[dname]
        pmeta.pop(pname, None)
        if not pmeta:
            self.property_meta.pop(dname, None)
        self.property_event.set()
        self.any_event.set()

    def listenBLOB(self, device_name, ccd_name, callback):
        with self.socketlock:
            self.setBLOBMode(PyIndi.B_ALSO, device_name, ccd_name)
        self.blob_listeners[device_name][ccd_name] = callback

    def unlistenBLOB(self, device_name, ccd_name):
        self.blob_listeners[device_name].pop(ccd_name, None)
        with self.socketlock:
            self.setBLOBMode(PyIndi.B_NEVER, device_name, ccd_name)

    def listenProperty(self, device_name, prop_name, callback):
        self.prop_listeners[device_name][prop_name].append(callback)

    def unlistenProperty(self, device_name, prop_name, callback):
        listeners = self.prop_listeners[device_name][prop_name]
        if callback in listeners:
            listeners.remove(callback)

    def _invokePropertyCallbacks(self, device_name, prop_name, *p, **kw):
        listeners = self.prop_listeners.get(device_name, {}).get(prop_name)
        if listeners:
            dead = []
            for callback in listeners:
                try:
                    if not callback(device_name, prop_name, *p, **kw):
                        dead.append(callback)
                except Exception:
                    logger.exception("Error invoking property callback")
            for callback in dead:
                if callback in listeners:
                    try:
                        listeners.remove(callback)
                    except ValueError:
                        pass

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
        pmeta = self.property_meta[svp.device].setdefault(svp.name, {})
        pmeta['state'] = svp.s
        pmeta['perm'] = svp.p
        self._invokePropertyCallbacks(svp.device, svp.name, svp)
        self.property_event.set()
        self.any_event.set()

    def newNumber(self, nvp):
        val = [ np.value for np in nvp ]
        self.properties[nvp.device][nvp.name] = val
        pmeta = self.property_meta[nvp.device].setdefault(nvp.name, {})
        pmeta['state'] = nvp.s
        pmeta['perm'] = nvp.p
        self._invokePropertyCallbacks(nvp.device, nvp.name, nvp)
        self.property_event.set()
        self.any_event.set()

    def newText(self, tvp):
        val = [ tp.text for tp in tvp ]
        self.properties[tvp.device][tvp.name] = val
        pmeta = self.property_meta[tvp.device].setdefault(tvp.name, {})
        pmeta['state'] = tvp.s
        pmeta['perm'] = tvp.p
        self._invokePropertyCallbacks(tvp.device, tvp.name, tvp)
        self.property_event.set()
        self.any_event.set()

    def newLight(self, lvp):
        self.any_event.set()

    def newMessage(self, d, m):
        self.any_event.set()

    def serverConnected(self):
        logger.info("INDI server connected")
        self._reconnect = False
        self.connection_event.set()
        self.any_event.set()

    def serverDisconnected(self, code):
        logger.info("INDI server disconnected")
        self._reconnect = True
        self.connection_event.set()
        self.any_event.set()

    def startWatchdog(self):
        if self._watchdog_thread is None:
            self._stop = False
            self._watchdog_thread = watchdog_thread = threading.Thread(target=self._watchdog)
            watchdog_thread.daemon = True
            watchdog_thread.start()

    def stopWatchdog(self, timeout = DEFAULT_TIMEOUT*2):
        if self._watchdog_thread is not None:
            self._stop = True
            self._watchdog_thread.join(timeout)
            self._watchdog_thread = None

    def autoReconnect(self, device):
        self._autoreconnect.append(device)

    def _watchdog(self):
        while not self._stop:
            self.connection_event.wait(5)
            self.connection_event.clear()
            if self._reconnect:
                try:
                    self.connectServer()
                    for device in self._autoreconnect:
                        device.d = self._waitDevice(device.d.getDeviceName())
                        device.connect()
                        device.onReconnect()
                except Exception:
                    logger.exception("Can't reconnect, will retry later")
                else:
                    self._reconnect = False

    def sendNewSwitch(self, *p, **kw):
        with self.socketlock:
            super(IndiClient, self).sendNewSwitch(*p, **kw)

    def sendNewNumber(self, *p, **kw):
        with self.socketlock:
            super(IndiClient, self).sendNewNumber(*p, **kw)

    def sendNewText(self, *p, **kw):
        with self.socketlock:
            super(IndiClient, self).sendNewText(*p, **kw)
