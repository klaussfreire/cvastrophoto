# -*- coding: utf-8 -*-
from __future__ import division

import avifilelib
import numpy
import threading

from . import rgb


DATA_TYPES = {
    8: numpy.uint8,
    16: numpy.uint16,
    24: numpy.uint8,
    32: numpy.uint32,
    48: numpy.uint16,
    96: numpy.uint32
}

DEFAULT_LINEAR = {
    8: False,
    16: False,
    24: False,
    32: False,
    48: False,
    96: False,
}

DEFAULT_AUTOSCALE = {
    8: True,
    16: False,
    24: True,
    32: False,
    48: False,
    96: False,
}

CHANNELS = {
    8: 1,
    16: 1,
    24: 3,
    32: 1,
    48: 3,
    96: 3,
}


class NoLock(object):

    def __enter__(self):
        return

    def __exit__(self, ext_type, exc_value, traceback):
        pass

nolock = NoLock()


class AVI(rgb.RGB):

    priority = 2
    concrete = True

    def _open_impl(self, path, avifile=None, aviframe=None, loadlock=None):
        if avifile is None:
            avifile = self._kw.get('avifile')
        if avifile is None and path is not None:
            avifile = avifilelib.AviFile(path)

        if aviframe is None:
            aviframe = self._kw.get('aviframe')
        if aviframe is None and avifile is not None:
            aviframe = avifile.movi.data_chunks[0]

        if loadlock is None:
            loadlock = self._kw.get('loadlock')
        if loadlock is None:
            loadlock = nolock

        bpp = aviframe.size * 8 // (avifile.avih.width * avifile.avih.height)
        dtype = DATA_TYPES[bpp]
        channels = CHANNELS[bpp]
        shape = (avifile.avih.height, avifile.avih.width)
        if channels > 1:
            shape += (channels,)

        with loadlock:
            aviframe.seek(0)
            img = numpy.frombuffer(aviframe.read(), dtype=dtype).reshape(shape)

        return super(AVI, self)._open_impl(path, img=img, copy=True)

    def all_frames(self):
        path = self.name
        avifile = avifilelib.AviFile(path)
        loadlock = threading.Lock()
        for i, aviframe in enumerate(avifile.movi.data_chunks):
            # Pass a shared lock to avoid race conditions when reading from the
            # shared file object
            bpp = aviframe.size * 8 // (avifile.avih.width * avifile.avih.height)
            frame = type(self)(
                path,
                default_pool=self.default_pool,
                avifile=avifile, aviframe=aviframe, loadlock=loadlock,
                linear=self._kw.get('linear', DEFAULT_LINEAR.get(bpp)),
                autoscale=self._kw.get('autoscale', DEFAULT_AUTOSCALE.get(bpp)))
            frame.name += '#%d' % i
            yield frame

    @classmethod
    def supports(cls, path):
        return path.rsplit('.', 1)[-1].lower() in ('avi',)
