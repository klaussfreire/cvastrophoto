from __future__ import absolute_import

import numpy
import astropy.io.fits
import six

from .base import ImageAccumulator
from . import fits
from cvastrophoto.util.arrays import asnative


if six.PY3:
    _RGB = numpy.array(list('RGB'), dtype='U1')
else:
    _RGB = numpy.array('RGB', dtype='c')


class MetaImage(object):

    def __init__(self, path=None, fits_header=None, mode=None, **parts):
        """ Represents an image with extra attributes

        Implements a dict-like interface wehre each key represents an attribute,
        and the value will be an accumulator that holds per-pixel values for
        that attribute (an instance of ImageAccumulator).

        The backing format is a multi-part FITS file, with special metadata
        to hold accumulator counters and part names, but otherwise a standard FITS file.

        Standard attributes:

            light: average accumulated light data
            weighted_light: accumulated pre-weighted light data
            weights: weights associated to light/weighted_light
            light2: average light-square data
            weighted_light2: pre-weighted light-square data

        Special attributes:

            main: picks between either 'light' or 'weighted_light', whichever
                is present.
        """
        self._accumulators = None
        self._fits_header = None
        self.name = path
        self.main = None

        if path is not None:
            self.open(path, mode=mode)
        elif parts:
            self.main = None
            self._accumulators = parts
            self._fits_header = fits_header or {}

    @classmethod
    def is_metaimage(cls, img):
        if isinstance(img, MetaImage):
            return True
        elif isinstance(img, fits.Fits) and img.name and img.fits_header.get('CVMIKEY'):
            return True
        return False

    @classmethod
    def get_metaimage(cls, img):
        if cls.is_metaimage(img):
            if isinstance(img, MetaImage):
                return img
            elif img.name is not None:
                return cls(img.name)

    def save(self, path=None, **kw):
        self._save(path or self.name, **kw)

    def open(self, path, **kw):
        self.name = path
        self.main = fits.Fits(path, **kw)
        self._fits_header = None

    def close(self):
        if self.main is not None:
            self.main.close()
            self._accumulators = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def fits_header(self):
        if self._fits_header is not None:
            return self._fits_header
        else:
            return self.main.fits_header

    @fits_header.setter
    def fits_header(self, value):
        self._fits_header = value

    @property
    def dark_calibrated(self):
        return self.fits_header.get('CVMIDKAP', False)

    @dark_calibrated.setter
    def dark_calibrated(self, value):
        self.fits_header['CVMIDKAP'] = value

    def set(self, **attrs):
        for attn, attval in attrs.items():
            setattr(self, attn, attval)
        return self

    def _save(self, path, overwrite=True, raw_pattern=None):
        assert self._accumulators is not None

        acc = self._accumulators

        for k in ('light', 'weighted_light'):
            main = acc.get(k)
            if main is not None:
                which_main = k
                break
        else:
            raise ValueError("MetaImage needs one of light or weighted_light at least")

        header = astropy.io.fits.Header()
        if self._fits_header:
            header.update(self._fits_header)
        header['CVMIKEY'] = which_main
        header['CVMINUM'] = main.num_images

        if raw_pattern is not None:
            if raw_pattern.max() <= 3:
                header['BAYERPAT'] = ''.join(_RGB[raw_pattern].flatten())
                header['BAYERSZ1'] = raw_pattern.shape[0]
                header['BAYERSZ2'] = raw_pattern.shape[1]

        primary_hdu = astropy.io.fits.PrimaryHDU(main.accum, header=header)
        hdul = [primary_hdu]

        for k, part in acc.items():
            if part is None or not part.num_images or k == which_main:
                continue
            kheader = astropy.io.fits.Header()
            kheader['CVMIKEY'] = k
            kheader['CVMINUM'] = part.num_images
            part_data = part.accum
            if not part_data.shape:
                # FITS can't handle scalars, but 1x1 broadcasts nicely
                part_data = part_data.reshape((1, 1))
            hdul.append(astropy.io.fits.ImageHDU(part_data, header=kheader))
            del part_data

        hdul = astropy.io.fits.HDUList(hdul)
        hdul.writeto(path, overwrite=overwrite)

        if self.name is None:
            self.name = path

    @property
    def hdul(self):
        return self.main.rimg.hdul

    @property
    def rimg(self):
        return self.main.rimg

    @property
    def mainimage(self):
        light = self.get('light')
        weights = self.get('weights')
        if light is None and weights is not None:
            light = self['weighted_light'].accum
            light = numpy.divide(light, weights.accum, where=weights.accum > 0)
        else:
            light = light.average
        return light

    @property
    def mainaccum(self):
        light = self.get('light')
        weights = self.get('weights')
        if light is None and weights is not None:
            light = self['weighted_light']
        return light

    @property
    def weights(self):
        return self.get('weights')

    @property
    def weights_data(self):
        return self.get('weights', getdata=True)

    @property
    def light(self):
        return self.get('light')

    @property
    def light_data(self):
        return self.get('light', getdata=True)

    @property
    def weighted_light(self):
        return self.get('weighted_light')

    @property
    def weighted_light_data(self):
        return self.get('weighted_light', getdata=True)

    @property
    def weighted_light2(self):
        return self.get('weighted_light2')

    @property
    def weighted_light2_data(self):
        return self.get('weighted_light2', getdata=True)

    @property
    def var_data(self):
        var = self.get('var', getdata=True)
        if var is not None:
            return var

        w = self.weights_data
        var = self.get('weighted_var', getdata=True)
        if var is not None:
            var = numpy.true_divide(var, w, where=w > 0)
            return var

        l2 = self.weighted_light2_data
        if l2 is None:
            return None

        l = self.mainimage
        if l is not None and l2 is not None and w is not None:
            l2 = numpy.true_divide(l2, w, where=w > 0)
            l2 -= numpy.square(l)
            num_images = self.weighted_light2.num_images
            if num_images >= 2:
                l2 *= float(num_images) / float(num_images - 1)
            return l2

    @property
    def std_data(self):
        var_data = self.var_data
        if var_data is not None:
            return numpy.sqrt(numpy.abs(var_data))

    @property
    def accumulators(self):
        if self._accumulators is None:
            self._accumulators = {
                hdu.header['CVMIKEY'] : ImageAccumulator(data=asnative(hdu.data), num=int(hdu.header['CVMINUM']))
                for hdu in self.hdul
            }
        return self._accumulators

    def get(self, key, deflt=None, getdata=False):
        rv = self.accumulators.get(key, deflt)
        if rv is not deflt and getdata:
            rv = rv.accum
        return rv

    def __getitem__(self, key):
        return self.accumulators[key]

    def __setitem__(self, key, newdata):
        self.accumulators[key] = newdata

    def __contains__(self, key):
        return key in self.accumulators
