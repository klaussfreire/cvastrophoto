from __future__ import absolute_import

import numpy
import astropy.io.fits

from .base import ImageAccumulator
from . import fits
from cvastrophoto.util.arrays import asnative


class MetaImage(object):

    def __init__(self, path=None, fits_header=None, **parts):
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
            self.open(path)
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

    def save(self, path=None):
        self._save(path or self.name)

    def open(self, path):
        self.name = path
        self.main = fits.Fits(path)
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

    def _save(self, path):
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
        primary_hdu = astropy.io.fits.PrimaryHDU(main.accum, header=header)
        hdul = [primary_hdu]

        for k, part in acc.items():
            if part is None or k == which_main:
                continue
            kheader = astropy.io.fits.Header()
            kheader['CVMIKEY'] = k
            kheader['CVMINUM'] = part.num_images
            hdul.append(astropy.io.fits.ImageHDU(part.accum, header=kheader))

        hdul = astropy.io.fits.HDUList(hdul)
        hdul.writeto(path)

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
