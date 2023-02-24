from __future__ import absolute_import

import unittest
import tempfile
import numpy
import numpy.random

from cvastrophoto.image.metaimage import MetaImage
from cvastrophoto.image import ImageAccumulator


class MetaImageTest(unittest.TestCase):

    def assertImMatches(self, im1, im2):
        self.assertEqual(im1 is None, im2 is None)
        if im1 is not None and im2 is not None:
            self.assertEqual(im1.shape, im2.shape)
            self.assertEqual(im1.dtype.char, im2.dtype.char)
            self.assertTrue(
                numpy.all(im1 == im2),
                "Images don't match:\n\tim1 = %s\n\tim2 = %s" % (im1, im2))

    def assertSavesOk(self, mim, **kw):
        with tempfile.TemporaryFile() as f:
            mim.save(f, **kw)
            f.seek(0)
            mim2 = MetaImage(f)
            self.assertImMatches(mim.light_data, mim2.light_data)
            self.assertImMatches(mim.weights_data, mim2.weights_data)
            self.assertImMatches(mim.weighted_light_data, mim2.weighted_light_data)
            self.assertImMatches(mim.weighted_light2_data, mim2.weighted_light2_data)
        return mim2

    def test_simple_mono(self):
        ldata = numpy.zeros((32, 32), 'H')
        mim = MetaImage(light=ImageAccumulator(data=ldata))
        self.assertSavesOk(mim)

    def test_simple_bayer(self):
        ldata = numpy.zeros((32, 32), 'H')
        mim = MetaImage(light=ImageAccumulator(data=ldata))
        mim2 = self.assertSavesOk(mim, raw_pattern=numpy.array([[0,1],[1,2]]))
        self.assertEqual('RGGB', mim2.fits_header.get('BAYERPAT'))
        self.assertEqual(2, mim2.fits_header.get('BAYERSZ1'))
        self.assertEqual(2, mim2.fits_header.get('BAYERSZ2'))

    def test_simple_rgb(self):
        ldata = numpy.zeros((32, 32, 3), 'H')
        mim = MetaImage(light=ImageAccumulator(data=ldata))
        mim2 = self.assertSavesOk(mim)
        self.assertEqual(3, mim2.fits_header.get('NAXIS'))

    def test_simple_weighted_mono(self):
        ldata = numpy.zeros((32, 32), 'H')
        wdata = numpy.ones_like(ldata, dtype='f')
        mim = MetaImage(
            light=ImageAccumulator(data=ldata),
            weights=ImageAccumulator(data=wdata))
        self.assertSavesOk(mim)
        self.assertImMatches(ldata, mim.mainimage)

    def test_simple_weighted_light_mono(self):
        refimg = numpy.ones((32, 32), 'f')
        wdata = numpy.random.uniform(10, 20, refimg.shape).astype('f')
        ldata = refimg * wdata
        mim = MetaImage(
            weighted_light=ImageAccumulator(data=ldata),
            weights=ImageAccumulator(data=wdata))
        self.assertSavesOk(mim)
        self.assertImMatches(refimg, mim.mainimage)
