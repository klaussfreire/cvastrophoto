from __future__ import absolute_import

import unittest
import tempfile
import numpy
import numpy.random
import os.path

from cvastrophoto.image.metaimage import MetaImage
from cvastrophoto.image import ImageAccumulator


class MetaImageTest(unittest.TestCase):

    SHAPE = (10, 10)

    LIGHT = numpy.ones(SHAPE, dtype='H')
    WEIGHTS = numpy.ones(SHAPE, dtype='f')

    def setUp(self):
        self.path = tempfile.mktemp()

    def tearDown(self):
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
        except OSError:
            pass

    def testLightOnly(self):
        mi = MetaImage(light=ImageAccumulator(data=self.LIGHT))
        mi.save(self.path)
        mi.close()

        mi = MetaImage(self.path)
        self.assertTrue((self.LIGHT == mi.mainimage).all())

    def testSetHeaderAfterCreate(self):
        mi = MetaImage(light=ImageAccumulator(data=self.LIGHT))
        mi.dark_calibrated = True
        mi.save(self.path)
        mi.close()

        mi = MetaImage(self.path)
        self.assertTrue((self.LIGHT == mi.mainimage).all())
        self.assertTrue(mi.dark_calibrated)

    def testReopenAfterCreate(self):
        # Immediately usable after creation
        mi = MetaImage(light=ImageAccumulator(data=self.LIGHT))
        self.assertTrue((self.LIGHT == mi.mainimage).all())

        # Save does not change mi
        mi.save(self.path)
        self.assertTrue((self.LIGHT == mi.mainimage).all())

        # Close doesn't dereference accumulators if not backed by a file
        mi.close()
        self.assertTrue((self.LIGHT == mi.mainimage).all())

        # Usable after open, close dereferences bur auto-reopens
        mi.open(self.path)
        self.assertTrue((self.LIGHT == mi.mainimage).all())
        oldlight = mi.get('light')
        mi.close()
        newlight = mi.get('light')
        self.assertIsNot(newlight, oldlight)
        self.assertTrue((self.LIGHT == mi.mainimage).all())

    def testReopenAfterClose(self):
        mi = MetaImage(light=ImageAccumulator(data=self.LIGHT))
        mi.save(self.path)

        mi = MetaImage(self.path)
        oldlight = mi.get('light')
        mi.close()
        newlight = mi.get('light')
        self.assertIsNot(newlight, oldlight)
        self.assertTrue((self.LIGHT == mi.mainimage).all())

    def testLightWithWeights(self):
        mi = MetaImage(
            light=ImageAccumulator(data=self.LIGHT * 2),
            weights=ImageAccumulator(data=self.WEIGHTS),
        )
        mi.save(self.path)

        mi = MetaImage(self.path)
        self.assertTrue(((self.LIGHT * 2) == mi.light.accum).all())
        self.assertTrue((self.WEIGHTS == mi.weights.accum).all())

    def testWeightedLight(self):
        mi = MetaImage(
            weighted_light=ImageAccumulator(data=self.LIGHT * 2),
            weights=ImageAccumulator(data=self.WEIGHTS * 2),
        )
        mi.save(self.path)

        mi = MetaImage(self.path)
        self.assertTrue((self.LIGHT == mi.mainimage).all())

    def testManyLights(self):
        mi = MetaImage(
            light=ImageAccumulator(data=self.LIGHT.copy(), mpy=10),
        )
        mi.save(self.path)

        mi = MetaImage(self.path)
        self.assertTrue((self.LIGHT == mi.mainimage).all())


class MultichannelMetaImageTest(unittest.TestCase):

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
            mim2 = MetaImage(f, mode='update')
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
