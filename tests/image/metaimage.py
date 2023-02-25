from __future__ import absolute_import

import unittest
import numpy
import tempfile
import os.path

from cvastrophoto.image import ImageAccumulator
from cvastrophoto.image.metaimage import MetaImage


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
