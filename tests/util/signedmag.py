from __future__ import absolute_import

import unittest

from cvastrophoto.util import signedmag


class SignedMagTest(unittest.TestCase):

    def testMinDirected(self):
        self.assertEqual(signedmag.min_directed(1, -1), 0)
        self.assertEqual(signedmag.min_directed(1, 1), 1)
        self.assertEqual(signedmag.min_directed(1, 2), 1)
        self.assertEqual(signedmag.min_directed(1, 0.5), 0.5)
        self.assertEqual(signedmag.min_directed(-1, 1), 0)
        self.assertEqual(signedmag.min_directed(-1, -1), -1)
        self.assertEqual(signedmag.min_directed(-1, -2), -1)
        self.assertEqual(signedmag.min_directed(-1, -0.5), -0.5)
        self.assertEqual(signedmag.min_directed(-1, 0), 0)
        self.assertEqual(signedmag.min_directed(1, 0), 0)
        self.assertEqual(signedmag.min_directed(0, -1), 0)
        self.assertEqual(signedmag.min_directed(0, 1), 0)
