# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re

from astropy.io import fits


class FitsClassificationMixIn(object):

    unsafe_re = re.compile(r'[^a-zA-Z0-9\s.]+')

    classification_tags = None

    def classify_frame(self, img_path):

        tags = getattr(img_path, 'fits_header', None)
        if tags is None:
            hdul = fits.open(img_path)
            tags = hdul[0].header
            hdul.close()

        return tuple([
            ','.join(map(self.escape_tag, map(tags.get, stags)))
            for stags in self.classification_tags
        ])

    def escape_tag(self, tag):
        if tag is None:
            return 'NA'

        return self.unsafe_re.sub('_', tag)
