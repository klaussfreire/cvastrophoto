# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re

from .extract import exif, fits


class TagClassificationMixIn(object):

    unsafe_re = re.compile(r'[^a-zA-Z0-9\s.]+')

    classification_tags = None
    exiftool_binary = 'exiftool'

    extractor_classes = [
        fits.FitsTagExtractor,
        exif.ExifTagExtractor,
    ]
    _extractors = None

    classification_tags = [
        ('Make',),
        (('Model', 'INSTRUME'),),
        ('InternalSerialNumber', 'SerialNumber'),
        (('ImageSize', 'NAXIS'), ('ExifImageWidth', 'NAXIS1'), ('ExifImageHeight', 'NAXIS2')),
        (
            'SensorWidth', 'SensorHeight',
            ('SensorLeftBorder', 'XORFSUBF'), ('SensorTopBorder', 'YORGSUBF'),
            'SensorRightBorder', 'SensorBottomBorder',
            ('PhotometricInterpretation', 'COLORSPC',),

            # Optional, truncated if empty
            'BINNING', 'XBINNING', 'YBINNING',
            'BAYERPAT',
        ),
        (('ISO', 'GAIN'),),
        (('ExposureTime', 'EXPTIME'), ('BulbDuration', 'EXPOSURE')),
    ]

    @property
    def extractors(self):
        extractors = self._extractors
        if extractors is None:
            self._extractors = extractors = [cls() for cls in self.extractor_classes]
        return extractors

    def get_tags(self, img_path):
        tags = None
        for extractor in self.extractors:
            tags = extractor.get_tags(img_path)
            if tags:
                break
        return tags

    def classify_frame(self, img_path):
        tags = self.get_tags(img_path)
        def tag_get(tagname):
            if isinstance(tagname, tuple):
                for tag in tagname:
                    rv = tags.get(tag)
                    if rv:
                        return rv
                else:
                    return None
            else:
                return tags.get(tagname)
        return tuple([
            ','.join(map(self.escape_tag, map(tag_get, stags)))
            for stags in self.classification_tags
        ])

    def escape_tag(self, tag):
        if tag is None:
            return 'NA'

        return self.unsafe_re.sub('_', tag)
