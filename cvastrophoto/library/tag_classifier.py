# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re

from .extract import exif, fits, naming


class TagClassificationMixIn(object):

    unsafe_re = re.compile(r'[^a-zA-Z0-9\s.]+')

    classification_tags = None
    exiftool_binary = 'exiftool'

    extractor_classes = [
        [naming.NamingConventionTagExtractor],
        [
            fits.FitsTagExtractor,
            exif.ExifTagExtractor,
        ],
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
            self._extractors = extractors = [
                [cls() for cls in clsses]
                for clsses in self.extractor_classes
            ]
        return extractors

    def clear_cache(self):
        extractors = self._extractors
        if extractors:
            for sextractors in extractors:
                for extractor in sextractors:
                    extractor.clear_cache()

    def get_tags(self, img_path):
        final_tags = None
        for sextractors in self.extractors:
            for extractor in sextractors:
                tags = extractor.get_tags(img_path)
                if tags:
                    break
            if final_tags is None:
                final_tags = tags
            else:
                final_tags.update(tags)
        return final_tags

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
