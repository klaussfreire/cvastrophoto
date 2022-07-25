from __future__ import absolute_import

import re
import datetime
import dateutil.parser

from .base import CollectionSortingBase
from cvastrophoto.image import Image
from cvastrophoto.library import tag_classifier


class CollectionTagSorting(tag_classifier.TagClassificationMixIn, CollectionSortingBase):

    unsafe_re = re.compile(r'[^-a-zA-Z0-9\s.]+')

    classification_tags = [
        (('OBJECT',),),
        (('DATEOBS', 'DATE-OBS',),),
        (
            ('RA', 'OBJCTRA'),
            ('DEC', 'OBJCTDEC'),
        ),
        (
            ('LAT-OBS', 'OBSGEO-B'),
            ('LONG-OBS', 'OBSGEO-L'),
            ('ALT-OBS', 'OBSGEO-H'),
        ),
        (('FILTER',),),
        (('FRAME',),),
    ]

    classification_semantic = [
        'OBJECT',
        'DATEOBS',
        'OBJCOORDS',
        'OBSCOORDS',
        'FILTER',
        'FRAME',
    ]

    def classify_frame(self, img_path):
        tags = list(super(CollectionTagSorting, self).classify_frame(img_path))

        tags[1] = self.session_from_dateobs(tags[1])
        return tuple(tags)

    def _reclassify_frame(self, img_path, new, cur):
        for tags, new_val, cur_val in zip(self.classification_tags, new, cur):
            if new_val != cur_val and len(tags) == 1:
                tag = tags[0][0]
                with Image.open(img_path) as img:
                    supports_update = img.supports_inplace_update
                if supports_update:
                    with Image.open(img_path, mode='update') as img:
                        img.fits_header[tag] = new_val

    def session_from_dateobs(self, dateobs):
        if dateobs and dateobs != 'NA':
            # Parse the obstime with timezone, add 12h to make the "day" start at 12pm
            # and extract the date. This mostly sorts subs by capture session.
            dateobs = dateobs.replace('_', ':')
            dateobs = dateutil.parser.isoparse(dateobs)
            dateobs += datetime.timedelta(hours=12)
            dateobs = dateobs.date()
            dateobs = dateobs.isoformat()
        return dateobs
