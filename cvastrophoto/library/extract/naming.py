# -*- coding: utf-8 -*-
from __future__ import absolute_import

from past.builtins import basestring
import os.path
import re


class NamingConventionTagExtractor(object):

    conventions = [
        re.compile(r'^(?P<FRAME>light|dark|flat|bias)(?:_(?P<FILTER>[a-z]+))?_[0-9]+\.(?:fits?|fits?\.gz|tiff?|cr2|nef)$', re.I),
    ]

    def clear_cache(self):
        pass

    def get_tags(self, img_path):
        if not isinstance(img_path, basestring):
            return
        for convention in self.conventions:
            filename = os.path.basename(img_path)
            m = convention.match(filename)
            if m:
                tags = m.groupdict()
                if 'FRAME' in tags:
                    tags['FRAME'] = tags['FRAME'].title()
                return tags
