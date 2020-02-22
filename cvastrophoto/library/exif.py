# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os.path
import subprocess
import re
import logging


class ExifClassificationMixIn(object):

    unsafe_re = re.compile(r'[^a-zA-Z0-9\s.]+')

    classification_tags = None
    exiftool_binary = 'exiftool'

    classification_cache = None

    def classify_frame(self, img_path):
        tags = getattr(img_path, 'exif_tags', None)
        classification_cache = None

        if not isinstance(img_path, basestring) and hasattr(img_path, 'name'):
            img_path = getattr(img_path, 'name', None)

        if isinstance(img_path, basestring):
            if not os.path.exists(img_path) and '#' in img_path:
                # Subframes are suffixed with #N, remove
                img_path, frameno = img_path.rsplit('#', 1)

        if tags is None:
            if img_path is None:
                tags = None
            else:
                classification_cache = self.classification_cache
                if classification_cache is None:
                    self.classification_cache = classification_cache = {}
                tags = classification_cache.get(img_path)

        if tags is None and isinstance(img_path, basestring) and os.path.exists(img_path):
            stdout, stderr = subprocess.Popen(
                [self.exiftool_binary, '-s'] + [
                    '-' + tag
                    for stags in self.classification_tags
                    for tag in stags
                ] + [img_path],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE).communicate()

            tags = {}

            if stderr:
                logging.warning("exiftool: %s", stderr)

            for line in stdout.splitlines():
                line = line.strip()
                if not line or ':' not in line:
                    continue

                tag, value = line.split(':', 1)
                tag = tag.strip()
                tags[tag] = value.strip()

        if tags is None:
            # Tough luck
            tags = {}
        elif img_path and classification_cache is not None:
            classification_cache.setdefault(img_path, tags)

        return tuple([
            ','.join(map(self.escape_tag, map(tags.get, stags)))
            for stags in self.classification_tags
        ])

    def escape_tag(self, tag):
        if tag is None:
            return 'NA'

        return self.unsafe_re.sub('_', tag)
