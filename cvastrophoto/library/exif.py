# -*- coding: utf-8 -*-
from __future__ import absolute_import

import subprocess
import re
import logging


class ExifClassificationMixIn(object):

    unsafe_re = re.compile(r'[^a-zA-Z0-9\s.]+')

    classification_tags = None
    exiftool_binary = 'exiftool'

    def classify_frame(self, img_path):

        tags = getattr(img_path, 'exif_tags', None)

        if tags is None:
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

        return tuple([
            ','.join(map(self.escape_tag, map(tags.get, stags)))
            for stags in self.classification_tags
        ])

    def escape_tag(self, tag):
        if tag is None:
            return 'NA'

        return self.unsafe_re.sub('_', tag)
