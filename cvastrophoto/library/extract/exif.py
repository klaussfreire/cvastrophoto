# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import basestring
import os.path
import subprocess
import logging
import math
import functools


def round_sigdigits(d, x):
    if not x:
        return x

    try:
        x = float(x)
    except (TypeError, ValueError):
        return x

    if not x:
        return x

    scale = 10.0 ** int(math.log10(x))
    return str(round(x / scale, d) * scale)


class ExifTagExtractor(object):

    exiftool_binary = 'exiftool'

    tag_cache = None

    extraction_tags = [
        'Make',
        'Model',
        'InternalSerialNumber', 'SerialNumber',
        'ImageSize', 'ExifImageWidth', 'ExifImageHeight',
        'SensorWidth', 'SensorHeight',
        'SensorLeftBorder', 'SensorTopBorder',
        'SensorRightBorder', 'SensorBottomBorder',
        'PhotometricInterpretation',
        'ISO',
        'ExposureTime', 'BulbDuration',
        'CameraTemperature',
    ]

    tag_roundfunc = {
        'ExposureTime': functools.partial(round_sigdigits, 2),
        'BulbDuration': functools.partial(round_sigdigits, 2),
    }


    def get_tags(self, img_path):
        tags = getattr(img_path, 'exif_tags', None)
        tag_cache = None

        if tags is not None:
            return tags

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
                tag_cache = self.tag_cache
                if tag_cache is None:
                    self.tag_cache = tag_cache = {}
                tags = tag_cache.get(img_path)

        if tags is None and isinstance(img_path, basestring) and os.path.exists(img_path):
            stdout, stderr = subprocess.Popen(
                [self.exiftool_binary, '-s'] + [
                    '-' + tag
                    for tag in self.extraction_tags
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
                value = value.strip()
                if tag in self.tag_roundfunc:
                    value = self.tag_roundfunc[tag](value)
                tags[tag] = value


        if tags is None:
            # Tough luck
            tags = {}
        elif img_path and tag_cache is not None:
            tag_cache.setdefault(img_path, tags)

        return tags
