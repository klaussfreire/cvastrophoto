# -*- coding: utf-8 -*-
from __future__ import absolute_import

from past.builtins import basestring
import os.path
import logging

import astropy.io.fits


class FitsTagExtractor(object):

    exiftool_binary = 'exiftool'

    tag_cache = None

    def get_tags(self, img_path):
        tags = getattr(img_path, 'fits_header', None)
        tag_cache = None

        if tags is not None:
            if isinstance(tags, astropy.io.fits.header.Header):
                tags = {k.upper(): str(tags[k]).strip() for k in tags.keys()}
            return tags

        if not isinstance(img_path, basestring) and hasattr(img_path, 'name'):
            img_path = getattr(img_path, 'name', None)

        frameno = 0
        if isinstance(img_path, basestring):
            if not (img_path.lower().endswith('.fit') or img_path.lower().endswith('.fits')):
                return None

            if not os.path.exists(img_path) and '#' in img_path:
                # Subframes are suffixed with #N, remove
                img_path, frameno = img_path.rsplit('#', 1)
                frameno = int(frameno)

        if tags is None:
            if img_path is None:
                tags = None
            else:
                tag_cache = self.tag_cache
                if tag_cache is None:
                    self.tag_cache = tag_cache = {}
                tags = tag_cache.get(img_path)

        if tags is None and isinstance(img_path, basestring) and os.path.exists(img_path):
            try:
                with astropy.io.fits.open(img_path) as hdul:
                    header = hdul[frameno].header
                    tags = {k.upper(): str(header[k]).strip() for k in header.keys()}
            except Exception as e:
                logging.warning("fits: %s", e)

        if tags is None:
            # Tough luck
            tags = {}
        elif img_path and tag_cache is not None:
            tag_cache.setdefault(img_path, tags)

        return tags
