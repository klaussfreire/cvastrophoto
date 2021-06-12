# -*- coding: utf-8 -*-
from __future__ import print_function

import logging

logger = logging.getLogger(__name__)


def get_extractors():
    from cvastrophoto.library.extract import exif, fits
    return [
        fits.FitsTagExtractor(),
        exif.ExifTagExtractor(),
    ]


def get_tags(img):
    for extractor in get_extractors():
        tags = extractor.get_tags(img)
        if tags:
            return tags


def add_opts(subp):
    ap = subp.add_parser('info', help="Show image metadata")
    ap.add_argument('inputs', nargs='+', help='Input image paths')


def main(opts, pool):
    from cvastrophoto.image import Image

    for path in opts.inputs:
        print("=== %s ===\n" % (path,))

        img = Image.open(path)
        if getattr(img, 'fits_header', None) is not None:
            print("FITS:\n")
            print(repr(img.fits_header))
        else:
            tags = get_tags(path)
            if tags:
                print("Standard classifiers:\n")
                for k, v in tags.items():
                    print("%s: \t%s" % (k, v))
