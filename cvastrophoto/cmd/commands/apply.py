# -*- coding: utf-8 -*-
from __future__ import print_function

import os.path
import logging
from functools import partial

from .process import build_rop as _build_rop

logger = logging.getLogger(__name__)


def build_rop(ropname, opts, pool, img):
    class wiz:
        class skyglow:
            raw = img

    return _build_rop(ropname, opts, pool, wiz)


def add_opts(subp):
    ap = subp.add_parser('apply', help="Apply ROPs to an image")

    ap.add_argument('--margin', type=int, help='Crop N pixels from the input image edges', metavar='N')

    ap.add_argument('input', help='Input image path')
    ap.add_argument('output', help='Output image path')
    ap.add_argument('rops', nargs='+', help='Raster operations to be applied')


def main(opts, pool):
    from cvastrophoto.rops import compound
    from cvastrophoto.image import Image, rgb

    rops = []

    open_kw = {}
    if opts.margin:
        open_kw['margins'] = (opts.margin,) * 4
    input_img = Image.open(opts.input, default_pool=pool, **open_kw)

    for ropname in opts.rops:
        rops.append(build_rop(ropname, opts, pool, input_img))

    if not rops:
        logger.error("Must specify at least one ROP")
        return 1

    rop_pipe = compound.CompoundRop(input_img, *rops)

    corrected = rop_pipe.correct(input_img.rimg.raw_image)
    input_img.set_raw_image(corrected)

    output_img = rgb.RGB(opts.output, img=input_img.postprocessed, linear=True, autoscale=False)
    output_img.save(opts.output)
