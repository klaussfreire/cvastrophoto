# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import argparse
import logging
import numpy
import sys
import multiprocessing.pool
import os.path

from .process import (
    add_tracking_opts, create_wiz_kwargs, TRACKING_METHODS, add_method_hook, invoke_method_hooks,
    build_rop as _build_rop, parse_params, make_track_cachedir
)
from .combine import align_inputs
from cvastrophoto.util import srgb

logger = logging.getLogger(__name__)


def add_opts(subp):
    ap = subp.add_parser('register', help="Perform image registration (alignment). Input images are modified in-place.")

    ap.add_argument('--margin', type=int, help='Crop N pixels from the input image edges', metavar='N')
    ap.add_argument('--linear', action='store_true', help='Assume input image is linear')
    ap.add_argument('--nonlinear', action='store_true', help='Assume input image is gamma-encoded')
    ap.add_argument('--no-autoscale', action='store_true', help='Don\'t auto-scale input data')

    ap.add_argument('--cache', help='Cache dir to store precomputed assets to speed up reprocessing')

    ap.add_argument('--reference', help=(
        'The image used as reference frame. '
        'By default, the first image is used as reference'))

    add_tracking_opts(subp, ap)

    ap.add_argument('inputs', nargs='+', help='Input/output images. Will be overwritten.')


def main(opts, pool):
    from cvastrophoto.image import Image, rgb

    open_kw = {'mode': 'update'}
    if opts.margin:
        open_kw['margins'] = (opts.margin,) * 4
    if opts.linear:
        open_kw['linear'] = True
    if opts.nonlinear:
        open_kw['linear'] = False
    if opts.no_autoscale:
        open_kw['autoscale'] = False

    reference = None
    inputs = [Image.open(fpath, default_pool=pool, **open_kw) for fpath in opts.inputs]

    if opts.reference:
        reference = Image.open(opts.reference, default_pool=pool, **open_kw)

    if reference is None:
        reference = inputs[0]

    if opts.cache is None:
        opts.cache = make_track_cachedir(opts, prefix='registration_cache')
    if not os.path.exists(opts.cache):
        os.makedirs(opts.cache)

    for img in align_inputs(opts, pool, reference, inputs, force_align=True, can_skip=True):
        if img is not reference:
            img.save(img.name)
        img.close()
