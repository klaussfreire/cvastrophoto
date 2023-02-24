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
    ap.add_argument('--linear', action='store_true', help='Assume input image is linear')
    ap.add_argument('--nonlinear', action='store_true', help='Assume input image is gamma-encoded')
    ap.add_argument('--autoscale', action='store_true', help="Force normalize input images")
    ap.add_argument('--noautoscale', action='store_true', help="Don't normalize input images")
    ap.add_argument('--gamma', type=float, help='Gamma adjustment for output image (default 2.4)')
    ap.add_argument('--use-camera-wb', action='store_true', help="When reading RAW, apply camera WB during postprocessing")
    ap.add_argument('--apply-darklib', action='store_true', help="Calibrate with dark from library")

    ap.add_argument('input', help='Input image path')
    ap.add_argument('output', help='Output image path')
    ap.add_argument('rops', nargs='+', help='Raster operations to be applied')


def main(opts, pool):
    from cvastrophoto.rops import compound
    from cvastrophoto.image import Image, rgb

    rops = []

    open_kw = {}
    pp_kw = {}
    save_kw = {}
    if opts.margin:
        open_kw['margins'] = (opts.margin,) * 4
    if opts.linear:
        open_kw['linear'] = True
    elif opts.nonlinear:
        open_kw['linear'] = False
    if opts.autoscale:
        open_kw['autoscale'] = True
    elif opts.noautoscale:
        open_kw['autoscale'] = False
    if opts.use_camera_wb:
        pp_kw['use_camera_wb'] = True
    if pp_kw:
        open_kw['pp_kw'] = pp_kw
    if opts.gamma:
        save_kw['gamma'] = opts.gamma
    input_img = Image.open(opts.input, default_pool=pool, **open_kw)

    for ropname in opts.rops:
        rops.append(build_rop(ropname, opts, pool, input_img))

    if not rops:
        logger.error("Must specify at least one ROP")
        return 1

    rop_pipe = compound.CompoundRop(input_img, *rops)

    if opts.apply_darklib:
        from cvastrophoto.library.darks import DarkLibrary
        from cvastrophoto.library.bias import BiasLibrary
        input_img.close()
        dark_library = DarkLibrary()
        dark = dark_library.get_master(dark_library.classify_frame(input_img))
        if dark is None:
            dark_library = BiasLibrary()
            dark = dark_library.get_master(dark_library.classify_frame(input_img))
        if dark is not None:
            input_img.denoise([dark])
        else:
            input_img.remove_bias()

    corrected = rop_pipe.correct(input_img.rimg.raw_image, img=input_img)
    input_img.set_raw_image(corrected, add_bias=True)

    output_img = rgb.RGB(opts.output, img=input_img.postprocessed, linear=True, autoscale=False)
    output_img.save(opts.output, **save_kw)
