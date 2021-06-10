# -*- coding: utf-8 -*-
from __future__ import print_function

from future.builtins import map as imap
from functools import partial
import logging

from .process import build_rop as _build_rop, add_output_rop

logger = logging.getLogger(__name__)


MEASURE_ROPS = {
    'focus': partial(add_output_rop, 'measures.focus', 'FocusMeasureRop'),
    'seeing': partial(add_output_rop, 'measures.seeing', 'SeeingMeasureRop'),
    'seeing+focus': partial(add_output_rop, 'measures.seeing', 'SeeingFocusRankingRop'),
    'fwhm': partial(add_output_rop, 'measures.fwhm', 'FWHMMeasureRop'),
    'stars': partial(add_output_rop, 'measures.fwhm', 'StarCountMeasureRop'),
    'snr': partial(add_output_rop, 'measures.stats', 'SNRMeasureRop'),
    'entropy': partial(add_output_rop, 'measures.entropy', 'LocalEntropyMeasureRop'),
    'bgavg': partial(add_output_rop, 'measures.stats', 'BgAvgMeasureRop'),
    'avg': partial(add_output_rop, 'measures.stats', 'AvgMeasureRop'),
    'std': partial(add_output_rop, 'measures.stats', 'StdDevMeasureRop'),
    'max': partial(add_output_rop, 'measures.stats', 'MaxMeasureRop'),
    'min': partial(add_output_rop, 'measures.stats', 'MinMeasureRop'),
    'median': partial(add_output_rop, 'measures.stats', 'MedianMeasureRop'),
}


def build_rop(ropname, opts, pool, img):
    class wiz:
        class skyglow:
            raw = img

    return _build_rop(ropname, opts, pool, wiz, rops_catalog=MEASURE_ROPS, _param_parts=2)


def add_opts(subp):
    ap = subp.add_parser('measure', help="Measure an image")

    ap.add_argument('--margin', type=int, help='Crop N pixels from the input image edges', metavar='N')
    ap.add_argument('--linear', action='store_true', help='Assume input image is linear')
    ap.add_argument('--nonlinear', action='store_true', help='Assume input image is gamma-encoded')
    ap.add_argument('--darklib', help='Location of the main dark library', default=None)
    ap.add_argument('--use-darklib', action='store_true', help='Use the default darklib. Default is not to.')

    ap.add_argument('rop', help='Measure operation to be applied')
    ap.add_argument('inputs', help='Input image paths', nargs='+')


def main(opts, pool):
    from cvastrophoto.image import Image
    from cvastrophoto.library import darks

    open_kw = {}
    if opts.margin is not None:
        open_kw['margins'] = (opts.margin,) * 4
    if opts.linear:
        open_kw['linear'] = True
    elif opts.nonlinear:
        open_kw['linear'] = False

    if len(opts.inputs) > 1:
        map_ = pool.imap_unordered
        rop_pool = None
    else:
        map_ = imap
        rop_pool = pool

    if opts.darklib:
        dark_library = darks.DarkLibrary(opts.darklib, pool)
    elif opts.use_darklib:
        dark_library = darks.DarkLibrary()
    else:
        dark_library = None

    def do_measure(path):
        input_img = Image.open(path, default_pool=rop_pool, **open_kw)

        if dark_library is not None:
            master_dark = dark_library.get_master(dark_library.classify_frame(input_img), raw=input_img)
            if master_dark is not None:
                input_img.denoise([master_dark], entropy_weighted=False)

        rop = build_rop(opts.rop, opts, pool, input_img)
        measure = rop.measure_scalar(input_img.rimg.raw_image)
        input_img.close()

        return path, measure

    for path, measure in map_(do_measure, opts.inputs):
        print("%s: %r" % (path, measure))
