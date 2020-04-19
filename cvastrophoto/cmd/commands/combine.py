# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
import numpy
import sys

from .process import add_tracking_opts, create_wiz_kwargs, TRACKING_METHODS, add_method_hook, invoke_method_hooks

logger = logging.getLogger(__name__)


class AbortError(Exception):
    pass


class AlignmentError(AbortError):
    pass


def add_opts(subp):
    ap = subp.add_parser('combine', help="Perform channel combination")

    ap.add_argument('--margin', type=int, help='Crop N pixels from the input image edges', metavar='N')
    ap.add_argument('--mode', choices=COMBINERS.keys(), metavar='MODE',
        help='One of the supported channel combination modes')

    ap.add_argument('--reference', help=(
        'The image used as reference frame - will not be included in the output. '
        'By default, the first channel is used as reference'))

    add_tracking_opts(subp, ap)

    ap.add_argument('output', help='Output image path')
    ap.add_argument('inputs', nargs='+', help='Input channels, in order for the channel combination mode')


def align_inputs(opts, pool, reference, inputs):
    from cvastrophoto.wizards import whitebalance

    # Construct a wizard to get its tracking factory
    method_hooks = []
    add_method_hook(method_hooks, TRACKING_METHODS, opts.tracking_method)

    wiz_kwargs = create_wiz_kwargs(opts)
    invoke_method_hooks(method_hooks, 'kw', opts, pool, wiz_kwargs)
    wiz = whitebalance.WhiteBalanceWizard(**wiz_kwargs)

    tracker = wiz.light_stacker.tracking_class(inputs[0])

    if reference is not None:
        tracker.correct(reference.rimg.raw_image, img=reference, save_tracks=False)
        reference.close()

    for img in inputs:
        corrected = tracker.correct([img.rimg.raw_image], img=img, save_tracks=False)
        if corrected is None:
            logger.error("Alignment of %s failed", img.name)
            raise AlignmentError
        else:
            corrected, = corrected

        img.set_raw_image(corrected, add_bias=True)
        yield img


def rgb_combination(opts, pool, output_img, reference, inputs):
    from cvastrophoto.util import demosaic

    image = output_img.postprocessed
    for ch, img in enumerate(align_inputs(opts, pool, reference, inputs[:3])):
        pp_data = img.postprocessed
        if len(pp_data.shape) > 2:
            pp_data = pp_data[:,:,0]
        image[:,:,ch] = pp_data

        del pp_data
        img.close()

    output_img.set_raw_image(demosaic.remosaic(image, output_img.rimg.raw_pattern), add_bias=True)


def lrgb_combination(opts, pool, output_img, reference, inputs):
    from skimage import color
    from cvastrophoto.util import demosaic

    lum_data = None
    image = output_img.postprocessed
    for ch, img in enumerate(align_inputs(opts, pool, reference, inputs[:4])):
        pp_data = img.postprocessed
        if len(pp_data.shape) > 2:
            pp_data = pp_data[:,:,0]

        if ch == 0:
            lum_data = pp_data
        else:
            image[:,:,ch-1] = pp_data

        del pp_data
        img.close()

    image = image.astype(numpy.float32, copy=False)
    lum_image = lum_data.astype(numpy.float32, copy=False)
    scale = max(image.max(), lum_data.max())
    if scale > 0:
        image *= (1.0 / scale)
        lum_image *= (1.0 / scale)

    image = color.rgb2hsv(image)
    lum = color.rgb2hsv(color.gray2rgb(lum_image))[:,:,2]
    image[:,:,2] = lum
    del lum

    image = color.hsv2rgb(image)

    if scale > 0:
        image *= scale

    output_img.set_raw_image(demosaic.remosaic(image, output_img.rimg.raw_pattern), add_bias=True)


def lbrgb_combination(opts, pool, output_img, reference, inputs):
    from skimage import color
    from cvastrophoto.util import demosaic

    lum_data = None
    image = output_img.postprocessed
    for ch, img in enumerate(align_inputs(opts, pool, reference, inputs[:4])):
        pp_data = img.postprocessed
        if len(pp_data.shape) > 2:
            pp_data = pp_data[:,:,0]

        if ch == 0:
            lum_data = pp_data
        else:
            image[:,:,ch-1] = pp_data

        del pp_data
        img.close()

    image = image.astype(numpy.float32, copy=False)
    lum_image = lum_data.astype(numpy.float32, copy=False)
    scale = max(image.max(), lum_data.max())
    if scale > 0:
        image *= (1.0 / scale)
        lum_image *= (1.0 / scale)

    lum = color.rgb2hsv(color.gray2rgb(lum_image))[:,:,2]
    blum = color.rgb2hsv(color.gray2rgb(image[:,:,2]))[:,:,2]
    image = color.rgb2hsv(image)
    image[:,:,2] = numpy.sqrt(lum * blum)
    del lum

    image = color.hsv2rgb(image)

    if scale > 0:
        image *= scale

    output_img.set_raw_image(demosaic.remosaic(image, output_img.rimg.raw_pattern), add_bias=True)


COMBINERS = {
    'rgb': rgb_combination,
    'lrgb': lrgb_combination,
    'lbrgb': lbrgb_combination,
}


def main(opts, pool):
    from cvastrophoto.image import Image, rgb

    open_kw = {}
    if opts.margin:
        open_kw['margins'] = (opts.margin,) * 4

    inputs = []
    reference = None

    inputs = [Image.open(fpath, default_pool=pool, **open_kw) for fpath in opts.inputs]

    if opts.reference:
        reference = Image.open(opts.reference, default_pool=pool, **open_kw)

    if reference is not None:
        output_img = reference.postprocessed
    else:
        output_img = inputs[0].postprocessed
    out_shape = output_img.shape[:2] + (3,)
    output_img = numpy.zeros(out_shape, output_img.dtype)
    output_img = rgb.RGB(opts.output, img=output_img, linear=True, autoscale=False)

    try:
        COMBINERS[opts.mode](opts, pool, output_img, reference, inputs)

        output_img.save(opts.output)
    except AbortError:
        sys.exit(1)
