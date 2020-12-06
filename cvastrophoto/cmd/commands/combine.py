# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging
import numpy
import sys
import multiprocessing.pool

from .process import (
    add_tracking_opts, create_wiz_kwargs, TRACKING_METHODS, add_method_hook, invoke_method_hooks,
    build_rop as _build_rop, parse_params,
)
from cvastrophoto.util import srgb

logger = logging.getLogger(__name__)


def build_rop(ropname, opts, pool, img, **kw):
    class wiz:
        class skyglow:
            raw = img

    return _build_rop(ropname, opts, pool, wiz, **kw)


class AbortError(Exception):
    pass


class AlignmentError(AbortError):
    pass


def add_opts(subp):
    epilogtext = [
        "Supported combination modes:",
    ]
    for mode, combiner in sorted(COMBINERS.items()):
        epilogtext.extend([
            "    " + mode,
        ])
        epilogtext.append(combiner.__doc__)

    ap = subp.add_parser('combine', help="Perform channel combination",
        epilog='\n'.join(epilogtext),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument('--margin', type=int, help='Crop N pixels from the input image edges', metavar='N')
    ap.add_argument('--mode', choices=COMBINERS.keys(), metavar='MODE',
        help='One of the supported channel combination modes')
    ap.add_argument('--no-align', default=False, action='store_true', help='Skip channel alignment and combine as-is')
    ap.add_argument('--linear', action='store_true', help='Assume input image is linear')
    ap.add_argument('--nonlinear', action='store_true', help='Assume input image is gamma-encoded')
    ap.add_argument('--no-autoscale', action='store_true', help='Don\'t auto-scale input data')
    ap.add_argument('--args', help='Parameters for the combination mode')

    ap.add_argument('--color-rops', help='ROPs to be applied to the color data before application of the luminance layer', nargs='+')
    ap.add_argument('--luma-rops', help='ROPs to be applied to the luma data before application of the color layer', nargs='+')

    ap.add_argument('--reference', help=(
        'The image used as reference frame - will not be included in the output. '
        'By default, the first channel is used as reference'))

    add_tracking_opts(subp, ap)

    ap.add_argument('output', help='Output image path')
    ap.add_argument('inputs', nargs='+', help='Input channels, in order for the channel combination mode')


def align_inputs(opts, pool, reference, inputs):
    if opts.no_align:
        for img in inputs:
            yield img
        return

    from cvastrophoto.wizards import whitebalance

    # Construct a wizard to get its tracking factory
    method_hooks = []
    add_method_hook(method_hooks, TRACKING_METHODS, opts.tracking_method)

    wiz_kwargs = create_wiz_kwargs(opts)
    invoke_method_hooks(method_hooks, 'kw', opts, pool, wiz_kwargs)
    wiz = whitebalance.WhiteBalanceWizard(**wiz_kwargs)

    tracker = wiz.light_stacker.tracking_class(inputs[0].dup())

    if reference is not None:
        logger.info("Analyzing reference frame %s", reference.name)
        tracker.correct([reference.rimg.raw_image], img=reference, save_tracks=False)
        reference.close()

    for img in inputs:
        logger.info("Registering %s", img.name)

        corrected = tracker.correct([img.rimg.raw_image], img=img, save_tracks=False)
        if corrected is None:
            logger.error("Alignment of %s failed", img.name)
            raise AlignmentError
        else:
            corrected, = corrected

        logger.info("Registered %s", img.name)

        img.set_raw_image(corrected, add_bias=True)
        yield img


def apply_rops(opts, pool, img, data, ropnames, **kw):
    from cvastrophoto.rops import compound
    rops = []
    for ropname in ropnames:
        rops.append(build_rop(ropname, opts, pool, img, **kw))
    crops = compound.CompoundRop(img, *rops)
    return crops.correct(data)


def apply_color_rops(opts, pool, img, data):
    return apply_rops(opts, pool, img, data, opts.color_rops, copy=False)


def apply_luma_rops(opts, pool, img, data):
    return apply_rops(opts, pool, img, data, opts.luma_rops, copy=False)


def rgb_combination(opts, pool, output_img, reference, inputs):
    """
        Combine RGB input channels into a color image
    """
    from cvastrophoto.util import demosaic
    from cvastrophoto.image import rgb

    image = output_img.postprocessed
    for ch, img in enumerate(align_inputs(opts, pool, reference, inputs[:3])):
        pp_data = img.postprocessed
        if len(pp_data.shape) > 2:
            pp_data = pp_data[:,:,0]
        image[:,:,ch] = pp_data

        del pp_data
        img.close()

    if opts.color_rops:
        image = apply_color_rops(opts, pool, rgb.Templates.RGB, image)

    output_img.set_raw_image(demosaic.remosaic(image, output_img.rimg.raw_pattern), add_bias=True)


def lrgb_combination_base(opts, pool, output_img, reference, inputs, keep_linear=False):
    from cvastrophoto.image import rgb

    lum_data = None
    image = output_img.postprocessed
    for ch, img in enumerate(align_inputs(opts, pool, reference, inputs[:4])):
        pp_data = img.postprocessed
        if len(pp_data.shape) > 2:
            ch_data = pp_data[:,:,0]
        else:
            ch_data = pp_data

        if ch == 0:
            lum_data = ch_data
        elif ch == 1 and len(inputs) == 2 and len(pp_data.shape) == 3:
            image[:] = pp_data
        else:
            image[:,:,ch-1] = ch_data

        del pp_data, ch_data
        img.close()

    image = image.astype(numpy.float32, copy=False)
    lum_image = lum_data.astype(numpy.float32, copy=False)

    scale = max(image.max(), lum_data.max())

    if opts.color_rops:
        image = apply_color_rops(opts, pool, rgb.Templates.RGB, image)
    if opts.luma_rops:
        lum_image = apply_luma_rops(opts, pool, rgb.Templates.LUMINANCE, lum_image)

    if not keep_linear:
        if scale > 0:
            image *= (1.0 / scale)
            lum_image *= (1.0 / scale)

        lum_image = srgb.encode_srgb(lum_image)
        image = srgb.encode_srgb(image)

    return lum_image, image, scale


def lrgb_finish(output_img, image, scale):
    from cvastrophoto.util import demosaic

    image = srgb.decode_srgb(image)

    if scale > 0:
        image *= scale

    output_img.set_raw_image(demosaic.remosaic(image, output_img.rimg.raw_pattern), add_bias=True)


def lrgb_combination(opts, pool, output_img, reference, inputs):
    """
        Combine LRGB input channels into a color (RGB) image by taking
        the color from the RGB channels and the luminance from the L
        channel (in CIE HCL space).
    """
    from skimage import color

    lum_image, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    image = color.lab2lch(color.rgb2lab(image))
    lum = color.lab2lch(color.rgb2lab(color.gray2rgb(lum_image)))[:,:,0]
    image[:,:,0] = lum
    del lum

    image = color.lab2rgb(color.lch2lab(image))

    lrgb_finish(output_img, image, scale)


def lbrgb_combination(opts, pool, output_img, reference, inputs):
    """
        Combine LRGB input channels into a color (RGB) image by taking
        the color from the RGB channels and the luminance from the L and B
        channels combined (in CIE HCL space). Improves spatial resolution when
        working near the diffraction limit by feeding from blue luminance,
        which tends to have higher spatial resolution than L alone.
    """
    from skimage import color

    lum_image, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    lum = color.lab2lch(color.rgb2lab(color.gray2rgb(lum_image)))[:,:,0]
    blum = color.lab2lch(color.rgb2lab(color.gray2rgb(image[:,:,2])))[:,:,0]
    image = color.lab2lch(color.rgb2lab(image))
    image[:,:,0] = numpy.sqrt(lum * blum)
    del lum

    image = color.lab2rgb(color.lch2lab(image))

    lrgb_finish(output_img, image, scale)


def llrgb_combination(opts, pool, output_img, reference, inputs):
    """
        Combine LRGB input channels into a color (RGB) image by taking
        the color from the RGB channels and the luminance from the L and RGB combined
        channels combined (in CIE HCL space).
    """
    from skimage import color

    lum_image, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    lum = color.lab2lch(color.rgb2lab(color.gray2rgb(lum_image)))[:,:,0]
    image = color.lab2lch(color.rgb2lab(image))
    slum = image[:,:,0]
    image[:,:,0] = (lum + slum) * 0.5
    del lum

    image = color.lab2rgb(color.lch2lab(image))

    lrgb_finish(output_img, image, scale)


def hargb_combination(opts, pool, output_img, reference, inputs,
        ha_w=1.0, ha_s=1.0, r_w=1.0, l_w=0.0, ha_fit=True, l_fit=False):
    """
        Combine HaRGB input channels into a color (RGB) image by taking
        the color from the RGB channels, adding Ha in R, and the luminance from the Ha
        channel (in CIE HCL space).

        Parameters:
         - ha_w: Ha weight
         - ha_s: Ha scale, when Ha fitting, Ha will be brightened over R by this much
         - r_w: R weight
         - l_w: L weight (default 0), to mix Ha with L into a superlum
         - l_fit: If 1 (default), ha is autoscaled to fit L when combined with L.
         - ha_fit: If 1 (default), ha is autoscaled to fit r. If 0, it's used as-is.
    """
    from skimage import color

    ha_w = hal_w = float(ha_w)
    ha_s = float(ha_s)
    r_w = float(r_w)
    l_w = float(l_w)
    ha_fit = bool(int(ha_fit))
    l_fit = bool(int(l_fit))

    lum_image, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    ha = lum_image
    if len(ha.shape) > 2:
        ha = ha[:,:,0]

    ha = ha.astype(numpy.float32, copy=False)
    image = image.astype(numpy.float32, copy=False)

    if ha_fit:
        ha_avg = numpy.average(ha)
        r_avg = numpy.average(image[:,:,0])
        r_max = image.max()
        fit_factor = (ha_s * r_avg / ha_avg)
    else:
        fit_factor = 1

    w = 1.0 / (ha_w + r_w)
    ha_w *= w
    r_w *= w

    halum = color.lab2lch(color.rgb2lab(color.gray2rgb(lum_image)))[:,:,0]
    image[:,:,0] = numpy.clip(image[:,:,0] * r_w + ha * (ha_w * fit_factor), None, r_max)
    image = color.lab2lch(color.rgb2lab(image))

    l_max = image[:,:,0].max()
    if l_fit:
        hal_avg = numpy.average(halum)
        l_avg = numpy.average(image[:,:,0])
        fit_factor = (l_avg / hal_avg)
    else:
        fit_factor = 1

    w = 1.0 / (hal_w + l_w)
    hal_w *= w
    l_w *= w

    image[:,:,0] = numpy.clip(image[:,:,2] * l_w + halum * (hal_w * fit_factor), None, l_max)

    image[:,:,0] = halum
    del ha, halum

    image = color.lab2rgb(color.lch2lab(image))

    lrgb_finish(output_img, image, scale)


def slum_combination(opts, pool, output_img, reference, inputs, weight=1):
    """
        Combine LRGB input channels into a grayscale superluminance image by taking
        the luminance from the RGB channels and the luminance from the L and combining them.

        weight: if given, the weight of the RGB component relative to L
    """
    from skimage import color
    from cvastrophoto.image import rgb

    weight = float(weight)

    if reference is None:
        eff_reference = inputs[0]
    else:
        eff_reference = reference
    ref_pp = eff_reference.postprocessed

    rgb_shape = SHAPE_COMBINERS['lrgb'](ref_pp.shape)
    rgb_img = numpy.zeros(rgb_shape, ref_pp.dtype)
    rgb_image = rgb.RGB(None, img=rgb_img, linear=True, autoscale=False)

    lum_image, image, scale = lrgb_combination_base(opts, pool, rgb_image, reference, inputs, keep_linear=True)

    lum = lum_image
    slum = color.rgb2gray(image)

    image = (lum + weight * slum) * (1.0 / (1 + weight))
    del lum, slum

    output_img.set_raw_image(image.reshape(output_img.rimg.raw_image.shape), add_bias=True)


def vbrgb_combination(opts, pool, output_img, reference, inputs):
    """
        Combine LRGB input channels into a color (RGB) image by taking
        the color from the RGB channels and the luminance from the L and B
        channels combined (in HSV space). Improves spatial resolution when
        working near the diffraction limit by feeding from blue luminance,
        which tends to have higher spatial resolution than L alone.
    """
    from skimage import color

    lum_image, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    lum = color.rgb2hsv(color.gray2rgb(lum_image))[:,:,2]
    blum = color.rgb2hsv(color.gray2rgb(image[:,:,2]))[:,:,2]
    image = color.rgb2hsv(image)
    image[:,:,2] = numpy.sqrt(lum * blum)
    del lum

    image = color.hsv2rgb(image)

    lrgb_finish(output_img, image, scale)


def vrgb_combination(opts, pool, output_img, reference, inputs):
    """
        Combine LRGB input channels into a color (RGB) image by taking
        the color from the RGB channels and the luminance from the L
        channel (in HSV space).
    """
    from skimage import color

    lum_image, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    image = color.rgb2hsv(image)
    lum = color.rgb2hsv(color.gray2rgb(lum_image))[:,:,2]
    image[:,:,2] = lum
    del lum

    image = color.hsv2rgb(image)

    lrgb_finish(output_img, image, scale)


def havrgb_combination(opts, pool, output_img, reference, inputs,
        ha_w=1.0, ha_s=1.0, r_w=1.0, l_w=0.0, ha_fit=True, l_fit=False):
    """
        Combine HaRGB input channels into a color (RGB) image by taking
        the color from the RGB channels, adding Ha in red, and the luminance from the Ha
        channel (in HSV space).

        Parameters:
         - ha_w: Ha weight
         - ha_s: Ha scale, when Ha fitting, Ha will be brightened over R by this much
         - r_w: R weight
         - l_w: L weight (default 0), to mix Ha with L into a superlum
         - l_fit: If 1 (default), ha is autoscaled to fit L when combined with L.
         - ha_fit: If 1 (default), ha is autoscaled to fit R. If 0, it's used as-is.
    """
    from skimage import color

    ha_w = hal_w = float(ha_w)
    ha_s = float(ha_s)
    r_w = float(r_w)
    l_w = float(l_w)
    ha_fit = bool(int(ha_fit))
    l_fit = bool(int(l_fit))

    lum_image, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    ha = lum_image
    if len(ha.shape) > 2:
        ha = ha[:,:,0]

    ha = ha.astype(numpy.float32, copy=False)
    image = image.astype(numpy.float32, copy=False)

    r_max = image.max()
    if ha_fit:
        ha_avg = numpy.average(ha)
        r_avg = numpy.average(image[:,:,0])
        fit_factor = (ha_s * r_avg / ha_avg)
    else:
        fit_factor = 1

    w = 1.0 / (ha_w + r_w)
    ha_w *= w
    r_w *= w

    image[:,:,0] = numpy.clip(image[:,:,0] * r_w + ha * (ha_w * fit_factor), None, r_max)
    image = color.rgb2hsv(image)
    halum = color.rgb2hsv(color.gray2rgb(lum_image))[:,:,2]

    l_max = image[:,:,2].max()
    if l_fit:
        hal_avg = numpy.average(halum)
        l_avg = numpy.average(image[:,:,2])
        fit_factor = (l_avg / hal_avg)
    else:
        fit_factor = 1

    w = 1.0 / (hal_w + l_w)
    hal_w *= w
    l_w *= w

    image[:,:,2] = numpy.clip(image[:,:,2] * l_w + halum * (hal_w * fit_factor), None, l_max)
    del ha, halum

    image = color.hsv2rgb(image)

    lrgb_finish(output_img, image, scale)


def star_transplant_combination(opts, pool, output_img, reference, inputs, **args):
    """
        Take the stars from the first image and transplant it into the background
        of the second image.
    """
    from cvastrophoto.rops.tracking.extraction import ExtractPureStarsRop, RemoveStarsRop

    if opts.parallel is None:
        pool2 = multiprocessing.pool.ThreadPool(multiprocessing.cpu_count())
    elif opts.parallel:
        pool2 = multiprocessing.pool.ThreadPool(opts.parallel)
    else:
        pool2 = multiprocessing.pool.ThreadPool(1)

    stars, bg = list(align_inputs(opts, pool, reference, inputs[:2]))

    extract_stars_rop = ExtractPureStarsRop(stars, default_pool=pool, **args)
    extract_bg_rop = RemoveStarsRop(bg, default_pool=pool, **args)

    mxdata = max(stars.rimg.raw_image.max(), bg.rimg.raw_image.max())

    star_data = pool2.apply_async(extract_stars_rop.correct, (stars.rimg.raw_image,), {'img': stars})
    bg_data = pool2.apply_async(extract_bg_rop.correct, (bg.rimg.raw_image,), {'img': bg})

    star_data = star_data.get()
    bg_data = bg_data.get()

    bg_data = bg_data.astype(numpy.float32)
    bg_data += star_data
    bg_data = numpy.clip(bg_data, 0, mxdata, out=bg_data)
    output_img.set_raw_image(bg_data, add_bias=True)


def rgb_shape(ref_shape):
    return ref_shape[:2] + (3,)


def lum_shape(ref_shape):
    return ref_shape[:2] + (1,)


def same_shape(ref_shape):
    return ref_shape


COMBINERS = {
    'rgb': rgb_combination,
    'lrgb': lrgb_combination,
    'llrgb': llrgb_combination,
    'lbrgb': lbrgb_combination,
    'hargb': hargb_combination,
    'slum': slum_combination,
    'vrgb': vrgb_combination,
    'havrgb': havrgb_combination,
    'vbrgb': vbrgb_combination,
    'star_transplant': star_transplant_combination,
}

SHAPE_COMBINERS = {
    'rgb': rgb_shape,
    'lrgb': rgb_shape,
    'llrgb': rgb_shape,
    'lbrgb': rgb_shape,
    'hargb': rgb_shape,
    'slum': lum_shape,
    'vrgb': rgb_shape,
    'havrgb': rgb_shape,
    'vbrgb': rgb_shape,
    'star_transplant': same_shape,
}


def main(opts, pool):
    from cvastrophoto.image import Image, rgb

    open_kw = {}
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

    if reference is not None:
        ref = reference.postprocessed
    else:
        ref = inputs[0].postprocessed
    out_shape = SHAPE_COMBINERS[opts.mode](ref.shape)
    output_img = numpy.zeros(out_shape, ref.dtype)
    output_img = rgb.RGB(opts.output, img=output_img, linear=True, autoscale=False)
    del ref

    if opts.args:
        args = parse_params(opts.args)
    else:
        args = {}

    try:
        COMBINERS[opts.mode](opts, pool, output_img, reference, inputs, **args)

        output_img.save(opts.output)
    except AbortError:
        sys.exit(1)
