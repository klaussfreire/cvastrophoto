# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging
import numpy
import sys
import multiprocessing.pool
import os.path
import gzip

try:
    import cPickle
except ImportError:
    import pickle as cPickle

from .process import (
    add_tracking_opts, create_wiz_kwargs, TRACKING_METHODS, add_method_hook, invoke_method_hooks,
    build_rop as _build_rop, parse_params, make_track_cachedir
)
from cvastrophoto.util import srgb

logger = logging.getLogger(__name__)


def build_rop(ropname, opts, pool, img, **kw):
    if pool is not img.default_pool:
        img = img.dup()
        img.default_pool = pool

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

    ap.add_argument('--cache', help='Cache dir to store precomputed assets to speed up reprocessing')

    ap.add_argument('--reference', help=(
        'The image used as reference frame - will not be included in the output. '
        'By default, the first channel is used as reference'))

    add_tracking_opts(subp, ap)

    ap.add_argument('output', help='Output image path')
    ap.add_argument('inputs', nargs='+', help='Input channels, in order for the channel combination mode')


def align_inputs(opts, pool, reference, inputs, force_align=False, can_skip=False):
    if not force_align and opts.no_align:
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

    if opts.cache:
        track_state_path = os.path.join(opts.cache, 'track_state.gz')
        if os.path.exists(track_state_path):
            with open(track_state_path, 'rb') as fileobj:
                fileobj = gzip.GzipFile(mode='rb', fileobj=fileobj)
                tracker.load_state(cPickle.load(fileobj))

    if reference is not None:
        logger.info("Analyzing reference frame %s", reference.name)
        tracker.correct([reference.rimg.raw_image], img=reference, save_tracks=False)
        reference.close()

    errors = []

    for img in inputs:
        logger.info("Registering %s", img.name)

        corrected = tracker.correct([img.rimg.raw_image], img=img, save_tracks=False)
        if corrected is None:
            logger.error("Alignment of %s failed", img.name)
            if can_skip and opts.continue_on_error:
                errors.append(img.name)
                img.close()
                continue
            raise AlignmentError
        else:
            corrected, = corrected

        logger.info("Registered %s", img.name)

        img.set_raw_image(corrected, add_bias=True)
        yield img

    if errors:
        logger.error("Failed to register %d images:", len(errors))
        for imname in errors:
            logger.error(" - %r", imname)

    if opts.cache:
        try:
            with open(track_state_path, 'wb') as fileobj:
                fileobj = gzip.GzipFile(mode='wb', fileobj=fileobj)
                try:
                    cPickle.dump(tracker.get_state(), fileobj)
                finally:
                    fileobj.close()
        except Exception:
            logger.exception("Error persisting tracking state")


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
        image = apply_color_rops(opts, pool, output_img, image)

    output_img.set_raw_image(demosaic.remosaic(image, output_img.rimg.raw_pattern), add_bias=True)


def lrgb_combination_base(opts, pool, output_img, reference, inputs,
        keep_linear=False, do_color_rops=True, do_luma_rops=True):
    from cvastrophoto.image import rgb

    lum_data = None
    lum_image_file = None
    image = output_img.postprocessed
    for ch, img in enumerate(align_inputs(opts, pool, reference, inputs[:4])):
        pp_data = img.postprocessed
        if len(pp_data.shape) > 2:
            ch_data = pp_data[:,:,0]
        else:
            ch_data = pp_data

        if ch == 0:
            lum_data = ch_data
            lum_image_file = img
        elif ch == 1 and len(inputs) == 2 and len(pp_data.shape) == 3:
            image[:] = pp_data
        else:
            image[:,:,ch-1] = ch_data

        del pp_data, ch_data
        img.close()

    image = image.astype(numpy.float32, copy=False)
    lum_image = lum_data.astype(numpy.float32, copy=False)

    scale = max(image.max(), lum_data.max())

    if opts.color_rops and do_color_rops:
        image = apply_color_rops(opts, pool, output_img, image)
    if opts.luma_rops and do_luma_rops:
        lum_image = apply_luma_rops(opts, pool, lum_image_file or rgb.Templates.LUMINANCE, lum_image)

    if not keep_linear:
        if scale > 0:
            image *= (1.0 / scale)
            lum_image *= (1.0 / scale)

        lum_image = srgb.encode_srgb(lum_image)
        image = srgb.encode_srgb(image)

    return lum_image, lum_image_file, image, scale


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

    lum_image, _, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

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

    lum_image, _, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

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

    lum_image, _, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    lum = color.lab2lch(color.rgb2lab(color.gray2rgb(lum_image)))[:,:,0]
    image = color.lab2lch(color.rgb2lab(image))
    slum = image[:,:,0]
    image[:,:,0] = (lum + slum) * 0.5
    del lum

    image = color.lab2rgb(color.lch2lab(image))

    lrgb_finish(output_img, image, scale)


def compute_broadband_scaling(pool, nb, bb, nr_l=1.0):
    from cvastrophoto.rops.denoise import diffusion
    from cvastrophoto.image import rgb
    nr = diffusion.StarlessDiffusionRop(rgb.Templates.LUMINANCE, L=nr_l, despeckle_size=8, pregauss_size=8)
    bbnr, nbnr = pool.map(nr.correct, [bb.copy(), nb.copy()])

    bb_scale = (bbnr + numpy.average(bbnr)) / (nbnr + numpy.average(nbnr))
    bb_scale = nr.correct(bb_scale)
    bb_scale /= numpy.average(bb_scale)

    return bb_scale


def hargb_combination(opts, pool, output_img, reference, inputs,
        ha_w=1.0, ha_s=1.0, r_w=1.0, l_w=0.0, ha_fit=True, l_fit=True, bb_color_fit=False, bb_lum_fit=False, bb_nr=1.0,
        v_fit=True):
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
         - v_fit: If 1 (default), a VRGB combination is applied before the LRGB combination to improve
           output luminance matching. If l_fit=1, each step does an independent ha-to-L fit.
         - ha_fit: If 1 (default), ha is autoscaled to fit r. If 0, it's used as-is.
         - bb_color_fit: If 1, ha and red are compared to map broadband sources
           and ha data will be scaled to fit broadband content in the color data
         - bb_lum_fit: If 1, broadband scaling will be applied to luminance data as well
         - bb_nr: Amount of noise reduction applied to the broadband scale map (default 1, very aggressive)
    """
    from skimage import color
    from cvastrophoto.image import rgb

    ha_w = hal_w = float(ha_w)
    ha_s = float(ha_s)
    r_w = float(r_w)
    l_w = float(l_w)
    ha_fit = bool(int(ha_fit))
    l_fit = bool(int(l_fit))
    v_fit = bool(int(v_fit))
    bb_color_fit = bool(int(bb_color_fit))
    bb_lum_fit = bool(int(bb_lum_fit))
    bb_nr = float(bb_nr)

    lum_image, lum_image_file, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs,
        do_color_rops=False, do_luma_rops=False, keep_linear=True)

    ha = lum_image
    if len(ha.shape) > 2:
        ha = ha[:,:,0]

    ha = ha.astype(numpy.float32)
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

    r = image[:,:,0]
    ha *= fit_factor
    if bb_color_fit or bb_lum_fit:
        bb_scale = compute_broadband_scaling(pool, ha, r, bb_nr)
        if bb_color_fit:
            bb_color_scale = bb_scale
        else:
            bb_color_scale = 1
    else:
        bb_scale = bb_color_scale = 1

    image[:,:,0] = numpy.clip(r * r_w + ha * (bb_color_scale * ha_w), None, r_max)

    if bb_lum_fit:
        lum_image = lum_image.astype(numpy.float32, copy=False)
        lum_image *= bb_scale

    if opts.luma_rops:
        lum_image = pool.apply_async(lambda lum_image=lum_image:
            apply_luma_rops(opts, pool, lum_image_file or rgb.Templates.LUMINANCE, lum_image)
        )

    if opts.color_rops:
        image = apply_color_rops(opts, pool, output_img, image)

    if opts.luma_rops:
        lum_image = lum_image.get()

    if scale > 0:
        image *= (1.0 / scale)
        lum_image *= (1.0 / scale)

    if v_fit:
        if l_fit:
            hal_avg = numpy.average(lum_image)
            l_avg = numpy.average(color.rgb2gray(image))
            fit_factor = (l_avg / hal_avg)
        else:
            fit_factor = 1

        vlum_image = srgb.encode_srgb(lum_image * fit_factor)
        vimage = srgb.encode_srgb(image)

        vimage = color.rgb2hsv(vimage)
        havlum = color.rgb2hsv(color.gray2rgb(vlum_image))[:,:,2]

        w = 1.0 / (hal_w + l_w)
        vhal_w = hal_w * w
        vl_w = l_w * w

        l_max = vimage[:,:,2].max()
        vimage[:,:,2] = numpy.clip(vimage[:,:,2] * vl_w + havlum * vhal_w, None, l_max)

        image = srgb.decode_srgb(color.hsv2rgb(vimage))
        del vimage, vlum_image, havlum

    if l_fit:
        hal_avg = numpy.average(lum_image)
        l_avg = numpy.average(color.rgb2gray(image))
        fit_factor = (l_avg / hal_avg)
        lum_image *= fit_factor

    lum_image = srgb.encode_srgb(lum_image)
    image = srgb.encode_srgb(image)

    halum = color.lab2lch(color.rgb2lab(color.gray2rgb(lum_image)))[:,:,0]
    image = color.lab2lch(color.rgb2lab(image))

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

    lum_image, _, image, scale = lrgb_combination_base(opts, pool, rgb_image, reference, inputs, keep_linear=True)

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

    lum_image, _, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

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

    lum_image, _, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs)

    image = color.rgb2hsv(image)
    lum = color.rgb2hsv(color.gray2rgb(lum_image))[:,:,2]
    image[:,:,2] = lum
    del lum

    image = color.hsv2rgb(image)

    lrgb_finish(output_img, image, scale)


def havrgb_combination(opts, pool, output_img, reference, inputs,
        ha_w=1.0, ha_s=1.0, r_w=1.0, l_w=0.0, ha_fit=True, l_fit=False, bb_color_fit=False, bb_lum_fit=False, bb_nr=1.0):
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
    from cvastrophoto.image import rgb

    ha_w = hal_w = float(ha_w)
    ha_s = float(ha_s)
    r_w = float(r_w)
    l_w = float(l_w)
    ha_fit = bool(int(ha_fit))
    l_fit = bool(int(l_fit))
    bb_color_fit = bool(int(bb_color_fit))
    bb_lum_fit = bool(int(bb_lum_fit))
    bb_nr = float(bb_nr)

    lum_image, lum_image_file, image, scale = lrgb_combination_base(opts, pool, output_img, reference, inputs,
        do_color_rops=False, do_luma_rops=False, keep_linear=True)

    ha = lum_image
    if len(ha.shape) > 2:
        ha = ha[:,:,0]

    ha = ha.astype(numpy.float32)
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

    r = image[:,:,0]
    ha *= fit_factor
    if bb_color_fit or bb_lum_fit:
        bb_scale = compute_broadband_scaling(pool, ha, r, bb_nr)
        if bb_color_fit:
            bb_color_scale = bb_scale
        else:
            bb_color_scale = 1
    else:
        bb_scale = bb_color_scale = 1

    image[:,:,0] = numpy.clip(image[:,:,0] * r_w + ha * (bb_color_scale * ha_w), None, r_max)

    if bb_lum_fit:
        lum_image = lum_image.astype(numpy.float32, copy=False)
        lum_image *= bb_scale

    if opts.luma_rops:
        lum_image = pool.apply_async(lambda lum_image=lum_image:
            apply_luma_rops(opts, pool, lum_image_file or rgb.Templates.LUMINANCE, lum_image)
        )

    if opts.color_rops:
        image = apply_color_rops(opts, pool, output_img, image)

    if opts.luma_rops:
        lum_image = lum_image.get()

    if scale > 0:
        image *= (1.0 / scale)
        lum_image *= (1.0 / scale)

    if l_fit:
        hal_avg = numpy.average(lum_image)
        l_avg = numpy.average(color.rgb2gray(image))
        fit_factor = (l_avg / hal_avg)
        lum_image *= fit_factor

    lum_image = srgb.encode_srgb(lum_image)
    image = srgb.encode_srgb(image)

    image = color.rgb2hsv(image)
    halum = color.rgb2hsv(color.gray2rgb(lum_image))[:,:,2]

    w = 1.0 / (hal_w + l_w)
    hal_w *= w
    l_w *= w

    l_max = image[:,:,2].max()
    image[:,:,2] = numpy.clip(image[:,:,2] * l_w + halum * hal_w, None, l_max)
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

    if opts.cache is None:
        opts.cache = make_track_cachedir(opts, prefix='combine_cache')
    if not os.path.exists(opts.cache):
        os.makedirs(opts.cache)

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
