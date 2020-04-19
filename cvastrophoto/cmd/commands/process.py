# -*- coding: utf-8 -*-
from __future__ import print_function

import os.path
import logging
from functools import partial

logger = logging.getLogger(__name__)


def add_tracking_opts(subp, ap):
    ap.add_argument('--trackphases', type=int,
        help='Enable multiphase tracking. Higher numbers create more phases. The default should be fine',
        default=1)
    ap.add_argument('--track-coarse-limit', type=float,
        help=(
            'When multiphase tracking, defines how rough the first tracking solution can be. '
            'The default should be fine, unless heavy misalignment is expected.'
        ))
    ap.add_argument('--track-coarse-downsample', type=float,
        help=(
            'When multiphase tracking, defines a downsample factor for initial tracking phases. '
            'The default should be fine, unless heavy misalignment is expected or memory reductions are desirable.'
        ))
    ap.add_argument('--track-fine-distance', type=int,
        help=(
            'When multiphase tracking, defines the search distance for the final alignment phase. '
            'The default should be fine.'
        ))
    ap.add_argument('--track-distance', type=int,
        help=(
            'Defines the search distance for the final alignment phase. '
            'The default should be fine.'
        ))
    ap.add_argument('--feature-tracking', action='store_true',
        help=(
            "Enables tracking through ORB feature detection and matching. This is able to roughly align "
            "Severely misaligned images, or ones that are mostly empty where image correlation fails, "
            "but is imprecise so it's only useful as initial rough alignment."
        ))
    ap.add_argument('--feature-tracking-params',
        help=(
            "Customize parameters of the feature tracking phase. Feature tracking must be enabled to "
            "make any difference, otherwise it will be ignored."
        ))
    ap.add_argument('--tracking-method', '-mt', help='Set sub alignment method', default='grid')


def add_opts(subp):
    ap = subp.add_parser('process', help="Process a session's data")

    ap.add_argument('--config', help='Load config from a file', default=None)

    ap.add_argument('--darklib', help='Location of the main dark library', default=None)
    ap.add_argument('--biaslib', help='Location of the bias library', default=None)
    ap.add_argument('--noautodarklib', help="Use darks as they are, don't build a local library",
        default=False, action='store_true')

    ap.add_argument('--path', '-p', help='Base path for all data files', default='.')
    ap.add_argument('--lightsdir', '-L', help='Location of light frames', default='Lights')
    ap.add_argument('--darksdir', '-D', help='Location of dark frames', default='Darks')
    ap.add_argument('--flatsdir', '-F', help='Location of light frames', default='Flats')
    ap.add_argument('--darkflatsdir', '-Df', help='Location of light frames', default='Dark Flats')
    ap.add_argument('--darkbadmap', help="Build a bad pixel map with data from lights and darks both",
        default=False, action='store_true')

    add_tracking_opts(subp, ap)

    ap.add_argument('--reference', '-r',
        help='Set reference frame. Must be the name of the reference light frame.')

    ap.add_argument('--light-method', '-m', help='Set light stacking method', default='drizzle',
        choices=LIGHT_METHODS.keys())
    ap.add_argument('--flat-method', '-mf', help='Set flat stacking method', default='median',
        choices=FLAT_METHODS.keys())
    ap.add_argument('--flat-mode', '-mfm', help='Set flat calibration mode', default='color')
    ap.add_argument('--flat-smoothing', type=float, help='Set flat smoothing radius, recommended for high-iso')
    ap.add_argument('--skyglow-method', '-ms', help='Set automatic background extraction method',
        default='localgradient')
    ap.add_argument('--weight-method', '-mw', default=None, help='Weight subs according to this method')
    ap.add_argument('--no-normalize-weights', default=False, action='store_true',
        help=(
            "Don't normalize weights, apply them verbatim as they come out of the selected "
            "weighting method. Weight normalization increases the separation between low-quality "
            "and high-quality data, and is normally a good thing. But in cases when it might "
            "not be needed or desirable, this flag disables it."
        ))
    ap.add_argument('--no-mirror-edges', default=False, action='store_true',
        help=(
            "When stacking, consider non-overlapping regions as zero-weight instead "
            "of mirroring edges. Edge mirroring improves the efficacy of skyglow methods "
            "but it can cause artifacts. If the image has few gradients, using this can "
            "avoid those artifacts at the expense of gradient removal efficacy."
        ))

    ap.add_argument('--cache', '-C', help="Set the cache location. By default, it's auto-generated based on settings")

    ap.add_argument('--preview', '-P', action='store_true', help='Enable preview generation')
    ap.add_argument('--preview-path', '-Pp', help='Specify a custom preview path template')
    ap.add_argument('--preview-brightness', '-Pb', type=float, default=32, help='Set output stretch brightness')
    ap.add_argument('--preview-quick', '-Pq', action='store_true',
        help='Do a rough preview, rather than a fullly accurate post-processing',
        default=False)

    ap.add_argument('--brightness', '-b', type=float, default=4, help='Set output stretch brightness')
    ap.add_argument('--whitebalance', '-w',
        help=(
            'Set extra white balance coefficients. Can be either a 4-tuple of floats, or '
            'a standard name from one of the standard white balance coefficients. '
            'See list-wb for a list'
        ))
    ap.add_argument('--hdr', action='store_true', help='Save output file in HDR')
    ap.add_argument('--hdr-stops', type=int, help='How many stops of HDR exposures to blend')
    ap.add_argument('output', help='Output path')

    ap.add_argument('--input-rops', '-Ri', nargs='+')
    ap.add_argument('--preskyglow-rops', '-Rs', nargs='+')
    ap.add_argument('--output-rops', '-Ro', nargs='+')
    ap.add_argument('--skyglow-preprocessing-rops', '-Rspp', nargs='+')

    ap.add_argument('--list-wb', action='store_true',
        help='Print a list of white balance coefficients and exit')

    ap.add_argument('--limit-first', type=int, metavar='N',
        help='Process only the first N subs, useful for quick previews')

    ap.add_argument('--selection-method', '-S', default=None,
        help='Select subs and keep the NSELECT%% best according to this method')
    ap.add_argument('--select-percent-best', '-Sr', type=float, metavar='NSELECT', default=0.7)


def create_wiz_kwargs(opts):
    wiz_kwargs = dict(
        tracking_2phase=opts.trackphases,
    )
    if opts.track_coarse_limit:
        wiz_kwargs['tracking_coarse_limit'] = opts.track_coarse_limit
    if opts.track_fine_distance:
        wiz_kwargs['tracking_fine_distance'] = opts.track_fine_distance
    if opts.track_distance:
        wiz_kwargs['tracking_coarse_distance'] = opts.track_distance
    if opts.track_coarse_downsample:
        wiz_kwargs['tracking_coarse_downsample'] = opts.track_coarse_downsample
    if opts.feature_tracking:
        from cvastrophoto.rops.tracking import orb

        orb_kw = dict(
            median_shift_limit=opts.track_coarse_limit or 2,
            downsample=opts.track_coarse_downsample or 2,
        )
        if opts.feature_tracking_params:
            orb_kw.update(parse_params(opts.feature_tracking_params))
        wiz_kwargs['feature_tracking_class'] = partial(orb.OrbFeatureTrackingRop, **orb_kw)

    return wiz_kwargs


def noop(*p, **kw):
    pass

def boolean(val):
    return val.lower() in ('1', 'true', 'yes')

CONFIG_TYPES = {
    'trackphases': int,
    'preview': boolean,
    'preview_brightness': float,
    'preview_quick': boolean,
    'brightness': float,
}

PARAM_TYPES = {
    'L': float,
    'T': float,
    'R': int,
    'thr': int,
    'steps': int,
    'track_distance': int,
}

def parse_params(params_str):
    params = dict([kvp.split('=') for kvp in params_str.split(',')])
    for k, v in params.iteritems():
        if k in PARAM_TYPES:
            params[k] = PARAM_TYPES[k](v)
    return params

def build_rop(ropname, opts, pool, wiz, **kw):
    parts = ropname.rsplit(':', 2)
    params = {}
    if len(parts) == 3:
        ropname, params = ropname.rsplit(':', 1)
        params = parse_params(params)
    return ROPS[ropname](opts, pool, wiz, params, **kw)

def add_method_hook(method_hooks, methods, method):
    if not method:
        return

    if ':' in method:
        method, params = method.rsplit(':', 1)
        params = parse_params(params)
    else:
        params = None

    method_info = methods[method].copy()
    if params:
        method_info['params'] = params

    method_hooks.append(method_info)

def invoke_method_hooks(method_hooks, step, opts, pool, wiz):
    for method_info in method_hooks:
        method_info.get(step, noop)(opts, pool, wiz, method_info.get('params', {}))

def main(opts, pool):
    from cvastrophoto.wizards.whitebalance import WhiteBalanceWizard
    from cvastrophoto.image import raw

    if opts.config:
        with open(opts.config, 'r') as config_file:
            for line in config_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                opt, val = line.split('=', 1)
                opt = opt.strip()
                val = val.strip()

                if opt in CONFIG_TYPES:
                    val = CONFIG_TYPES[opt](val)

                setattr(opts, opt, val)

    if opts.list_wb:
        for name, coeffs in WhiteBalanceWizard.WB_SETS.iteritems():
            print(name, ':', coeffs)

    if opts.cache is None:
        opts.cache = '.cvapstatecache/state_cache'
        opts.cache += '_track%dp' % opts.trackphases
        if opts.light_method.startswith('drizzle') or opts.light_method.startswith('interleave'):
            # Drizzle uses a different tracking resolution
            opts.cache += '_' + opts.light_method
        if opts.tracking_method != 'grid':
            opts.cache += '_trackm%s' % opts.tracking_method
        if opts.reference:
            opts.cache += '_ref%s' % opts.reference
        if opts.track_coarse_limit:
            opts.cache += '_trkcl%d' % opts.track_coarse_limit
        if opts.track_fine_distance:
            opts.cache += '_trkfd%d' % opts.track_fine_distance
        if opts.track_distance:
            opts.cache += '_trkd%d' % opts.track_distance
    if not os.path.exists(opts.cache):
        os.makedirs(opts.cache)

    state_cache = os.path.join(opts.cache, 'tracking_state')
    accum_cache = os.path.join(opts.cache, 'stacked')
    if opts.limit_first:
        accum_cache += '_l%d' % (opts.limit_first,)
    if opts.flat_method:
        accum_cache += '_flat%s-%s' % (opts.flat_method, opts.flat_mode)
    if opts.flat_smoothing:
        accum_cache += '_flatsmooth%s' % opts.flat_smoothing

    method_hooks = []
    add_method_hook(method_hooks, SKYGLOW_METHODS, opts.skyglow_method)
    add_method_hook(method_hooks, LIGHT_METHODS, opts.light_method)
    add_method_hook(method_hooks, FLAT_METHODS, opts.flat_method)
    add_method_hook(method_hooks, FLAT_MODES, opts.flat_mode)
    add_method_hook(method_hooks, TRACKING_METHODS, opts.tracking_method)
    add_method_hook(method_hooks, WEIGHT_METHODS, opts.weight_method)

    wiz_kwargs = create_wiz_kwargs(opts)
    invoke_method_hooks(method_hooks, 'kw', opts, pool, wiz_kwargs)

    if opts.no_normalize_weights:
        wiz_kwargs.setdefault('light_stacker_kwargs', {})['normalize_weights'] = False
    if opts.no_mirror_edges:
        wiz_kwargs.setdefault('light_stacker_kwargs', {})['mirror_edges'] = False

    wiz = WhiteBalanceWizard(**wiz_kwargs)
    invoke_method_hooks(method_hooks, 'wiz', opts, pool, wiz)

    dark_library = bias_library = None
    if opts.darklib:
        from cvastrophoto.library import darks
        dark_library = darks.DarkLibrary(opts.darklib, pool)
    if opts.biaslib:
        from cvastrophoto.library import bias
        bias_library = bias.BiasLibrary(opts.biaslib, pool)

    if not opts.darksdir or not os.path.exists(opts.darksdir):
        if opts.darksdir:
            logger.info("No darks found, using dark library")
        opts.darksdir = None
    if not opts.flatsdir or not os.path.exists(opts.flatsdir):
        if opts.flatsdir:
            logger.warning("No flats found, flat calibration is important, get some flats")
        opts.flatsdir = None
    if not opts.darkflatsdir or not os.path.exists(opts.darkflatsdir):
        if opts.darkflatsdir:
            logger.info("No darks flats found, using dark library")
        opts.darkflatsdir = None

    rops_kw = {}
    process_kw = dict(rops_kwargs=rops_kw)
    image_kw = dict(bright=opts.brightness)

    if opts.whitebalance:
        if opts.whitebalance not in wiz.WB_SETS:
            opts.whitebalance = list(map(float, opts.whitebalance.split(',')))
        rops_kw['extra_wb'] = opts.whitebalance

    if opts.preview:
        preview_image_kw = image_kw.copy()
        process_kw['preview'] = True
        process_kw['preview_kwargs'] = preview_kw = dict(image_kwargs=preview_image_kw, quick=opts.preview_quick)
        if opts.preview_path:
            preview_kw['preview_path'] = opts.preview_path
        if opts.preview_brightness:
            preview_image_kw['bright'] = opts.preview_brightness

    load_set_kw = {}
    if opts.noautodarklib:
        load_set_kw['auto_dark_library'] = None

    if opts.input_rops:
        for ropname in opts.input_rops:
            wiz.extra_input_rops.append(build_rop(ropname, opts, pool, wiz, get_factory=True))

    wiz.load_set(
        base_path=opts.path,
        light_path=opts.lightsdir, dark_path=opts.darksdir,
        flat_path=opts.flatsdir, dark_flat_path=opts.darkflatsdir,
        dark_library=dark_library, bias_library=bias_library,
        **load_set_kw)
    invoke_method_hooks(method_hooks, 'postload', opts, pool, wiz)

    if opts.selection_method:
        from cvastrophoto.wizards import selection

        sel_kw = dict(best_ratio=opts.select_percent_best / 100.0)
        sel_wiz = selection.SubSelectionWizard(**sel_kw)
        sel_wiz.load_set(wiz.light_stacker.lights, dark_library=dark_library)
        wiz.light_stacker.lights[:] = [
            light
            for i, light in sel_wiz.select(wiz.light_stacker.lights)
        ]

    if opts.reference:
        names = [os.path.basename(light.name) for light in wiz.light_stacker.lights]
        wiz.set_reference_frame(names.index(opts.reference))

    if opts.limit_first:
        del wiz.light_stacker.lights[opts.limit_first:]

    if opts.output_rops:
        for ropname in opts.output_rops:
            wiz.extra_output_rops.append(build_rop(ropname, opts, pool, wiz))

    if opts.preskyglow_rops:
        for ropname in opts.preskyglow_rops:
            wiz.preskyglow_rops.append(build_rop(ropname, opts, pool, wiz))

    if opts.skyglow_preprocessing_rops:
        from cvastrophoto.rops.compound import CompoundRop
        rops = []
        for ropname in opts.skyglow_preprocessing_rops:
            rops.append(build_rop(ropname, opts, pool, wiz))
        wiz.skyglow.preprocessing_rop = CompoundRop(wiz.skyglow.raw, *rops)

    if opts.flat_smoothing:
        wiz.vignette.gauss_size = opts.flat_smoothing

    if os.path.exists(state_cache):
        try:
            wiz.load_state(path=state_cache)
        except Exception:
            logger.warning("Could not load state cache, rebuilding")
    elif isinstance(wiz.light_stacker.lights[0], raw.Raw):
        wiz.detect_bad_pixels(
            include_darks=opts.darkbadmap,
            include_lights=[wiz.light_stacker.lights],
            max_samples_per_set=6 if opts.darkbadmap else 8)

    accum_loaded = False
    if os.path.exists(accum_cache + '.meta'):
        try:
            wiz.load_accum(accum_cache)
        except Exception:
            logger.exception("Could not load stack cache, will re-stack")
        else:
            accum_loaded = True

    if not accum_loaded:
        wiz.process(**process_kw)

        try:
            wiz.save_state(path=state_cache)
        except Exception:
            logger.exception("Could not save state cache, will not be able to reuse")

        try:
            wiz.save_accum(accum_cache)
        except Exception:
            logger.exception("Could not save stack cache, will not be able to reuse")
    else:
        wiz.process_rops(**rops_kw)

    save_kw = image_kw.copy()
    if opts.hdr:
        save_kw['hdr'] = True if not opts.hdr_stops else opts.hdr_stops

    wiz.save(opts.output, **save_kw)


def setup_drizzle_kw(opts, pool, kwargs, params):
    from cvastrophoto.wizards import stacking

    kwargs['light_stacker_kwargs'] = dict(light_method=stacking.DrizzleStackingMethod)


def setup_interleave_kw(opts, pool, kwargs, params):
    from cvastrophoto.wizards import stacking

    kwargs['light_stacker_kwargs'] = dict(light_method=stacking.InterleaveStackingMethod)


def setup_light_method_kw(method_name, opts, pool, kwargs, params):
    from cvastrophoto.wizards import stacking

    kwargs['light_stacker_kwargs'] = dict(light_method=getattr(stacking, method_name))


def setup_flat_method_kw(method_name, opts, pool, kwargs, params):
    from cvastrophoto.wizards import stacking

    kwargs['flat_stacker_kwargs'] = dict(light_method=getattr(stacking, method_name))


def get_rop(package_name, method_name, params):
    import importlib
    package = importlib.import_module('cvastrophoto.rops.' + package_name)
    method_class = getattr(package, method_name)
    if params:
        method_class = partial(method_class, **params)
    return method_class


def setup_rop_kw(argname, package_name, method_name, opts, pool, kwargs, params):
    kwargs[argname] = get_rop(package_name, method_name, params)


def add_kw(add_kw, opts, pool, kwargs, params):
    kwargs.update(add_kw)


def add_stacking_kw(stackargname, argname, package_name, method_name, opts, pool, kwargs, params):
    kwargs.setdefault(stackargname, {})[argname] = get_rop(package_name, method_name, params)


def setup_drizzle_wiz_postload(opts, pool, wiz, params):
    if hasattr(wiz.skyglow, 'minfilter_size'):
        wiz.skyglow.minfilter_size *= 2
        wiz.skyglow.gauss_size *= 2
        wiz.skyglow.luma_minfilter_size *= 2
        wiz.skyglow.luma_gauss_size *= 2


def add_output_rop(package_name, method_name, opts, pool, wiz, params, get_factory=False):
    cls = get_rop(package_name, method_name, params)
    if get_factory:
        return cls
    else:
        return cls(wiz.skyglow.raw)


LIGHT_METHODS = {
    'average': dict(kw=partial(setup_light_method_kw, 'AverageStackingMethod')),
    'median': dict(kw=partial(setup_light_method_kw, 'MedianStackingMethod')),
    'adaptive': dict(kw=partial(setup_light_method_kw, 'AdaptiveWeightedAverageStackingMethod')),
    'drizzle': dict(
        kw=partial(setup_light_method_kw, 'DrizzleStackingMethod'),
        postload=setup_drizzle_wiz_postload),
    'drizzle2x': dict(
        kw=partial(setup_light_method_kw, 'Drizzle2xStackingMethod'),
        postload=setup_drizzle_wiz_postload),
    'drizzle3x': dict(
        kw=partial(setup_light_method_kw, 'Drizzle3xStackingMethod'),
        postload=setup_drizzle_wiz_postload),
    'interleave': dict(
        kw=partial(setup_light_method_kw, 'InterleaveStackingMethod'),
        postload=setup_drizzle_wiz_postload),
    'interleave2x': dict(
        kw=partial(setup_light_method_kw, 'Interleave2xStackingMethod'),
        postload=setup_drizzle_wiz_postload),
    'interleave3x': dict(
        kw=partial(setup_light_method_kw, 'Interleave3xStackingMethod'),
        postload=setup_drizzle_wiz_postload),
}

FLAT_METHODS = {
    'average': dict(kw=partial(setup_flat_method_kw, 'AverageStackingMethod')),
    'median': dict(kw=partial(setup_flat_method_kw, 'MedianStackingMethod')),
    'min': dict(kw=partial(setup_flat_method_kw, 'MinStackingMethod')),
    'max': dict(kw=partial(setup_flat_method_kw, 'MaxStackingMethod')),
}

FLAT_MODES = {
    'gray': dict(),
    'color': dict(kw=partial(setup_rop_kw, 'vignette_class', 'vignette.flats', 'ColorFlatImageRop')),
}

ROPS = {
    'nr:diffusion': partial(add_output_rop, 'denoise.diffusion', 'DiffusionRop'),
    'nr:tv': partial(add_output_rop, 'denoise.skimage', 'TVDenoiseRop'),
    'nr:wavelet': partial(add_output_rop, 'denoise.skimage', 'WaveletDenoiseRop'),
    'nr:bilateral': partial(add_output_rop, 'denoise.skimage', 'BilateralDenoiseRop'),
    'abr:localgradient': partial(add_output_rop, 'bias.localgradient', 'LocalGradientBiasRop'),
    'abr:uniform': partial(add_output_rop, 'bias.uniform', 'UniformBiasRop'),
    'norm:fullstat': partial(add_output_rop, 'normalization.background', 'FullStatsNormalizationRop'),
    'norm:bgstat': partial(add_output_rop, 'normalization.background', 'BackgroundNormalizationRop'),
    'sharp:drizzle_deconvolution': partial(add_output_rop, 'sharpening.deconvolution', 'DrizzleDeconvolutionRop'),
    'sharp:gaussian_deconvolution': partial(add_output_rop, 'sharpening.deconvolution', 'GaussianDeconvolutionRop'),
    'sharp:double_gaussian_deconvolution': partial(
        add_output_rop, 'sharpening.deconvolution', 'DoubleGaussianDeconvolutionRop'),
    'sharp:airy_deconvolution': partial(
        add_output_rop, 'sharpening.deconvolution', 'AiryDeconvolutionRop'),
    'stretch:hdr': partial(add_output_rop, 'stretch.hdr', 'HDRStretchRop'),
    'color:convert': partial(add_output_rop, 'colorspace.convert', 'ColorspaceConversionRop'),
}

SKYGLOW_METHODS = {
    'no': dict(kw=partial(setup_rop_kw, 'skyglow_class', 'base', 'NopRop')),
    'localgradient': dict(kw=partial(setup_rop_kw, 'skyglow_class', 'bias.localgradient', 'LocalGradientBiasRop')),
}

TRACKING_METHODS = {
    'no': dict(kw=partial(add_kw, dict(tracking_class=None))),
    'grid': dict(kw=partial(setup_rop_kw, 'tracking_class', 'tracking.grid', 'GridTrackingRop')),
    'correlation': dict(kw=partial(setup_rop_kw, 'tracking_class', 'tracking.correlation', 'CorrelationTrackingRop')),
}

SELECTION_METHODS = {
    'focus': dict(kw=partial(setup_rop_kw, 'selection_class', 'measures.focus', 'FocusMeasureRop')),
}

WEIGHT_METHODS = {
    'focus': dict(kw=partial(
        add_stacking_kw, 'light_stacker_kwargs', 'weight_class', 'measures.focus', 'FocusMeasureRop')),
    'snr': dict(kw=partial(
        add_stacking_kw, 'light_stacker_kwargs', 'weight_class', 'measures.stats', 'SNRMeasureRop')),
}
