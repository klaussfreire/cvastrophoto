# -*- coding: utf-8 -*-
from __future__ import print_function

import os.path
import logging
import multiprocessing.pool
from functools import partial

logger = logging.getLogger(__name__)


def add_tracking_opts(subp, ap):
    ap.add_argument('--trackphases', type=int,
        help='Enable multiphase tracking. Higher numbers create more phases. The default should be fine',
        default=1)
    ap.add_argument('--track-refinement-phases', type=int,
        help='Extra refinement tracking phases.',
        default=0)
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
    ap.add_argument('--comet-tracking', action='store_true',
        help=(
            "Enables comet tracking after regular star tracking. Improves on merely selecting a comet tracking method "
            "in that it first aligns stars, matching rotation accurately, before tracking the comet."
        ))
    ap.add_argument('--comet-tracking-params',
        help=(
            "Customize comet tracking phase. Comet tracking must be enabled, otherwise it will be ignored."
        ))


def add_opts(subp):
    ap = subp.add_parser('process', help="Process a session's data")

    ap.add_argument('--config', help='Load config from a file', default=None)

    ap.add_argument('--darklib', help='Location of the main dark library', default=None)
    ap.add_argument('--biaslib', help='Location of the bias library', default=None)
    ap.add_argument('--noautodarklib', help="Use darks as they are, don't build a local library",
        default=False, action='store_true')
    ap.add_argument('--dark-annot', help='Print out for each light/flat which dark frame will be applied',
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
    ap.add_argument('--light-pedestal', help='Adds a constant to avoid losing dark detail to dark variance', type=int)
    ap.add_argument('--flat-method', '-mf', help='Set flat stacking method', default='median',
        choices=FLAT_METHODS.keys())
    ap.add_argument('--flat-mode', '-mfm', help='Set flat calibration mode', default='color')
    ap.add_argument('--flat-smoothing', type=float, help='Set flat smoothing radius, recommended for high-iso')
    ap.add_argument('--flat-pattern', type=int,
        help=(
            'Set flat smoothing pattern size, recommended for certain mono sensors that exhibit CFA-like PRNU patterns. '
            'Usually, when necessary, the value 2 works.'
        ))
    ap.add_argument('--flat-rops', nargs='+', help='Set flat preprocessing ROPs, recommended to apply NR at high-iso')
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
    ap.add_argument('--weights-file', '-W',
        help=(
            "Set the weights file, a CSV mapping light filename to sub weight. "
            "By default, it checks weights.csv."
        ))

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
    ap.add_argument('--tracking-preprocessing-rops', '-Rtpp', nargs='+')
    ap.add_argument('--tracking-color-rops', '-Rtcp', nargs='+')

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
        tracking_refinement_phases=opts.track_refinement_phases,
    )
    if opts.parallel:
        wiz_kwargs['pool'] = multiprocessing.pool.ThreadPool(opts.parallel)
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
    if opts.comet_tracking:
        from cvastrophoto.rops.tracking import correlation
        comet_kw = {}
        if opts.track_fine_distance:
            comet_kw['track_distance'] = opts.track_fine_distance
        if opts.comet_tracking_params:
            comet_kw.update(parse_params(opts.comet_tracking_params))
        wiz_kwargs['tracking_post_class'] = partial(correlation.CometTrackingRop, **comet_kw)

    return wiz_kwargs


def noop(*p, **kw):
    pass

def boolean(val):
    return val.lower() in ('1', 'true', 'yes')

CONFIG_TYPES = {
    'trackphases': int,
    'track_refinement_phases': int,
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
    for k, v in params.items():
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

def annotate_calibration(dark_library, bias_library, lights):
    darkless = []
    biasless = []

    for light in lights:
        if dark_library:
            dark_class = dark_library.classify_frame(light.name)
            dark = dark_library.get_master(dark_class, raw=light)
            if dark is None:
                darkless.append(light)
        else:
            dark = dark_class = None

        if bias_library:
            bias_class = bias_library.classify_frame(light.name)
            bias = bias_library.get_master(bias_class, raw=light)
            if dark is None and bias is None:
                biasless.append(light)
        else:
            bias = bias_class = None

        logger.info("Light: %r", light.name)
        logger.info("  dark: %r", dark.name if dark is not None else "N/A")
        logger.info("  bias: %r", bias.name if bias is not None else "N/A")
        logger.info("  dark-class: %r", dark_class)
        logger.info("  bias-class: %r", bias_class)

    if darkless:
        logger.warning("Darkless: %d", len(darkless))
        for light in darkless:
            logger.warning("  %r", light.name)

    if biasless:
        logger.warning("Biasless: %d", len(biasless))
        for light in biasless:
            logger.warning("  %r", light.name)

def load_weights_file(fname):
    weights = {}
    with open(fname, "r") as f:
        import csv
        for row in csv.reader(f):
            if len(row) >= 2:
                weights[row[0]] = float(row[1])
    return weights

def build_compound_rop(opts, pool, wiz, raw, rops_desc, **kw):
    from cvastrophoto.rops.compound import CompoundRop
    rops = []
    for ropname in rops_desc:
        rops.append(build_rop(ropname, opts, pool, wiz, raw=raw, **kw))
    return CompoundRop(raw, *rops)

def main(opts, pool):
    from cvastrophoto.wizards.whitebalance import WhiteBalanceWizard
    from cvastrophoto.image import raw, rgb
    from cvastrophoto.rops.vignette import flats

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
        for name, coeffs in WhiteBalanceWizard.WB_SETS.items():
            print(name, ':', coeffs)

    if opts.cache is None:
        opts.cache = '.cvapstatecache/state_cache'
        opts.cache += '_track%dp' % opts.trackphases
        if opts.track_refinement_phases:
            opts.cache += '_trefine%d' % opts.track_refinement_phases
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
        if opts.comet_tracking:
            opts.cache += '_comet'
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
        if opts.flat_pattern:
            accum_cache += 'pat%s' % opts.flat_pattern

    method_hooks = []
    add_method_hook(method_hooks, SKYGLOW_METHODS, opts.skyglow_method)
    add_method_hook(method_hooks, LIGHT_METHODS, opts.light_method)
    add_method_hook(method_hooks, FLAT_METHODS, opts.flat_method)
    add_method_hook(method_hooks, FLAT_MODES, opts.flat_mode)
    add_method_hook(method_hooks, TRACKING_METHODS, opts.tracking_method)
    add_method_hook(method_hooks, WEIGHT_METHODS, opts.weight_method)

    wiz_kwargs = create_wiz_kwargs(opts)
    invoke_method_hooks(method_hooks, 'kw', opts, pool, wiz_kwargs)

    if opts.tracking_preprocessing_rops:
        import cvastrophoto.rops.tracking.grid
        tracking_class = wiz_kwargs.get('tracking_class', cvastrophoto.rops.tracking.grid.GridTrackingRop)
        luma_pp_rop = build_compound_rop(
            opts, pool, None, rgb.Templates.LUMINANCE, opts.tracking_preprocessing_rops)
        tracking_class = partial(tracking_class, luma_preprocessing_rop=luma_pp_rop)
        wiz_kwargs['tracking_class'] = tracking_class

    if opts.tracking_color_rops:
        import cvastrophoto.rops.tracking.grid
        base_tracking_class = wiz_kwargs.get('tracking_class', cvastrophoto.rops.tracking.grid.GridTrackingRop)
        def tracking_class(raw, *p, **kw):
            kw['color_preprocessing_rop'] = build_compound_rop(
                opts, pool, None, kw.get('lraw', raw), opts.tracking_color_rops,
                copy=False)
            return base_tracking_class(raw, *p, **kw)
        wiz_kwargs['tracking_class'] = tracking_class

    if opts.no_normalize_weights:
        wiz_kwargs.setdefault('light_stacker_kwargs', {})['normalize_weights'] = False
    if opts.no_mirror_edges:
        wiz_kwargs.setdefault('light_stacker_kwargs', {})['mirror_edges'] = False
    if opts.light_pedestal:
        wiz_kwargs.setdefault('light_stacker_kwargs', {})['pedestal'] = opts.light_pedestal

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

    if opts.weights_file:
        load_set_kw['weights'] = load_weights_file(opts.weights_file)

    wiz.load_set(
        base_path=opts.path,
        light_path=opts.lightsdir, dark_path=opts.darksdir,
        flat_path=opts.flatsdir, dark_flat_path=opts.darkflatsdir,
        dark_library=dark_library, bias_library=bias_library,
        **load_set_kw)
    invoke_method_hooks(method_hooks, 'postload', opts, pool, wiz)

    if opts.flat_rops:
        from cvastrophoto.rops.compound import CompoundRop

        flat_rops = []
        for ropname in opts.flat_rops:
            flat_rops.append(build_rop(ropname, opts, pool, wiz, raw=wiz.vignette.raw))
        wiz.vignette.flat_rop = CompoundRop(wiz.vignette.raw, *flat_rops)

    if opts.output_rops:
        for ropname in opts.output_rops:
            wiz.extra_output_rops.append(build_rop(ropname, opts, pool, wiz))

    if opts.preskyglow_rops:
        for ropname in opts.preskyglow_rops:
            wiz.preskyglow_rops.append(build_rop(ropname, opts, pool, wiz))

    if opts.skyglow_preprocessing_rops:
        wiz.skyglow.preprocessing_rop = build_compound_rop(
            opts, pool, wiz, wiz.skyglow.raw, opts.skyglow_preprocessing_rops)

    if opts.flat_smoothing:
        wiz.vignette.gauss_size = opts.flat_smoothing
    if opts.flat_pattern:
        wiz.vignette.pattern_size = opts.flat_pattern

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

    if not accum_loaded or opts.dark_annot:
        if opts.selection_method:
            from cvastrophoto.wizards import selection

            sel_method_hooks = []
            add_method_hook(sel_method_hooks, SELECTION_METHODS, opts.selection_method)

            sel_kw = dict(best_ratio=opts.select_percent_best / 100.0)
            invoke_method_hooks(sel_method_hooks, 'kw', opts, pool, sel_kw)

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

        if opts.dark_annot:
            annotate_calibration(
                wiz.flat_stacker.dark_library,
                wiz.flat_stacker.bias_library,
                wiz.flat_stacker.lights)

            annotate_calibration(
                wiz.light_stacker.dark_library,
                wiz.light_stacker.bias_library,
                wiz.light_stacker.lights)
            return

        def save_state(*p, **kw):
            try:
                wiz.save_state(path=state_cache)
            except Exception:
                logger.exception("Could not save state cache, will not be able to reuse")

        process_kw['on_phase_completed'] = save_state

        wiz.process(**process_kw)

        save_state()
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


def add_output_rop(package_name, method_name, opts, pool, wiz, params, get_factory=False, raw=None, **kw):
    cls = get_rop(package_name, method_name, params)
    if get_factory:
        if kw:
            return partial(cls, **kw)
        else:
            return cls
    else:
        return cls(wiz.skyglow.raw if raw is None else raw, **kw)


LIGHT_METHODS = {
    'average': dict(kw=partial(setup_light_method_kw, 'AverageStackingMethod')),
    'weighted': dict(kw=partial(setup_light_method_kw, 'WeightedAverageStackingMethod')),
    'median': dict(kw=partial(setup_light_method_kw, 'MedianStackingMethod')),
    'approx_median': dict(kw=partial(setup_light_method_kw, 'ApproxMedianStackingMethod')),
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
    'drizzlemedian': dict(
        kw=partial(setup_light_method_kw, 'DrizzleMedianStackingMethod'),
        postload=setup_drizzle_wiz_postload),
    'interleavemedian': dict(
        kw=partial(setup_light_method_kw, 'InterleaveMedianStackingMethod'),
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
    'nr:starlessdiffusion': partial(add_output_rop, 'denoise.diffusion', 'StarlessDiffusionRop'),
    'nr:tv': partial(add_output_rop, 'denoise.skimage', 'TVDenoiseRop'),
    'nr:wavelet': partial(add_output_rop, 'denoise.skimage', 'WaveletDenoiseRop'),
    'nr:bilateral': partial(add_output_rop, 'denoise.skimage', 'BilateralDenoiseRop'),
    'nr:debanding': partial(add_output_rop, 'denoise.debanding', 'DebandingFilterRop'),
    'nr:flatdebanding': partial(add_output_rop, 'denoise.debanding', 'FlatDebandingFilterRop'),
    'nr:starlessdebanding': partial(add_output_rop, 'denoise.debanding', 'StarlessDebandingFilterRop'),
    'neutralization:bg': partial(add_output_rop, 'denoise.neutralization', 'BackgroundNeutralizationRop'),
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
    'sharp:sampled_deconvolution': partial(
        add_output_rop, 'sharpening.deconvolution', 'SampledDeconvolutionRop'),
    'stretch:hdr': partial(add_output_rop, 'stretch.hdr', 'HDRStretchRop'),
    'stretch:linear': partial(add_output_rop, 'stretch.simple', 'LinearStretchRop'),
    'stretch:starlesslinear': partial(add_output_rop, 'stretch.starless', 'StarlessLinearStretchRop'),
    'stretch:starlesshdr': partial(add_output_rop, 'stretch.starless', 'StarlessHDRStretchRop'),
    'color:convert': partial(add_output_rop, 'colorspace.convert', 'ColorspaceConversionRop'),
    'color:extract': partial(add_output_rop, 'colorspace.extract', 'ExtractChannelRop'),
    'color:wb': partial(add_output_rop, 'colorspace.whitebalance', 'WhiteBalanceRop'),
    'extract:stars': partial(add_output_rop, 'tracking.extraction', 'ExtractPureStarsRop'),
    'extract:starstuff': partial(add_output_rop, 'tracking.extraction', 'ExtractStarsRop'),
    'extract:starless': partial(add_output_rop, 'tracking.extraction', 'RemoveStarsRop'),
    'extract:wtophat': partial(add_output_rop, 'tracking.extraction', 'WhiteTophatRop'),
}

SKYGLOW_METHODS = {
    'no': dict(kw=partial(setup_rop_kw, 'skyglow_class', 'base', 'NopRop')),
    'localgradient': dict(kw=partial(setup_rop_kw, 'skyglow_class', 'bias.localgradient', 'LocalGradientBiasRop')),
    'uniform': dict(kw=partial(setup_rop_kw, 'skyglow_class', 'bias.uniform', 'UniformBiasRop')),
}

TRACKING_METHODS = {
    'no': dict(kw=partial(add_kw, dict(tracking_class=None))),
    'grid': dict(kw=partial(setup_rop_kw, 'tracking_class', 'tracking.grid', 'GridTrackingRop')),
    'correlation': dict(kw=partial(setup_rop_kw, 'tracking_class', 'tracking.correlation', 'CorrelationTrackingRop')),
    'comet': dict(kw=partial(setup_rop_kw, 'tracking_class', 'tracking.correlation', 'CometTrackingRop')),
}

SELECTION_METHODS = {
    'focus': dict(kw=partial(setup_rop_kw, 'selection_class', 'measures.focus', 'FocusMeasureRop')),
    'seeing': dict(kw=partial(setup_rop_kw, 'selection_class', 'measures.seeing', 'SeeingMeasureRop')),
    'seeing+focus': dict(kw=partial(setup_rop_kw, 'selection_class', 'measures.seeing', 'SeeingFocusRankingRop')),
    'fwhm': dict(kw=partial(setup_rop_kw, 'selection_class', 'measures.fwhm', 'FWHMMeasureRop')),
}

WEIGHT_METHODS = {
    'focus': dict(kw=partial(
        add_stacking_kw, 'light_stacker_kwargs', 'weight_class', 'measures.focus', 'FocusMeasureRop')),
    'snr': dict(kw=partial(
        add_stacking_kw, 'light_stacker_kwargs', 'weight_class', 'measures.stats', 'SNRMeasureRop')),
}
