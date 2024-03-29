import logging
import numpy

from six import iteritems
from past.builtins import xrange, basestring

logger = logging.getLogger(__name__)

class BaseRop(object):

    PROCESSING_MARGIN = 0

    _rmask = _gmask = _bmask = None
    _rmask_image = _gmask_image = _bmask_image = None
    _raw_pattern_cached = _raw_colors_cached = _raw_sizes_cached = None

    def __init__(self, raw=None, copy=True, **kw):
        self.raw = raw.dup() if copy else raw

        # Generic way to set simple parameters at construction time
        cls = type(self)
        for k, v in iteritems(kw):
            if hasattr(cls, k):
                defv = getattr(self, k)
                if isinstance(defv, bool):
                    setattr(self, k, bool(int(v)))
                elif isinstance(defv, (int, float, basestring)):
                    setattr(self, k, type(defv)(v))
            else:
                logger.warn("Ignored unrecognized parameter %s of %s", k, cls.__name__)

    def init_pattern(self):
        self._raw_pattern
        self._raw_sizes

    def __str__(self):
        cls = type(self)
        kw = {
            k: getattr(self, k)
            for k, v in vars(cls).items()
            if isinstance(v, (basestring, int, float, bool)) and hasattr(self, k) and getattr(self, k) != v
        }
        return "%s(%s)" % (cls.__name__, ", ".join(["%s=%r" % (k, v) for k, v in kw.items()]))

    @property
    def _raw_pattern(self):
        if self._raw_pattern_cached is None:
            # otherwise the pattern might not be fully initialized
            self.raw.postprocessed

            self._raw_pattern_cached = self.raw.rimg.raw_pattern
        return self._raw_pattern_cached

    @property
    def _raw_colors(self):
        if self._raw_colors_cached is None:
            # otherwise the pattern might not be fully initialized
            self.raw.postprocessed

            self._raw_colors_cached = self.raw.rimg.raw_colors
        return self._raw_colors_cached

    @property
    def _raw_sizes(self):
        if self._raw_sizes_cached is None:
            # otherwise the info might not be fully initialized
            self.raw.postprocessed

            self._raw_sizes_cached = self.raw.rimg.sizes
        return self._raw_sizes_cached

    @property
    def rmask(self):
        if self._rmask is None:
            self._rmask = self._raw_pattern == 0
        return self._rmask

    @property
    def gmask(self):
        if self._gmask is None:
            self._gmask = self._raw_pattern == 1
        return self._gmask

    @property
    def bmask(self):
        if self._bmask is None:
            self._bmask = self._raw_pattern == 2
        return self._bmask

    @property
    def rmask_image(self):
        if self._rmask_image is None:
            self._rmask_image = self._raw_colors == 0
        return self._rmask_image

    @property
    def gmask_image(self):
        if self._gmask_image is None:
            self._gmask_image = self._raw_colors == 1
        return self._gmask_image

    @property
    def bmask_image(self):
        if self._bmask_image is None:
            self._bmask_image = self._raw_colors == 2
        return self._bmask_image

    def get_state(self):
        return None

    def load_state(self, state):
        pass

    def demargin(self, accum, raw_pattern=None, sizes=None, raw=None):
        if raw_pattern is None:
            raw_pattern = self._raw_pattern
        if sizes is None:
            sizes = self._raw_sizes
        if raw is None:
            raw = self.raw
        if sizes:
            raw_shape = (sizes.raw_height, sizes.raw_width)
            if raw_shape != accum.shape:
                logger.warning(
                    "Demargin invoked on mismatching shapes (expected %r got %r)",
                    raw_shape, accum.shape)
        return raw.demargin(accum, raw_pattern=raw_pattern, sizes=sizes)

    def effective_roi(self, roi):
        t, l, b, r = roi
        path, patw = self._raw_pattern.shape

        # Add margin
        margin = self.PROCESSING_MARGIN
        t = max(0, t - margin)
        l = max(0, l - margin)
        b += margin
        r += margin

        # Round to pattern boundaries
        t -= t % path
        l -= l % patw
        b += (path - (b % path)) % path
        r += (patw - (r % path)) % patw

        return t, l, b, r

    def roi_precrop(self, roi, data):
        t, l, b, r = eff_roi = self.effective_roi(roi)
        return eff_roi, data[t:b, l:r]

    def roi_postcrop(self, roi, eff_roi, data):
        t, l, b, r = roi
        et, el, eb, er = eff_roi
        return data[t-et:b-et,l-el:r-el]

    def parallel_channel_task(self, data, dest, fn, *p, **kw):
        if self.raw.default_pool is not None:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        raw_pattern = kw.pop('raw_pattern', None)
        if raw_pattern is None:
            raw_pattern = self._raw_pattern
        path, patw = raw_pattern.shape

        if len(data.shape) == 3:
            dedupe_channels = True
            def cfn(task):
                data, y, x = task
                dest[:,:,raw_pattern[y,x]] = fn(data[:,:,raw_pattern[y,x]], *p, **kw)
        else:
            dedupe_channels = False
            def cfn(task):
                data, y, x = task
                dest[y::path, x::patw] = fn(data[y::path, x::patw], *p, **kw)

        tasks = []
        channels_done = set()
        for y in xrange(path):
            for x in xrange(patw):
                if not dedupe_channels or raw_pattern[y,x] not in channels_done:
                    tasks.append((data, y, x))
                    channels_done.add(raw_pattern[y,x])

        for _ in map_(cfn, tasks):
            pass

class NopRop(BaseRop):

    def __init__(self, raw=None):
        super(NopRop, self).__init__(raw)

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        return data

class PerChannelRop(BaseRop):

    pre_demargin = False
    single_channel = -1
    parallel_channels = True

    _PASS_PROCESS_KW = {'img'}

    def process_channel(self, channel_data, detected=None, channel=None):
        raise NotImplementedError

    def detect(self, data, **kw):
        pool = kw.get('pool', self.raw.default_pool)
        if pool is not None and self.parallel_channels:
            map_ = pool.imap_unordered
        else:
            map_ = map

        raw_pattern = self._raw_pattern
        path, patw = raw_pattern.shape

        roi = kw.get('roi')
        detect_method = kw.get('detect_method', self.detect_channel)

        rv = {}

        if not isinstance(data, list):
            data = [data]

        if len(data[0].shape) == 3:
            demargin_safe = False
            def process_channel(task):
                try:
                    data, y, x = task
                    if roi is not None:
                        eff_roi, data = self.roi_precrop(roi, data)
                    rv[y,x] = detect_method(data[:,:,raw_pattern[y, x]], channel=(y, x))
                except Exception:
                    logger.exception("Error processing channel data")
                    raise
        else:
            demargin_safe = self.raw.demargin_safe
            def process_channel(task):
                try:
                    data, y, x = task
                    if roi is not None:
                        eff_roi, data = self.roi_precrop(roi, data)
                    rv[y,x] = detect_method(data[y::path, x::patw], channel=(y, x))
                except Exception:
                    logger.exception("Error processing channel data")
                    raise

        for sdata in data:
            if sdata is None:
                continue

            if self.pre_demargin and demargin_safe:
                self.demargin(sdata)

            tasks = []
            for y in xrange(path):
                for x in xrange(patw):
                    if self.single_channel != -1 and raw_pattern[y,x] != self.single_channel:
                        continue
                    tasks.append((sdata, y, x))

        for _ in map_(process_channel, tasks):
            pass

        return rv

    def detect_channel(self, channel_data, channel=None):
        pass

    def correct(self, data, detected=None, **kw):
        logger.debug("Applying %s", self)

        pool = kw.get('pool', self.raw.default_pool)
        if pool is not None:
            map_ = pool.imap_unordered
        else:
            map_ = map

        if detected is None:
            detected = self.detect(data, **kw)

        raw_pattern = self._raw_pattern
        path, patw = raw_pattern.shape

        roi = kw.pop('roi', None)
        process_method = kw.pop('process_method', self.process_channel)
        rv_method = kw.pop('rv_method', None)

        rv = data

        if not isinstance(data, list):
            data = [data]

        process_kw = {k: v for k, v in kw.items() if k in self._PASS_PROCESS_KW}

        if len(data[0].shape) == 3:
            dedupe_channels = True
            def process_channel(task):
                try:
                    data, y, x = task
                    if roi is not None:
                        eff_roi, data = self.roi_precrop(roi, data)
                    processed = process_method(data[:,:,raw_pattern[y, x]], detected, channel=(y, x), **process_kw)

                    if (hasattr(processed, 'dtype') and processed.shape and processed.dtype != data.dtype
                            and data.dtype.kind in ('i', 'u')):
                        limits = numpy.iinfo(data.dtype)
                        processed = numpy.clip(processed, limits.min, limits.max, out=processed)

                    if rv_method is None:
                        data[:,:,raw_pattern[y, x]] = processed
                    else:
                        rv_method(data, y, x, processed)
                    del processed

                    if roi is not None:
                        data = self.roi_postcrop(roi, eff_roi, data)
                except Exception:
                    logger.exception("Error processing channel data")
                    raise
        else:
            dedupe_channels = False
            def process_channel(task):
                try:
                    data, y, x = task
                    if roi is not None:
                        eff_roi, data = self.roi_precrop(roi, data)
                    processed = process_method(data[y::path, x::patw], detected, channel=(y, x), **process_kw)

                    if (hasattr(processed, 'dtype') and processed.shape and processed.dtype != data.dtype
                            and data.dtype.kind in ('i', 'u')):
                        limits = numpy.iinfo(data.dtype)
                        processed = numpy.clip(processed, limits.min, limits.max, out=processed)

                    if rv_method is None:
                        data[y::path, x::patw] = processed
                    else:
                        rv_method(data, y, x, processed)
                    del processed

                    if roi is not None:
                        data = self.roi_postcrop(roi, eff_roi, data)
                except Exception:
                    logger.exception("Error processing channel data")
                    raise

        for sdata in data:
            if sdata is None:
                continue

            if self.pre_demargin and self.raw.demargin_safe:
                self.demargin(sdata)

            tasks = []
            channels_done = set()
            for y in xrange(path):
                for x in xrange(patw):
                    if self.single_channel != -1 and raw_pattern[y,x] != self.single_channel:
                        continue
                    if not dedupe_channels or raw_pattern[y,x] not in channels_done:
                        tasks.append((sdata, y, x))
                        channels_done.add(raw_pattern[y,x])

        for _ in map_(process_channel, tasks):
            pass

        return rv
