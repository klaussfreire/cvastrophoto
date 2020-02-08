import logging

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
        for k, v in kw.iteritems():
            if hasattr(cls, k):
                defv = getattr(self, k)
                if isinstance(defv, bool):
                    setattr(self, k, bool(int(v)))
                elif isinstance(defv, (int, float, basestring)):
                    setattr(self, k, type(defv)(v))

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

    def demargin(self, accum, raw_pattern=None, sizes=None):
        if raw_pattern is None:
            raw_pattern = self._raw_pattern
        if sizes is None:
            sizes = self._raw_sizes
        return self.raw.demargin(accum, raw_pattern=raw_pattern, sizes=sizes)

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
        b += path - 1 - (b % path)
        r += patw - 1 - (r % path)

        return t, l, b, r

    def roi_precrop(self, roi, data):
        t, l, b, r = eff_roi = self.effective_roi(roi)
        return eff_roi, data[t:b, l:r]

    def roi_postcrop(self, roi, eff_roi, data):
        t, l, b, r = roi
        et, el, eb, er = eff_roi
        return data[t-et:b-et,l-el:r-el]

class NopRop(BaseRop):

    def __init__(self, raw=None):
        super(NopRop, self).__init__(raw)

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        return data

class PerChannelRop(BaseRop):

    def process_channel(self, channel_data, detected=None):
        raise NotImplementedError

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        if self.raw.default_pool is not None:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        path, patw = self._raw_pattern.shape

        roi = kw.get('roi')

        def process_channel(task):
            try:
                data, y, x = task
                if roi is not None:
                    data, eff_roi = self.roi_precrop(roi, data)
                data[y::path, x::patw] = self.process_channel(data[y::path, x::patw], detected)
                if roi is not None:
                    data = self.roi_postcrop(roi, eff_roi, data)
            except Exception:
                logger.exception("Error processing channel data")
                raise

        rv = data

        if not isinstance(data, list):
            data = [data]

        for sdata in data:
            if sdata is None:
                continue

            tasks = []
            for y in xrange(path):
                for x in xrange(patw):
                    tasks.append((sdata, y, x))

        for _ in map_(process_channel, tasks):
            pass

        return rv
