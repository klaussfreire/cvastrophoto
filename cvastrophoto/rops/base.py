class BaseRop(object):

    _rmask = _gmask = _bmask = None
    _rmask_image = _gmask_image = _bmask_image = None
    _raw_pattern_cached = _raw_colors_cached = _raw_sizes_cached = None

    def __init__(self, raw=None, copy=True):
        self.raw = raw.dup() if copy else raw

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

class NopRop(BaseRop):

    def __init__(self, raw=None):
        super(NopRop, self).__init__(raw)

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        return data
