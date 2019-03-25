class BaseRop(object):

    _rmask = _gmask = _bmask = None
    _rmask_image = _gmask_image = _bmask_image = None

    def __init__(self, raw=None):
        self.raw = raw

    @property
    def _raw_pattern(self):
        # otherwise the pattern might not be fully initialized
        self.raw.postprocessed

        return self.raw.rimg.raw_pattern

    @property
    def _raw_colors(self):
        # otherwise the pattern might not be fully initialized
        self.raw.postprocessed

        return self.raw.rimg.raw_colors

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


class NopRop(BaseRop):

    def __init__(self, raw=None):
        super(NopRop, self).__init__(raw)

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        return data
