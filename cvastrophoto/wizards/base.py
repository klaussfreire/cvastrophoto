from __future__ import absolute_import

import numpy

class BaseWizard:

    pool = None

    def _get_raw_instance(self):
        raise NotImplementedError

    def get_hdr_set(self, steps, size=8, **kw):
        # Optional imports, so do locally
        import skimage.morphology
        import skimage.filters.rank

        # Get the different exposure steps
        iset = []
        scale = kw.pop('bright', 1.0)
        for step in steps:
            img = self.get_image(bright=scale*step, **kw).postprocessed
            iset.append((step, img))

        # Compute local entropy weights
        selem = skimage.morphology.disk(size)

        def append_entropy(entry):
            step, img = entry
            gray = numpy.average(img, axis=2).astype(img.dtype)
            gray = numpy.right_shift(gray, 8, out=gray)
            gray = numpy.clip(gray, 0, 255, out=gray)
            ent = skimage.filters.rank.entropy(gray.astype(numpy.uint8), selem).astype(numpy.float32)
            return (step, img, ent)

        if self.pool is not None:
            map_ = self.pool.map
        else:
            map_ = map
        iset = list(map_(append_entropy, iset))

        # Fix all-zero weights
        min_ent = iset[0][2].copy()
        for step, img, ent in iset[1:]:
            min_ent = numpy.minimum(min_ent, ent)

        if min_ent.min() <= 0:
            # All-zero weights happen with always-saturated pixels
            clippers = min_ent == 0
            for step, img, ent in iset:
                ent[clippers] = 1

        return iset

    def get_hdr_image(self, steps, size=8, **kw):
        iset = self.get_hdr_set(steps, size, **kw)

        # Do the entropy-weighted average
        step, img, ent = iset[0]
        hdr_img = numpy.zeros(img.shape, numpy.float32)
        ent_sum = numpy.zeros(ent.shape, ent.dtype)
        for step, img, ent in iset:
            for c in xrange(hdr_img.shape[2]):
                hdr_img[:,:,c] += img[:,:,c] * ent
                ent_sum += ent
        if ent_sum.min() <= 0:
            ent_sum[ent_sum <= 0] = 1
        for c in xrange(hdr_img.shape[2]):
            hdr_img[:,:,c] /= ent_sum
        hdr_img *= 65535.0 / max(1, hdr_img.max())
        hdr_img = numpy.clip(hdr_img, 0, 65535, out=hdr_img)

        img = self._get_raw_instance()
        img.postprocessed[:] = hdr_img
        return img

    def get_image(self, bright=1.0, gamma=2.4, hdr=False):
        if hdr:
            if hdr is True:
                hdr = 6
            if isinstance(hdr, int):
                hdr = [1, 2, 4, 8, 16, 32][:hdr]
            return self.get_hdr_image(hdr, bright=bright, gamma=gamma)

        img = self._get_raw_instance()

        accum = self.accum
        accum = accum.astype(numpy.float32) * (float(bright) / accum.max())
        accum = numpy.clip(accum, 0, 1, out=accum)

        img.set_raw_image(accum * 65535, add_bias=True)

        return img
