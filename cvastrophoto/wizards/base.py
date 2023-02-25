from __future__ import absolute_import

from past.builtins import xrange
import numpy
import imageio
import gzip
import logging
import os.path

try:
    import cPickle
except ImportError:
    import pickle as cPickle

from cvastrophoto.util import srgb, entropy
from cvastrophoto.image.base import ImageAccumulator
from cvastrophoto.image.metaimage import MetaImage

logger = logging.getLogger(__name__)

class BaseWizard(object):

    pool = None

    def _get_raw_instance(self):
        raise NotImplementedError

    def get_hdr_set(self, steps, size=32, **kw):
        # Optional imports, so do locally
        import skimage.morphology
        import skimage.filters.rank

        # Get the different exposure steps
        iset = []
        scale = kw.pop('bright', 1.0)
        gamma = kw.get('gamma', 2.4)
        for step in steps:
            img = self.get_image(bright=scale*step, **kw).postprocessed.copy()
            iset.append((step, img))

        # Compute local entropy weights
        selem = skimage.morphology.disk(size)

        def append_entropy(entry):
            try:
                step, img = entry
                gray = numpy.average(img.astype(numpy.float32), axis=2)
                ent = entropy.local_entropy(gray, selem=selem, gamma=gamma, copy=False)
                return (step, img, ent)
            except:
                logger.exception("Error in append_entropy")
                raise

        if self.pool is not None:
            map_ = self.pool.map
        else:
            map_ = map
        iset = list(map_(append_entropy, iset))

        # Fix all-zero weights
        max_ent = iset[0][2].copy()
        for step, img, ent in iset[1:]:
            max_ent = numpy.maximum(max_ent, ent)

        if max_ent.min() <= 0:
            # All-zero weights happen with always-saturated pixels
            clippers = max_ent <= 0
            for step, img, ent in iset:
                ent[clippers] = 1

        return iset

    def _get_hdr_img(self, steps, size=32, **kw):
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

        return hdr_img

    def get_hdr_image(self, *p, **kw):
        hdr_img = self._get_hdr_img(*p, **kw)

        img = self._get_raw_instance()
        img.postprocessed[:] = hdr_img
        return img

    def get_image(self, bright=1.0, gamma=2.4, hdr=False, maxval=None, accum=None):
        if hdr:
            if hdr is True:
                hdr = 6
            if isinstance(hdr, int):
                hdr = [1, 2, 4, 8, 16, 32, 64, 128, 256][:hdr]
            return self.get_hdr_image(hdr, bright=bright, gamma=gamma)

        img = self._get_raw_instance()

        if accum is None:
            accum = self.accum
        if maxval is None:
            maxval = accum.max()
        if maxval > 0:
            accum = accum.astype(numpy.float32) * (float(bright) / maxval)
        else:
            accum = accum.copy()
        accum = numpy.clip(accum, 0, 1, out=accum)
        accum *= 65535

        img.set_raw_image(accum, add_bias=True)

        return img

    def save(self, path, bright=1.0, gamma=2.4, meta=dict(compress=6), hdr=False, size=32, maxval=None):
        def normalize_srgb(accum, bright, maxval=maxval, scale=65535):
            if maxval is None:
                maxval = accum.max()
            if maxval > 0:
                in_scale = float(bright) / maxval
            else:
                in_scale = 1
            accum = accum.copy()
            accum = srgb.encode_srgb(accum, in_scale=in_scale, out_scale=scale, gamma=gamma, out_max=65535)
            return accum

        if hdr:
            if hdr is True:
                hdr = 6
            if isinstance(hdr, int):
                hdr = [1, 2, 4, 8, 16, 32, 64, 128, 256][:hdr]
            postprocessed = self._get_hdr_img(hdr, bright=bright, size=size)
            postprocessed = normalize_srgb(postprocessed, 1.0)
            img = postprocessed = postprocessed.astype(numpy.uint16, copy=False)
        else:
            img = self._get_raw_instance()
            accum = normalize_srgb(self.accum, bright)
            img.set_raw_image(accum, add_bias=True)
            postprocessed = img.postprocessed.astype(numpy.uint16, copy=False)

        with imageio.get_writer(path, mode='i', software='cvastrophoto') as writer:
            writer.append_data(postprocessed, meta)

        return img

    def load_state(self, state=None, fileobj=None, path=None, compressed=True):
        if state is not None:
            self._load_state(state)
        elif fileobj is not None:
            if compressed:
                fileobj = gzip.GzipFile(mode='rb', fileobj=fileobj)
            self._load_state(cPickle.load(fileobj))
        elif path is not None:
            with open(path, 'rb') as f:
                self.load_state(fileobj=f, compressed=compressed)
        else:
            raise ValueError("Must pass either state or path/fileobj")

    def get_state(self):
        return None

    def _load_state(self, state):
        pass

    def save_state(self, fileobj=None, path=None, compress=True):
        if fileobj is not None:
            if compress:
                fileobj = gzip.GzipFile(mode='wb', fileobj=fileobj)
            cPickle.dump(self.get_state(), fileobj, 2)
            if compress:
                fileobj.flush()
                fileobj.close()
        elif path is not None:
            with open(path, 'wb') as f:
                self.save_state(fileobj=f)

    def save_accum(self, path_prefix, compress=True, meta=dict(compress=6)):
        accumulator = self.light_stacker.accumulator
        accum = accumulator.accum
        raw = self.light_stacker.stacked_image_template

        accum_meta = dict(
            num_images=accumulator.num_images,
            shape=accum.shape,
            dtype=accum.dtype,
            image_template=raw.name,
            raw_pattern=raw.rimg.raw_pattern,
            raw_sizes=raw.sizes,
        )

        metapath = path_prefix + '.accum.meta.fits.gz'
        metaimage = self.light_stacker.metaimage
        metaimage.save(metapath, raw_pattern=raw.rimg.raw_pattern)

        with open(path_prefix + '.meta', 'wb') as fileobj:
            if compress:
                gzobj = gzip.GzipFile(mode='wb', fileobj=fileobj)
            else:
                gzobj = fileobj
            cPickle.dump(accum_meta, gzobj, 2)
            if compress:
                gzobj.flush()
                gzobj.close()

    def load_accum(self, path_prefix, compressed=True):
        if os.path.exists(path_prefix + '.accum.meta.fits.gz'):
            self.light_stacker.metaimage = MetaImage(path_prefix + '.accum.meta.fits.gz')
        else:
            with open(path_prefix + '.meta', 'rb') as fileobj:
                if compressed:
                    gzobj = gzip.GzipFile(mode='rb', fileobj=fileobj)
                else:
                    gzobj = fileobj
                accum_meta = cPickle.load(gzobj)

            accumulator = ImageAccumulator(accum_meta['dtype'])
            accumulator += imageio.imread(path_prefix + '.accum.tiff')
            accumulator.num_images = accum_meta['num_images']
            self.light_stacker.accumulator = accumulator
