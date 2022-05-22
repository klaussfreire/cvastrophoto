import numpy
import logging

import skimage.transform

from cvastrophoto.image import Image


logger = logging.getLogger(__name__)


def estimate_transform(transform_type, src, dst):
    if len(src) == 0 or len(dst) == 0:
        return None
    if transform_type == 'shift':
        shifts = dst - src
        if len(shifts) > 1:
            median_shift = numpy.median(shifts, axis=0)
            shift_spread = numpy.std(shifts, axis=0)
            bad_samples = numpy.max((shifts - median_shift) > (shift_spread * 2), axis=1)
            shifts = shifts[~bad_samples]
        translation = numpy.average(shifts, axis=0)
        return skimage.transform.EuclideanTransform(translation=translation)
    else:
        return skimage.transform.estimate_transform(transform_type, src, dst)


def find_transform(translations, transform_type, median_shift_limit, force_pass, fallback_transform_type=None):
    median_shift_mag = float('inf')
    transform = None
    if transform_type == 'shift':
        min_translations = 0
    elif transform_type == 'euclidean':
        min_translations = 2
    else:
        min_translations = 4
    while median_shift_mag > median_shift_limit and len(translations) >= min_translations:
        if len(translations) < min_translations and fallback_transform_type is not None:
            transform_type = fallback_transform_type

        # Estimate transform parameters out of valid measurements
        transform = estimate_transform(
            transform_type,
            translations[:, [3, 2]],
            translations[:, [1, 0]])

        # Weed out outliers
        transformed = transform(translations[:, [3, 2]])
        shift_mags = numpy.sum(numpy.square(translations[:, [1, 0]] - transformed), axis=1)
        median_shift_mag = numpy.median(shift_mags)
        logger.info("Median shift error: %.3f", median_shift_mag)
        if median_shift_mag > median_shift_limit:
            # Pick the worst and get it out of the way
            ntranslations = translations[shift_mags < shift_mags.max()]
            if len(ntranslations) >= max(1, min_translations - 1):
                logger.info("Removed %d bad grid points", len(translations) - len(ntranslations))
                translations = ntranslations
            else:
                logger.info("Can't remove any more grid points")
                return None

    if (median_shift_mag > median_shift_limit or len(translations) <= min_translations) and not force_pass:
        return None

    if transform is None:
        if len(translations) >= 2:
            # Use fallback transform type
            transform = estimate_transform(
                fallback_transform_type,
                translations[:, [3, 2]],
                translations[:, [1, 0]])

            # Weed out outliers
            transformed = transform(translations[:, [3, 2]])
            shift_mags = numpy.sum(numpy.square(translations[:, [1, 0]] - transformed), axis=1)
            median_shift_mag = numpy.median(shift_mags)
            logger.info("Median shift error: %.3f", median_shift_mag)
        elif len(translations):
            # Just a translation - fake translations and use estimate_transform just to respect the transform type
            ftranslations = numpy.array(list(translations)*3)
            ftranslations[1, [3, 1]] += 1
            ftranslations[2, [2, 0]] += 1
            transform = estimate_transform(
                fallback_transform_type,
                ftranslations[:, [3, 2]],
                ftranslations[:, [1, 0]])
        else:
            return None

    logger.info("Using %d reference grid points", len(translations))

    return transform


class TrackMaskMixIn(object):

    track_mask = None

    def __init__(self, *p, **kw):
        track_mask = kw.pop('track_mask', None)

        super(TrackMaskMixIn, self).__init__(*p, **kw)

        self._track_mask_bits = None
        self._track_mask_slice = None
        self.track_mask = Image.open(track_mask) if track_mask is not None else None

    def track_mask_bits(self, shape, scale=1, dt='f', threshold=0, slice=None, preshape=None):
        if self.track_mask is None:
            return None
        elif self._track_mask_bits is None or self._track_mask_slice != slice:
            bits = self.track_mask.luma_image(same_shape=False)
            if threshold is not None:
                bits = bits > threshold
            if preshape is None:
                preshape = shape
            if slice is not None and bits.shape == preshape:
                bits = bits[slice]
                slice = None
            elif slice is not None:
                shape = preshape
            bits = skimage.transform.resize(bits, shape)
            if slice is not None:
                bits = bits[slice]
            if threshold is None and scale is not None:
                bits = bits.astype(numpy.float32, copy=False) * (float(scale) / bits.max())
            bits = bits.astype(dt, copy=False)
            self._track_mask_bits = bits
            self._track_mask_slice = slice
            self.track_mask.close()
        return self._track_mask_bits

    def apply_gray_mask(self, img, threshold=None, scale=1, **kw):
        track_mask_bits = self.track_mask_bits(img.shape, threshold=threshold, scale=scale, **kw)
        if track_mask_bits is not None:
            if img.dtype.kind == track_mask_bits.dtype.kind:
                img *= track_mask_bits
            else:
                img = (img * track_mask_bits).astype(img.dtype)
        return img

    def apply_threshold_mask(self, img, threshold=0, **kw):
        track_mask_bits = self.track_mask_bits(img.shape, threshold=threshold, scale=None, dt='?', **kw)
        if track_mask_bits is not None:
            img[~track_mask_bits] = track_mask_bits
        return img
