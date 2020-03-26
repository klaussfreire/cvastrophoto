import numpy
import logging

import skimage.transform


logger = logging.getLogger(__name__)


def find_transform(translations, transform_type, median_shift_limit, force_pass):
    median_shift_mag = float('inf')
    while median_shift_mag > median_shift_limit and len(translations) > 3:
        # Estimate transform parameters out of valid measurements
        transform = skimage.transform.estimate_transform(
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
            if len(ntranslations) >= 3:
                logger.info("Removed %d bad grid points", len(translations) - len(ntranslations))
                translations = ntranslations
            else:
                logger.info("Can't remove any more grid points")
                return None

    if (median_shift_mag > median_shift_limit or len(translations) <= 4) and not force_pass:
        return None

    logger.info("Using %d reference grid points", len(translations))

    return transform
