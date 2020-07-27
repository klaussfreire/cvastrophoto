import numpy
import logging

import skimage.transform


logger = logging.getLogger(__name__)


def find_transform(translations, transform_type, median_shift_limit, force_pass, fallback_transform_type=None):
    median_shift_mag = float('inf')
    transform = None
    while median_shift_mag > median_shift_limit and len(translations) > 3:
        if len(translations) < 4 and fallback_transform_type is not None:
            transform_type = fallback_transform_type

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

    if transform is None:
        if len(translations) >= 2:
            # Use fallback transform type
            transform = skimage.transform.estimate_transform(
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
            transform = skimage.transform.estimate_transform(
                fallback_transform_type,
                ftranslations[:, [3, 2]],
                ftranslations[:, [1, 0]])
        else:
            return None

    logger.info("Using %d reference grid points", len(translations))

    return transform
