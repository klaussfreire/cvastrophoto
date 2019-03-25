# -*- coding: utf-8 -*-
import numpy

def decode_srgb(raw_image, gamma=2.4):
    """
    Decodes srgb in-place in the normalized float raw_image
    """
    nonlinear_range = raw_image >= 0.04045
    raw_image[raw_image < 0.04045] *= 1.0 / 12.92
    raw_image[nonlinear_range] = numpy.power(
        (raw_image[nonlinear_range]+0.055) * (1.0/1.055),
        gamma,
    )
    return raw_image

def encode_srgb(raw_image, gamma=2.4):
    """
    Encodes srgb in-place in the normalized float raw_image
    """
    nonlinear_range = raw_image >= 0.0031308
    raw_image[raw_image < 0.0031308] *= 12.92
    raw_image[nonlinear_range] = 1.055 * numpy.power(
        raw_image[nonlinear_range],
        1.0 / gamma,
    ) - 0.055
    return raw_image
