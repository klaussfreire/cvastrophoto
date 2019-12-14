# -*- coding: utf-8 -*-
import numpy
import numpy.linalg

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

def color_matrix(in_, matrix, out_):
    out_[:,:,0] = in_[:,:,0] * matrix[0, 0]
    out_[:,:,0] += in_[:,:,1] * matrix[0, 1]
    out_[:,:,0] += in_[:,:,2] * matrix[0, 2]

    out_[:,:,1] = in_[:,:,0] * matrix[1, 0]
    out_[:,:,1] += in_[:,:,1] * matrix[1, 1]
    out_[:,:,1] += in_[:,:,2] * matrix[1, 2]

    out_[:,:,2] = in_[:,:,0] * matrix[2, 0]
    out_[:,:,2] += in_[:,:,1] * matrix[2, 1]
    out_[:,:,2] += in_[:,:,2] * matrix[2, 2]

    return out_

MATRIX_XYZ2RGB = numpy.array([
    [0.41847, -0.15866, -0.082835],
    [-0.091169, 0.25243, 0.015708],
    [0.0009209, -0.0025498, 0.17860],
], numpy.float32)

def xyz2rgb(in_, out_):
    return color_matrix(in_, MATRIX_XYZ2RGB, out_)

def camera2xyz(in_, rimg, out_):
    return color_matrix(in_, numpy.linalg.inv(rimg.rgb_xyz_matrix[:3,:3]), out_)

def camera2rgb(in_, rimg_or_matrix, out_):
    if hasattr(rimg_or_matrix, 'rgb_xyz_matrix'):
        rgb_xyz_matrix = rimg_or_matrix.rgb_xyz_matrix
    else:
        rgb_xyz_matrix = rimg_or_matrix
    return color_matrix(
        in_,
        numpy.matmul(
            MATRIX_XYZ2RGB,
            numpy.linalg.inv(rgb_xyz_matrix[:3,:3]),
        ),
        out_)
