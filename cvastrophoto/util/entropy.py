# -*- coding: utf-8 -*-
import numpy
import skimage.filters.rank
import skimage.morphology

from . import srgb

try:
    # Some versions of rank.entropy can't take a float32 output
    skimage.filters.rank.entropy(
        numpy.array([[1,2],[3,4]], dtype=numpy.uint16),
        skimage.morphology.disk(size),
        out=numpy.empty((2,2), dtype=numpy.float32)
    )
    _direct_to_f32 = True
except Exception:
    _direct_to_f32 = False

def local_entropy_quantize(gray, gamma=2.4, copy=True, white=1.0):
    gray = gray.astype(numpy.float32, copy=copy)
    gray *= 65535.0 / (white * 65535)
    gray = numpy.clip(gray, 0, 65535, out=gray)
    gray = gray.astype(numpy.uint16)
    gray = srgb.encode_srgb(gray, gamma=gamma)
    gray = numpy.right_shift(gray, 8, out=gray)
    gray = gray.astype(numpy.uint8)
    return gray

def local_entropy(gray, gamma=2.4, selem=None, size=32, copy=True, white=1.0, quantized=False):
    if selem is None:
        selem = skimage.morphology.disk(size)
    if quantized:
        gray = gray.astype(numpy.uint8, copy=copy)
    else:
        gray = local_entropy_quantize(gray, gamma, copy, white)
    if _direct_to_f32:
        ent = numpy.empty(gray.shape, dtype=numpy.float32)
    else:
        ent = None
    ent = skimage.filters.rank.entropy(gray, selem, out=ent).astype(numpy.float32, copy=False)
    return ent
