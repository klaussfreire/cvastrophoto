# -*- coding: utf-8 -*-
import numpy
import skimage.filters.rank
import skimage.morphology

from . import srgb

def local_entropy(gray, gamma=2.4, selem=None, size=32, copy=True, white=1.0):
    if selem is None:
        selem = skimage.morphology.disk(size)
    gray = gray.astype(numpy.float32, copy=copy)
    gray *= 65535.0 / (white * 65535)
    gray = numpy.clip(gray, 0, 65535, out=gray)
    gray = gray.astype(numpy.uint16)
    gray = srgb.encode_srgb(gray, gamma=gamma)
    gray = numpy.right_shift(gray, 8, out=gray)
    gray = gray.astype(numpy.uint8)
    ent = skimage.filters.rank.entropy(gray, selem).astype(numpy.float32)
    return ent
