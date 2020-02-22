# -*- coding: utf-8 -*-
from . import base, rgb, raw

from .base import BaseImage as Image, find_entropy_weights, entropy, ImageAccumulator
from .raw import Raw
from .rgb import RGB

__ALL__ = (
    'Image',
    'Raw',
    'RGB',
    'ImageAccumulator',
    'entropy',
    'find_entropy_weights',
)

try:
    from .avi import AVI
    __ALL__ += (
        'AVI',
    )
except ImportError:
    pass
