from __future__ import absolute_import, division

import math
import numpy

from cvastrophoto.util import vectorize

from skimage import color
from skimage.color import colorconv, rgb2hsv as sk_rgb2hsv, hsv2rgb as sk_hsv2rgb

@vectorize.auto_guvectorize(
    [
        'float32[:],float32[:]',
        'float64[:],float64[:]',
    ],
    '(n)->(n)',
)
def _f_rgb2hsv(rgb, hsv):
    mx = max(rgb[0], rgb[1], rgb[2])
    mn = min(rgb[0], rgb[1], rgb[2])
    delta = mx - mn

    out_v = mx

    if mx != 0 and delta != 0:
        out_s = delta / mx

        if mx == rgb[0]:
            # red is max
            out_h = (rgb[1] - rgb[2]) / delta
        elif mx == rgb[1]:
            # green is max
            out_h = 2 + (rgb[2] - rgb[0]) / delta
        else:
            # blue is max
            out_h = 4 + (rgb[0] - rgb[1]) / delta
        out_h = (out_h / 6) % 1
    else:
        out_s = 0
        out_h = 0

    hsv[0] = out_h
    hsv[1] = out_s
    hsv[2] = out_v

@vectorize.auto_guvectorize(
    [
        'uint32[:],uint64,uint64,uint32[:]',
        'uint16[:],uint32,uint32,uint16[:]',
        'uint8[:],uint32,uint32,uint8[:]',
    ],
    '(n),(),()->(n)',
)
def _i_rgb2hsv(rgb, maxval, maxval6, hsv):
    mx = max(rgb[0], rgb[1], rgb[2])
    mn = min(rgb[0], rgb[1], rgb[2])
    delta = mx - mn

    out_v = mx

    if mx != 0 and delta != 0:
        out_s = delta * maxval // mx

        if mx == rgb[0]:
            # red is max
            out_h = (numpy.int64(rgb[1]) - numpy.int64(rgb[2])) * maxval6 // delta
        elif mx == rgb[1]:
            # green is max
            out_h = 2 * maxval6 + (numpy.int64(rgb[2]) - numpy.int64(rgb[0])) * maxval6 // delta
        else:
            # blue is max
            out_h = 4 * maxval6 + (numpy.int64(rgb[0]) - numpy.int64(rgb[1])) * maxval6 // delta
        out_h &= maxval
    else:
        out_s = 0
        out_h = 0

    hsv[0] = out_h
    hsv[1] = out_s
    hsv[2] = out_v

@vectorize.auto_guvectorize(
    [
        'float32[:],float32[:]',
        'float64[:],float64[:]',
    ],
    '(n)->(n)',
)
def _f_hsv2rgb(hsv, rgb):
    h6 = hsv[0] * 6
    s = hsv[1]
    v = hsv[2]
    hi = int(math.floor(h6))
    f = h6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    hi6 = hi % 6
    if hi6 == 0:
        rgb[0] = v
        rgb[1] = t
        rgb[2] = p
    elif hi6 == 1:
        rgb[0] = q
        rgb[1] = v
        rgb[2] = p
    elif hi6 == 2:
        rgb[0] = p
        rgb[1] = v
        rgb[2] = t
    elif hi6 == 3:
        rgb[0] = p
        rgb[1] = q
        rgb[2] = v
    elif hi6 == 4:
        rgb[0] = t
        rgb[1] = p
        rgb[2] = v
    elif hi6 == 5:
        rgb[0] = v
        rgb[1] = p
        rgb[2] = q

@vectorize.auto_guvectorize(
    [
        'uint32[:],uint64,uint32[:]',
        'uint16[:],uint32,uint16[:]',
        'uint8[:],uint32,uint8[:]',
    ],
    '(n),()->(n)',
)
def _i_hsv2rgb(hsv, maxval, rgb):
    h6 = hsv[0] * 6.0 / maxval
    s = hsv[1] / maxval
    v = hsv[2] / maxval
    hi = int(math.floor(h6))
    f = h6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    p *= maxval
    q *= maxval
    t *= maxval
    v *= maxval

    hi6 = hi % 6
    if hi6 == 0:
        rgb[0] = v
        rgb[1] = t
        rgb[2] = p
    elif hi6 == 1:
        rgb[0] = q
        rgb[1] = v
        rgb[2] = p
    elif hi6 == 2:
        rgb[0] = p
        rgb[1] = v
        rgb[2] = t
    elif hi6 == 3:
        rgb[0] = p
        rgb[1] = q
        rgb[2] = v
    elif hi6 == 4:
        rgb[0] = t
        rgb[1] = p
        rgb[2] = v
    elif hi6 == 5:
        rgb[0] = v
        rgb[1] = p
        rgb[2] = q

def rgb2hsv(rgb, channel_axis=-1):
    if rgb.shape[channel_axis] != 3 or rgb.dtype.char not in 'dfBHI':
        return sk_rgb2hsv(rgb, channel_axis=channel_axis)

    if rgb.dtype.kind == 'f':
        return _f_rgb2hsv(rgb, axis=channel_axis)
    elif rgb.dtype.char == 'B':
        mxval = 0xFF
    elif rgb.dtype.char == 'H':
        mxval = 0xFFFF
    elif rgb.dtype.char == 'I':
        mxval = 0xFFFFFFFF
    else:
        return sk_rgb2hsv(rgb, channel_axis=channel_axis)

    return _i_rgb2hsv(rgb, mxval, mxval//6, axis=channel_axis)


def hsv2rgb(rgb, channel_axis=-1):
    if rgb.shape[channel_axis] != 3 or rgb.dtype.char not in 'dfBHI':
        return sk_hsv2rgb(rgb, channel_axis=channel_axis)

    if rgb.dtype.kind == 'f':
        return _f_hsv2rgb(rgb, axis=channel_axis)
    elif rgb.dtype.char == 'B':
        mxval = 0xFF
    elif rgb.dtype.char == 'H':
        mxval = 0xFFFF
    elif rgb.dtype.char == 'I':
        mxval = 0xFFFFFFFF
    else:
        return sk_hsv2rgb(rgb, channel_axis=channel_axis)

    return _i_hsv2rgb(rgb, mxval, axis=channel_axis)


def monkeypatch():
    color.rgb2hsv = rgb2hsv
    color.hsv2rgb = hsv2rgb
    colorconv.rgb2hsv = rgb2hsv
    colorconv.hsv2rgb = hsv2rgb
