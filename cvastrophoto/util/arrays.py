import numpy


def asnative(a):
    if not a.dtype.isnative:
        a = a.astype(a.dtype.newbyteorder('='))
    return a
