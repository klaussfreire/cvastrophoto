# -*- coding: utf-8 -*-
from astropy.coordinates import SkyCoord, FK5, ICRS
import astropy.units as u

def parse_coord_string(coord):
    if coord is None or isinstance(coord, SkyCoord):
        return coord

    components = coord.split(',', 2)

    ra, dec = components[:2]
    if len(components) > 2:
        epoch = components[2]

        kw = dict(obstime=epoch, frame=FK5(equinox=epoch))
    else:
        kw = {}

    try:
        gc = SkyCoord('%s %s' % (ra, dec), **kw)
    except Exception:
        gc = SkyCoord(ra*u.hour, dec*u.degree, **kw)
    gc = gc.transform_to(ICRS())

    return gc
