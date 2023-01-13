from __future__ import absolute_import

from past.builtins import basestring
import re
import math
import numpy
import scipy.ndimage

from astropy.coordinates import SkyCoord, Angle
import  astropy.units as u

from cvastrophoto.util.coords import parse_coord_string
from ..base import BaseCatalog, CatalogObject

from pyongc import ongc

STOP_TERMS = {
    'nebula',
    'galaxy',
    'the',
}

class OpenNGCObject(CatalogObject):

    # Defaults for optional fields
    ra = None
    dec = None
    epoch = 'J2000'
    coords = None
    mags = None

    @classmethod
    def from_ongc(cls, dso, **kw):
        name = dso.getName()
        messier_name, alt_ngc_names, alt_ic_names, common_names, other_names = dso.getIdentifiers()
        alt_names = []

        if common_names:
            descriptive_name = '%s, %s' % (name, common_names[0])
        else:
            descriptive_name = None

        if messier_name:
            alt_names.append(messier_name)
        if alt_ngc_names:
            alt_names.extend(alt_ngc_names)
        if alt_ic_names:
            alt_names.extend(alt_ic_names)
        if common_names:
            alt_names.extend(common_names)
        if other_names:
            alt_names.extend(other_names)

        _ra = dso.getRA()
        _dec = dso.getDec()

        rv = cls(
            name,
            alt_names=alt_names,
            descriptive_name=descriptive_name,
            _ra=_ra, _dec=_dec, _dso=dso,
            obj_type=dso.getType(),
            dims=dso.getDimensions(),
            surface_mag=dso.getSurfaceBrightness(),
            hubble=dso.getHubble(),
            constellation=dso.getConstellation(),
            id=dso.getId(),
        )
        rv._fill(**kw)
        return rv

    def _fill(self, with_coords=True, with_mags=True):
        dso = self._dso

        if with_coords:
            try:
                coords = dso.getCoords()
                if isinstance(coords, numpy.ndarray):
                    (rah, ram, ras), (decdeg, decm, decs) = coords
                    decsign = ''
                elif isinstance(coords, tuple):
                    (rah, ram, ras), (decsign, decdeg, decm, decs) = coords
                else:
                    raise ValueError("unexpected coord type")
            except ValueError:
                ra = dec = coords = None
            else:
                ra = '%dh%dm%2.2fs' % (rah, ram, ras)
                dec = '%s%dd%dm%2.2fs' % (decsign, decdeg, decm, decs)
                coords = SkyCoord(
                    self._ra, self._dec,
                    obstime='J2000',
                    frame='icrs',
                    unit=('hour', 'deg'),
                )
        else:
            ra = dec = coords = None
        self.ra = ra
        self.dec = dec
        self.coords = coords

        if with_mags:
            self.magnitudes = dict(zip('BVJHK', dso.getMagnitudes()))

class OpenNGC0Catalog(BaseCatalog):

    CONCRETE = True
    DEFAULTABLE = True
    OFFLINE = True

    def get_object(self, name):
        try:
            dso = ongc.Dso(name)
        except ValueError:
            dso = None
        if dso is None:
            try:
                dso = ongc.searchAltId(name)
            except (ValueError, TypeError):
                dso = None
            if isinstance(dso, basestring):
                dso = None
        if dso is not None:
            return OpenNGCObject.from_ongc(dso)

    def search_name(self, terms,
            catalog=None, obj_type=None, constellation=None,
            with_coords=True, with_mags=True,
            **kw):
        seen = set()

        # Try by exact search
        if terms:
            obj = self.get_object(terms)
            if obj is not None:
                yield obj
                seen.add(obj.id)

        fill_opts = dict(with_coords=with_coords, with_mags=with_mags)
        needs_fill = any(fill_opts.values())

        # Detect a specific catalog to speed up fuzzy match
        filters = {}
        if terms:
            match = re.match(r'(M|NGC|IC).*', terms, re.I)
            if match:
                filters['catalog'] = match.group(1).upper().strip()
        if catalog:
            filters['catalog'] = catalog
        if obj_type:
            filters['type'] = obj_type
        if constellation:
            filters['constellation'] = constellation

        # Fuzzy match
        if terms:
            normalized_terms = re.sub(
                r'(M,NGC,IC) *([0-9*])',
                lambda m:'%s%s' % (m.group(1), m.group(2)),
                terms, re.I)
            normalized_terms = set(filter(None, normalized_terms.lower().split()))
            normalized_terms -= STOP_TERMS
            def term_match(n):
                item_terms = set(filter(None, n.lower().split()))
                item_terms -= STOP_TERMS
                return (
                    normalized_terms and item_terms
                    and not normalized_terms.isdisjoint(item_terms)
                )
        else:
            term_match = None

        for dso in ongc.listObjects(**filters):
            dso_id = dso.getId()
            if dso_id in seen:
                continue

            obj = OpenNGCObject.from_ongc(dso, with_coords=False, with_mags=False)
            if term_match is None or any(term_match(n) for n in obj.names):
                if needs_fill:
                    obj._fill(**fill_opts)
                if not kw or self._eval_filter(obj, **kw):
                    yield obj
                    seen.add(dso_id)

    def search_nearby(self, coords, radius, **kw):
        if isinstance(radius, Angle):
            radius = radius.degree
        if isinstance(coords, basestring):
            coords = parse_coord_string(coords)

        ra = int(coords.ra.hour)
        dec = int(coords.dec.degree)

        # Build a selection grid to quickly filter ra/dec pairs to speed up search
        # We sample with 2-degree margin in DEC and 16-degree margin in RA to avoid grid sampling issues
        # This is about 2.5x faster than a modified ongc.getSeparation, and 21x faster than using
        # SkyCoords on all objects one at a time.
        ragrid, decgrid = numpy.meshgrid(numpy.arange(24), numpy.arange(180) - 90)
        selgrid = numpy.zeros_like(ragrid, dtype='?')
        for raoff in numpy.linspace(0.0, 1.0, 15):
            scgrid = SkyCoord(ragrid + raoff, decgrid, unit=('hour', 'deg'), obstime='J2000', frame='icrs')
            selgrid |= scgrid.separation(coords).degree <= (radius + 2)
        selgrid[dec + 90, ra] = True

        for obj in self.search_name(None, with_coords=False, with_mags=False, **kw):
            objdec = obj._dec
            objra = obj._ra
            if not objdec or not objra:
                continue

            objra = int(objra.split(':', 1)[0]) % 24
            objdec = (int(objdec.split(':', 1)[0]) + 90) % 180

            if not selgrid[objdec, objra]:
                continue

            obj._fill()
            obj_coords = obj.coords
            if obj_coords and coords.separation(obj.coords).degree <= radius:
                yield obj


__ALL__ = [
    'OpenNGC0Catalog',
]
