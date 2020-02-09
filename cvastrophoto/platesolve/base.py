# -*- coding: utf-8 -*-
from astropy.io import fits

class PlateSolver(object):

    @staticmethod
    def ra_h_to_deg(ra):
        return ra * 180.0 / 12

    def set_hint(self, fits_path, hint, image_scale=None):
        """ Set coordinate hints on the given FITS file

        Update the FITS headers of the file referenced in ``fits_path``
        with guessed coordinates.

        :param tuple[float] hint: If given, hint coordinates where it is believed
            this snapshot was taken. Hints are given in ``(x, y, ra, dec)`` tuples,
            where ``x, y`` is the pixel that's assumed to be at ``ra, dec``,
            which ought to be in degrees.

        :param float image_scale: If give, hint image scale will be set
        """
        hdul = fits.open(fits_path, mode='update')
        try:
            hdu = hdul[0].header
            hdu['CTYPE1'] = 'RA---TAN'
            hdu['CTYPE2'] = 'DEC--TAN'
            hdu['CUNIT1'] = 'DEG'
            hdu['CUNIT2'] = 'DEG'
            hdu['CRPIX1'] = hint[0]
            hdu['CRPIX2'] = hint[1]
            hdu['CRVAL1'] = hint[2]
            hdu['CRVAL2'] = hint[3]
            if image_scale is not None:
                hdu['SCALE'] = image_scale
        finally:
            hdul.close()

    def get_coords(self, fits_path):
        """ Read coordinates from the given FITS file

        Returns the coordinates of the given FITS file in ``(x, y, ra, dec)``
        hint-like fashion
        """
        hdul = fits.open(fits_path, mode='readonly')
        try:
            hdu = hdul[0].header
            return (
                float(hdu['CRPIX1']),
                float(hdu['CRPIX2']),
                float(hdu['CRVAL1']),
                float(hdu['CRVAL2'])
            )
        finally:
            hdul.close()

    def solve(self, fits_path, hint=None, image_scale=None):
        """ Find the actual coordinates of the given snapshot

        Update the FITS headers of the file referenced in ``fits_path``
        with accurate coordinates.

        Returns True on success, False on failure.

        :param tuple[float] hint: If given, hint coordinates where it is believed
            this snapshot was taken. Hints are given in ``(x, y, ra, dec)`` tuples,
            where ``x, y`` is the pixel that's assumed to be at ``ra, dec``,
            which ought to be in degrees.
        """
        if hint:
            self.set_hint(fits_path, hint, image_scale)

        return self._solve_impl(fits_path)

    def annotate(self, fits_path, hint=None, image_scale=None):
        """ Annotate the given image with known objects

        Annotates the given image with recognizable objects and
        returns the annotated image file.

        :param tuple[float] hint: If given, hint coordinates where it is believed
            this snapshot was taken. Hints are given in ``(x, y, ra, dec)`` tuples,
            where ``x, y`` is the pixel that's assumed to be at ``ra, dec``,
            which ought to be in degrees.
        """
        if hint:
            self.set_hint(fits_path, hint, image_scale)

        return self._annotate_impl(fits_path)

    def _solve_impl(self, fits_path):
        raise NotImplementedError

    def _annotate_impl(self, fits_path):
        raise NotImplementedError
