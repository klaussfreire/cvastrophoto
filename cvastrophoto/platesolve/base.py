# -*- coding: utf-8 -*-
from astropy.io import fits
from astropy import wcs

class PlateSolver(object):

    last_solve = None

    @staticmethod
    def ra_h_to_deg(ra):
        return ra * 180.0 / 12

    @staticmethod
    def ra_deg_to_h(ra):
        return ra * 12.0 / 180

    def set_hint(self, fits_path, hint, image_scale=None, **kw):
        """ Set coordinate hints on the given FITS file

        Update the FITS headers of the file referenced in ``fits_path``
        with guessed coordinates.

        :param tuple[float] hint: If given, hint coordinates where it is believed
            this snapshot was taken. Hints are given in ``(x, y, ra, dec[, equinox])``
            tuples, where ``x, y`` is the pixel that's assumed to be at ``ra, dec``,
            which ought to be in degrees, and ``equinox`` is the equinox used for
            the ra/dec coordinates (by default J2000).

        :param float image_scale: If give, hint image scale will be set
        """
        try:
            hdul = fits.open(fits_path, mode='update')
        except Exception:
            return

        try:
            hdu = hdul[0].header

            if hint is not None and ('RA' not in hdu and 'DEC' not in hdu and 'EQUINOX' not in hdu):
                hdu['RA'] = hint[2]
                hdu['DEC'] = hint[3]
                hdu['EQUINOX'] = 2000 if len(hint) < 5 else hint[4]

            if image_scale is not None and ('SCALE' not in hdu):
                hdu['SCALE'] = image_scale
        finally:
            hdul.close()

    def get_coords(self, fits_path):
        """ Read coordinates from the given FITS file

        Returns the coordinates of the given FITS file in ``(x, y, ra, dec)``
        hint-like fashion
        """
        hdul = None
        try:
            last_solve = self.last_solve
            if last_solve is not None and last_solve[0] == fits_path:
                hdu = last_solve[1]
            else:
                hdul = fits.open(fits_path, mode='readonly')
                hdu = hdul[0].header

            hdu = hdul[0].header
            w = wcs.WCS(hdu)
            crpix = w.wcs.crpix
            crval = w.wcs_pix2world([crpix], 1, ra_dec_order=True)[0]
            return (
                float(crpix[0]),
                float(crpix[1]),
                float(crval[0]),
                float(crval[1]),
            )
        finally:
            if hdul is not None:
                hdul.close()

    def solve(self, fits_path, **kw):
        """ Find the actual coordinates of the given snapshot

        Update the FITS headers of the file referenced in ``fits_path``
        with accurate coordinates.

        Returns True on success, False on failure.

        :param tuple[float] hint: If given, hint coordinates where it is believed
            this snapshot was taken. Hints are given in ``(x, y, ra, dec[, equinox])``
            tuples, where ``x, y`` is the pixel that's assumed to be at ``ra, dec``,
            which ought to be in degrees, and ``equinox`` is the equinox used for
            the ra/dec coordinates (by default J2000).

        :param float image_scale: If given, hint of the image scale that will
            be used to aid plate solving. Ignored if fov is given.

        :param float fov: If given, image FOV that will be used to aid plate
            solving.
        """
        if kw.get('hint'):
            self.set_hint(fits_path, **kw)

        return self._solve_impl(fits_path, **kw)

    def annotate(self, fits_path, **kw):
        """ Annotate the given image with known objects

        Annotates the given image with recognizable objects and
        returns the annotated image file.

        :param tuple[float] hint: If given, hint coordinates where it is believed
            this snapshot was taken. Hints are given in ``(x, y, ra, dec[, equinox])``
            tuples, where ``x, y`` is the pixel that's assumed to be at ``ra, dec``,
            which ought to be in degrees, and ``equinox`` is the equinox used for
            the ra/dec coordinates (by default J2000).

        :param float image_scale: If given, hint of the image scale that will
            be used to aid plate solving. Ignored if fov is given.

        :param float fov: If given, image FOV that will be used to aid plate
            solving.
        """
        if kw.get('hint'):
            self.set_hint(fits_path, **kw)

        return self._annotate_impl(fits_path, **kw)

    def _solve_impl(self, fits_path, **kw):
        raise NotImplementedError

    def _annotate_impl(self, fits_path, **kw):
        raise NotImplementedError
