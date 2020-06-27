# -*- coding: utf-8 -*-
from .base import PlateSolver

import tempfile
import os.path
import subprocess

from cvastrophoto.image import rgb

try:
    from cvastrophoto.image import raw
except ImportError:
    raw = None


class ASTAPSolver(PlateSolver):

    ASTAP_PATH = None

    ASTAP_PATHS = [
        '/usr/bin/astap',
        '/opt/astap/astap',
        '/usr/local/bin/astap',
    ]

    search_radius = 10
    downsample_factor = 0
    tolerance = None
    supports_raw = True

    TOLERANCES = {
        'high': 0.007,
        'normal': 0.005,
        'low': 0.003,
    }

    max_stars = 500

    def get_astap(self):
        if self.ASTAP_PATH is not None:
            return self.ASTAP_PATH

        for path in self.ASTAP_PATHS:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                self.ASTAP_PATH = path
                break

        return self.ASTAP_PATH

    def _basecmd(self, fits_path, tmpprefix, hint=None, fov=None):
        cmd = [
            self.get_astap(),
            '-f',
            fits_path,
            '-r', str(self.search_radius),
            '-s', str(self.max_stars),
            '-z', str(self.downsample_factor),
        ]
        if tmpprefix is not None:
            cmd.extend([
                '-o',
                tmpprefix,
            ])
        if self.tolerance is not None:
            cmd.extend([
                '-t',
                str(self.TOLERANCES.get(self.tolerance, self.tolerance)),
            ])
        if hint is not None:
            cmd.extend([
                '-ra', str(self.ra_deg_to_h(hint[2])),
                '-spd', str(hint[3] + 90),
            ])
        if fov is not None:
            cmd.extend([
                '-fov',
                str(fov)
            ])
        return cmd

    def _solve_impl(self, fits_path, hint=None, fov=None, **kw):
        basename = os.path.basename(fits_path)
        basename, ext = os.path.splitext(fits_path)
        dirname = os.path.dirname(fits_path)
        tmpprefix = os.path.join(dirname, basename)
        xtemp = []

        if not self.supports_raw and raw is not None and raw.Raw.supports(fits_path):
            # Convert to a temp jpg
            img = raw.Raw(fits_path)
            if img.postprocessing_params is not None:
                img.postprocessing_params.half_size = True
            tmpname = '%s_astap_tmp_' % (basename,)
            fits_path = tempfile.mktemp(
                dir=dirname,
                prefix=tmpname,
                suffix='.jpg')
            tmpprefix = fits_path[:-4]
            xtemp.append(fits_path)
            img.save(fits_path)

        cmd = self._basecmd(fits_path, None, hint, fov)
        cmd.append('-update')
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            return False
        finally:
            self._cleanup(tmpprefix, xtemp)
        return True

    def _cleanup(self, tmpprefix, xtemp=()):
        # Remove leftover files we don't need
        for suffix in ('.wcs', '.ini'):
            path = tmpprefix + suffix
            if os.path.isfile(path):
                os.unlink(path)
        for path in xtemp:
            if os.path.isfile(path):
                os.unlink(path)

    def _annotate_impl(self, fits_path, hint=None, fov=None, **kw):
        suffix = '_annotated.jpg'
        dirname = os.path.dirname(fits_path)
        basename, ext = os.path.splitext(fits_path)
        ofile = tempfile.NamedTemporaryFile(
            dir=dirname,
            suffix=suffix)
        tmpprefix = ofile.name[:-len(suffix)]
        xtemp = []

        if not self.supports_raw and raw is not None and raw.Raw.supports(fits_path):
            # Convert to a temp jpg
            img = raw.Raw(fits_path)
            if img.postprocessing_params is not None:
                img.postprocessing_params.half_size = True
            tmpname = '%s_astap_tmp_' % (basename,)
            fits_path = tempfile.mktemp(
                dir=dirname,
                prefix=tmpname,
                suffix='.jpg')
            xtemp.append(fits_path)
            img.save(fits_path)

        cmd = self._basecmd(fits_path, tmpprefix, hint, fov)
        cmd.append('-annotate')

        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            return None
        finally:
            self._cleanup(tmpprefix, xtemp)

        rv = rgb.RGB(ofile.name)
        rv.fileobj = ofile
        return rv
