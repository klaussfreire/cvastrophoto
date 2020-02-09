# -*- coding: utf-8 -*-
from .base import PlateSolver

import tempfile
import os.path
import subprocess

from cvastrophoto.image import rgb


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
        tmpprefix = os.path.join(os.path.dirname(fits_path), basename)
        cmd = self._basecmd(fits_path, None, hint, fov)
        cmd.append('-update')
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            return False
        finally:
            self._cleanup(tmpprefix)
        return True

    def _cleanup(self, tmpprefix):
        # Remove leftover files we don't need
        for suffix in ('.wcs', '.ini'):
            path = tmpprefix + suffix
            if os.path.isfile(path):
                os.unlink(path)

    def _annotate_impl(self, fits_path, hint=None, fov=None, **kw):
        suffix = '_annotated.jpg'
        ofile = tempfile.NamedTemporaryFile(
            dir=os.path.dirname(fits_path),
            suffix=suffix)
        tmpprefix = ofile.name[:-len(suffix)]
        cmd = self._basecmd(fits_path, tmpprefix, hint, fov)
        cmd.append('-annotate')

        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            return None
        finally:
            self._cleanup(tmpprefix)

        rv = rgb.RGB(ofile.name)
        rv.fileobj = ofile
        return rv