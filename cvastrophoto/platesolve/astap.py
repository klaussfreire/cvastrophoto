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

    search_radius = 30
    downsample_factor = 0

    max_stars = 500

    def get_astap(self):
        if self.ASTAP_PATH is not None:
            return self.ASTAP_PATH

        for path in self.ASTAP_PATHS:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                self.ASTAP_PATH = path
                break

        return self.ASTAP_PATH

    def _solve_impl(self, fits_path):
        tmpprefix = tempfile.mktemp(dir=os.path.dirname(fits_path))
        cmd = [
            self.get_astap(),
            '-update',
            '-f',
            fits_path,
            '-o',
            tmpprefix,
            '-r', str(self.search_radius),
            '-s', str(self.max_stars),
            '-z', str(self.downsample_factor),
        ]
        try:
            subprocess.check_call(cmd)
        finally:
            self._cleanup(tmpprefix)

    def _cleanup(self, tmpprefix):
        # Remove leftover files we don't need
        for suffix in ('.wcs', '.ini'):
            path = tmpprefix + suffix
            if os.path.isfile(path):
                os.unlink(path)

    def _annotate_impl(self, fits_path):
        suffix = '_annotated.jpg'
        ofile = tempfile.NamedTemporaryFile(
            dir=os.path.dirname(fits_path),
            suffix=suffix)
        tmpprefix = ofile.name[:-len(suffix)]
        cmd = [
            self.get_astap(),
            '-annotate',
            '-f',
            fits_path,
            '-o',
            tmpprefix,
            '-r', str(self.search_radius),
            '-s', str(self.max_stars),
            '-z', str(self.downsample_factor),
        ]

        try:
            subprocess.check_call(cmd)
        finally:
            self._cleanup(tmpprefix)

        rv = rgb.RGB(ofile.name)
        rv.fileobj = ofile
        return rv
