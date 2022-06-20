# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import scipy.ndimage
import os
import subprocess
import imageio
import tempfile
import logging

from ..denoise import median
from ..compound import CompoundRop
from ..base import PerChannelRop
from ..bias import localgradient
from cvastrophoto.util import demosaic
from cvastrophoto.image import Image


logger = logging.getLogger(__name__)


class BackgroundRemovalRop(localgradient.LocalGradientBiasRop):

    minfilter_size = 32
    gauss_size = 32
    pregauss_size = 0
    despeckle_size = 0
    chroma_filter_size = None
    luma_minfilter_size = None
    luma_gauss_size = None
    close_factor = 1.0
    despeckle = False
    aggressive = False
    protect_white = False
    gain = 1.0
    offset = 0


class StarnetStarRemovalRop(localgradient.LocalGradientBiasRop):

    starnet_path = os.environ.get('STARNET_PATH', '')
    starnet_bin = os.environ.get('STARNET_BIN', './starnet++')

    def correct(self, data, detected=None, img=None):
        starnet_dir = self.starnet_path
        if not starnet_dir:
            starnet_dir = '.'

        cwd = os.getcwd()
        infile = tempfile.mktemp(prefix='.cvap-starnet-', suffix='-in.tiff')
        outfile = tempfile.mktemp(prefix='.cvap-starnet-', suffix='-out.tiff')

        ppdata = data.copy()
        if ppdata.dtype.char in 'lLiI':
            dmax = ppdata.max()
            if dmax <= 65535:
                ppdata = ppdata.astype('H')
            else:
                ppdata = ppdata.astype('f')
        if ppdata.dtype.kind == 'f':
            dmax = ppdata.max()
            scale = dmax
            ppdata *= 65535.0 / scale
            ppdata = numpy.clip(ppdata, 0, 65535, out=ppdata)
            ppdata = ppdata.astype('H')
        else:
            scale = None
        try:
            logger.info("Invoking starnet from %r", starnet_dir)
            if len(ppdata.shape) == 3:
                self.raw.set_raw_image(demosaic.remosaic(ppdata, self._raw_pattern))
            else:
                self.raw.set_raw_image(ppdata)
            self.raw.save(infile, meta={})
            os.chdir(starnet_dir)
            if subprocess.check_call([self.starnet_bin, infile, outfile]):
                raise RuntimeError("Error invoking starnet")
            with Image.open(outfile, autoscale=False) as starless_img:
                if len(ppdata.shape) == 3:
                    ppdata = starless_img.postprocessed
                else:
                    ppdata = starless_img.rimg.raw_image
            if scale is not None:
                ppdata = ppdata.astype('f')
                ppdata *= scale / 65535.0
            data[:] = ppdata
        finally:
            if os.path.exists(infile):
                os.unlink(infile)
            if os.path.exists(outfile):
                os.unlink(outfile)
            os.chdir(cwd)

        return data


class StarnetBackgroundRemovalRop(StarnetStarRemovalRop):

    def correct(self, data, detected=None, **kw):
        bg = super(StarnetBackgroundRemovalRop, self).correct(data.copy(), **kw)
        data -= numpy.minimum(bg, data)
        return data


class WhiteTophatFilterRop(PerChannelRop):

    size = 1

    def process_channel(self, data, detected=None, img=None):
        return scipy.ndimage.white_tophat(data, self.size)


class ExtractStarsRop(CompoundRop):

    quick = False

    median_size = 3
    median_sigma = 1.0
    star_size = 32
    close_factor = 1.0
    offset = 0

    method = 'localgradient'

    starnet_path = StarnetStarRemovalRop.starnet_path
    starnet_bin = StarnetStarRemovalRop.starnet_bin

    def __init__(self, raw, **kw):
        self.median_size = median_size = int(kw.pop('median_size', self.median_size))
        self.median_sigma = median_sigma = float(kw.pop('median_sigma', self.median_sigma))
        self.star_size = star_size = int(kw.pop('star_size', self.star_size))
        self.offset = offset = int(kw.pop('offset', self.offset))
        self.method = method = kw.pop('method', self.method)

        if self.quick:
            extract_rop = WhiteTophatFilterRop(raw, size=star_size, **kw)
        else:
            if method == 'localgradient':
                skw = kw.copy()
                skw.setdefault('minfilter_size', star_size)
                skw.setdefault('gauss_size', star_size)
                skw.setdefault('star_size', star_size)
                extract_rop = BackgroundRemovalRop(raw, offset=offset, **skw)
            else:
                extract_rop = StarnetBackgroundRemovalRop(raw, **kw)

        super(ExtractStarsRop, self).__init__(
            raw,
            extract_rop,
            median.MaskedMedianFilterRop(raw, size=median_size, sigma=median_sigma, **kw)
        )


class ExtractPureStarsRop(ExtractStarsRop):

    quick = False
    despeckle_size = 3
    pregauss_size = 3
    aggressive = True

    def __init__(self, raw, **kw):
        kw.setdefault('despeckle', True)
        kw.setdefault('despeckle_size', 3)
        kw.setdefault('pregauss_size', 3)
        kw.setdefault('aggressive', True)
        super(ExtractPureStarsRop, self).__init__(raw, **kw)


class RemoveStarsRop(ExtractPureStarsRop):

    def correct(self, data, *p, **kw):
        stars = super(RemoveStarsRop, self).correct(data.copy(), *p, **kw)
        data -= numpy.clip(stars, None, data)
        return data


class ExtractPureBackgroundRop(ExtractStarsRop):

    def correct(self, data, *p, **kw):
        stars = super(ExtractPureBackgroundRop, self).correct(data.copy(), *p, **kw)
        data -= numpy.clip(stars, None, data)
        return data
