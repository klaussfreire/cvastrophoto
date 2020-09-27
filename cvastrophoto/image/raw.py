# -*- coding: utf-8 -*-
import operator
import rawpy
try:
    from rawpy import enhance
except ImportError:
    enhance = None  # lint:ok

import logging

from .base import BaseImage, Sizes

logger = logging.getLogger(__name__)

class Raw(BaseImage):

    priority = 1
    concrete = True

    def __init__(self, path,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            **kw):
        super(Raw, self).__init__(path, **kw)
        self._kw['demosaic_algorithm'] = demosaic_algorithm
        self.postprocessing_params = rawpy.Params(
            output_bps=16,
            #output_color=rawpy.ColorSpace.raw,
            no_auto_bright=True,
            demosaic_algorithm=demosaic_algorithm,
            gamma=(1,1),
            user_flip=0,
        )

    def _open_impl(self, path):
        return rawpy.imread(path)

    @property
    def sizes(self):
        return Sizes(*operator.attrgetter(*Sizes._fields)(self.rimg.sizes))

    @classmethod
    def find_bad_pixels(cls, images, **kw):
        if enhance is None:
            logger.warning("Could not import rawpy.enhance, install dependencies to enable bad pixel detection")
            return None

        logger.info("Analyzing %d images to detect bad pixels...", len(images))
        coords = rawpy.enhance.find_bad_pixels([img.name for img in images], **kw)
        logger.info("Found %d bad pixels", len(coords))
        return coords

    def repair_bad_pixels(self, coords, **kw):
        if coords is None or not len(coords):
            return

        if enhance is None:
            logger.warning("Could not import rawpy.enhance, install dependencies to enable bad pixel correction")
            return

        rawpy.enhance.repair_bad_pixels(self.rimg, coords, **kw)
        logger.info("Done repairing %d bad pixels...", len(coords))

    @classmethod
    def supports(cls, path):
        return path.rsplit('.', 1)[-1].lower() in ('nef', 'cr2')
