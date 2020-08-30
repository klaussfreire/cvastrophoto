# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
from six import iteritems

import cvastrophoto.wizards.stacking

from . import base, tag_classifier


logger = logging.getLogger(__name__)


class BiasLibrary(tag_classifier.TagClassificationMixIn, base.LibraryBase):

    min_subs = 20
    max_duration = 0.05

    classification_tags = [
        ('Make',),
        (('Model', 'INSTRUME'),),
        ('InternalSerialNumber', 'SerialNumber'),
        (('ImageSize', 'NAXIS'), ('ExifImageWidth', 'NAXIS1'), ('ExifImageHeight', 'NAXIS2')),
        (
            'SensorWidth', 'SensorHeight',
            ('SensorLeftBorder', 'XORFSUBF'), ('SensorTopBorder', 'YORGSUBF'),
            'SensorRightBorder', 'SensorBottomBorder',
            ('PhotometricInterpretation', 'COLORSPC',),

            # Optional, truncated if empty
            'BINNING', 'XBINNING', 'YBINNING',
            'BAYERPAT',
        ),
        (('ISO', 'GAIN', 'EGAIN'),),
        (('ExposureTime', 'EXPTIME'), ('BulbDuration', 'EXPOSURE')),
    ]

    default_stacking_wizard_kwargs = dict(
        light_method=cvastrophoto.wizards.stacking.MedianStackingMethod,
        fbdd_noiserd=None,
        denoise=False,
        remove_bias=False,
    )

    default_base_path = '~/.cvastrophoto/biaslib'

    def __init__(self, base_path=None,
            stacking_wizard_class=cvastrophoto.wizards.stacking.StackingWizard,
            stacking_wizard_kwargs={},
            **kwargs):
        super(BiasLibrary, self).__init__(base_path, **kwargs)

        stacking_wizard_kwargs = stacking_wizard_kwargs.copy()
        if 'default_pool' in kwargs:
            stacking_wizard_kwargs.setdefault('pool', kwargs.get('default_pool'))
        for arg, val in iteritems(self.default_stacking_wizard_kwargs):
            stacking_wizard_kwargs.setdefault(arg, val)

        self.stacking_wizard_class = stacking_wizard_class
        self.stacking_wizard_kwargs = stacking_wizard_kwargs

    def vary(self, key, for_build=False):
        sensor_info = key[4]
        if sensor_info.endswith(',NA,NA,NA,NA'):
            sensor_info = sensor_info[:-12]
            key = key[:4] + (sensor_info,) + key[5:]

        exptime, bulb = key[-1].split(',', 1)
        if bulb and bulb != 'NA':
            duration = float(bulb)
        elif exptime and ('/' in exptime or '_' in exptime):
            if '/' in exptime:
                num, denom = exptime.split('/', 1)
            else:
                num, denom = exptime.split('_', 1)
            num = float(num)
            denom = float(denom)
            duration = num / max(1, denom)
        elif exptime and exptime != 'NA':
            duration = float(exptime)
        else:
            duration = None

        if for_build and (duration is None or duration > self.max_duration):
            return []

        return [key[:-1]]

    def build_master(self, key, frames):
        logging.info("Building master bias with %d subs for %r", len(frames), key)
        stacking_wizard = self.stacking_wizard_class(**self.stacking_wizard_kwargs)
        stacking_wizard.load_set(light_files=frames, dark_path=None)
        stacking_wizard.process()
        return stacking_wizard.accumulator.average

    def default_filter(self, dirpath, dirname, filename):
        return (
            filename is None
            or (dirpath is not None and 'bias' in dirpath.lower())
            or (dirname is not None and 'bias' in dirname.lower())
            or (filename is not None and 'bias' in filename.lower())
        )
