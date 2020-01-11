# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging

import cvastrophoto.wizards.stacking

from . import base, exif


logger = logging.getLogger(__name__)


class BiasLibrary(exif.ExifClassificationMixIn, base.LibraryBase):

    min_subs = 5
    max_duration = 0.05

    classification_tags = [
        ('Make',),
        ('Model',),
        ('InternalSerialNumber', 'SerialNumber'),
        ('ImageSize', 'ExifImageWidth', 'ExifImageHeight'),
        (
            'SensorWidth', 'SensorHeight',
            'SensorLeftBorder', 'SensorTopBorder', 'SensorRightBorder', 'SensorBottomBorder',
            'PhotometricInterpretation',
        ),
        ('ISO',),
        ('ExposureTime', 'BulbDuration'),
    ]

    default_stacking_wizard_kwargs = dict(
        light_method=cvastrophoto.wizards.stacking.MedianStackingMethod,
        fbdd_noiserd=None,
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
        for arg, val in self.default_stacking_wizard_kwargs.iteritems():
            stacking_wizard_kwargs.setdefault(arg, val)

        self.stacking_wizard_class = stacking_wizard_class
        self.stacking_wizard_kwargs = stacking_wizard_kwargs

    def vary(self, key):
        exptime, bulb = key[-1].split(',', 1)
        if bulb:
            duration = float(bulb)
        elif exptime and '/' in exptime:
            num, denom = exptime.split('/', 1)
            num = float(num)
            denom = float(denom)
            duration = num / max(1, denom)
        elif exptime:
            duration = float(exptime)
        else:
            duration = None

        if duration is None or duration > self.max_duration:
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