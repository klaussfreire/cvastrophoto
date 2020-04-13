# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging

import cvastrophoto.wizards.stacking

from . import base, tag_classifier


logger = logging.getLogger(__name__)


class DarkLibrary(tag_classifier.TagClassificationMixIn, base.LibraryBase):

    temp_steps = [
        1.0,
        2.0,
        5.0,
        10.0,
        None,
    ]

    min_subs = 10

    default_version = 2
    fallback_versions = [1]
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
        (('CameraTemperature', 'TEMP', 'CCD-TEMP'),)
    ]

    default_stacking_wizard_kwargs = dict(
        light_method=cvastrophoto.wizards.stacking.MedianStackingMethod,
        fbdd_noiserd=None,
    )

    default_base_path = '~/.cvastrophoto/darklib'

    def __init__(self, base_path=None,
            stacking_wizard_class=cvastrophoto.wizards.stacking.StackingWizard,
            stacking_wizard_kwargs={},
            **kwargs):
        super(DarkLibrary, self).__init__(base_path, **kwargs)

        stacking_wizard_kwargs = stacking_wizard_kwargs.copy()
        if 'default_pool' in kwargs:
            stacking_wizard_kwargs.setdefault('pool', kwargs.get('default_pool'))
        for arg, val in self.default_stacking_wizard_kwargs.iteritems():
            stacking_wizard_kwargs.setdefault(arg, val)

        self.stacking_wizard_class = stacking_wizard_class
        self.stacking_wizard_kwargs = stacking_wizard_kwargs

    def vary(self, key):
        sensor_info = key[4]
        if sensor_info.endswith(',NA,NA,NA,NA'):
            sensor_info = sensor_info[:-12]
            key = key[:4] + (sensor_info,) + key[5:]

        temp = key[-1].split()[0].lower()
        if temp.endswith('c') or temp.endswith('f'):
            temp = temp[:-1]

        if temp == 'na':
            # No temperature to vary
            return [key[:-1] + ('NA', key[-1])]

        temp = float(temp)

        keys = []
        for step in self.temp_steps:
            if step is None:
                qtemp = 'all'
                step = 'inf'
            else:
                qtemp = int(temp / step) * step
            keys.append(key[:-1] + (step, qtemp,))

        return keys

    def build_master(self, key, frames):
        logging.info("Building master dark with %d subs for %r", len(frames), key)
        stacking_wizard = self.stacking_wizard_class(**self.stacking_wizard_kwargs)
        stacking_wizard.load_set(light_files=frames, dark_path=None)
        stacking_wizard.process()
        return stacking_wizard.accumulator.average

    def default_filter(self, dirpath, dirname, filename):
        return (
            filename is None
            or (dirpath is not None and 'dark' in dirpath.lower())
            or (dirname is not None and 'dark' in dirname.lower())
            or (filename is not None and 'dark' in filename.lower())
        )
