# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
from six import iteritems

import cvastrophoto.wizards.stacking

from . import base, tag_classifier
from .darks import DarkLibrary
from .bias import BiasLibrary


logger = logging.getLogger(__name__)


class FlatLibrary(tag_classifier.TagClassificationMixIn, base.LibraryBase):

    min_subs = 4

    focallen_delta_steps = [2, 5, 10, 20, 50, None]

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
        (('LensModel', 'LensType', 'Lens'),),
        ('LensSerialNumber',),
        ('FNumber',),
        (('FocalLength', 'FOCALLEN'),),
    ]

    default_stacking_wizard_kwargs = dict(
        light_method=cvastrophoto.wizards.stacking.AdaptiveWeightedAverageStackingMethod,
        fbdd_noiserd=None,
        denoise=True,
        remove_bias=True,
    )

    default_base_path = '~/.cvastrophoto/flatslib'

    def __init__(self, base_path=None,
            stacking_wizard_class=cvastrophoto.wizards.stacking.StackingWizard,
            stacking_wizard_kwargs={},
            dark_library=None,
            bias_library=None,
            **kwargs):
        super(FlatLibrary, self).__init__(base_path, **kwargs)

        stacking_wizard_kwargs = stacking_wizard_kwargs.copy()
        if 'default_pool' in kwargs:
            stacking_wizard_kwargs.setdefault('pool', kwargs.get('default_pool'))
        for arg, val in iteritems(self.default_stacking_wizard_kwargs):
            stacking_wizard_kwargs.setdefault(arg, val)

        self.stacking_wizard_class = stacking_wizard_class
        self.stacking_wizard_kwargs = stacking_wizard_kwargs

        if dark_library is None:
            dark_library = DarkLibrary()
        if bias_library is None:
            bias_library = BiasLibrary()
        self.dark_library = dark_library
        self.bias_library = bias_library

    def vary(self, key, for_build=False):
        focallen = key[-1].split(' ')[0]
        try:
            focallen = float(focallen)
            if focallen > 1:
                focallen_delta_steps = self.focallen_delta_steps
            else:
                focallen_delta_steps = (None,)
        except (TypeError, ValueError):
            focallen = None
            focallen_delta_steps = (None,)

        keys = []
        for focallen_step in focallen_delta_steps:
            if focallen_step is None:
                qfocallen = 'all'
            else:
                qfocallen = int(focallen / focallen_step) * focallen_step
            keys.append(key[:-1] + (focallen_step, qfocallen,))

        return keys

    def build_master(self, key, frames):
        logging.info("Building master flat with %d subs for %r", len(frames), key)
        stacking_wizard = self.stacking_wizard_class(**self.stacking_wizard_kwargs)
        stacking_wizard.load_set(
            light_files=frames, dark_path=None,
            dark_library=self.dark_library,
            bias_library=self.bias_library)
        stacking_wizard.process()
        master = stacking_wizard.accumulator.average
        image_template = stacking_wizard._get_raw_instance()
        master = master.astype(image_template.rimg.raw_image.dtype)
        image_template.close()
        return master

    def default_filter(self, dirpath, dirname, filename):
        return (
            filename is None
            or (dirpath is not None and 'flats' in dirpath.lower())
            or (dirname is not None and 'flats' in dirname.lower())
            or (filename is not None and 'flats' in filename.lower())
        )
