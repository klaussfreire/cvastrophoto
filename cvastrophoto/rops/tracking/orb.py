# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import numpy
import logging
import operator
import os.path
import PIL.Image

import cv2
import skimage.transform
import skimage.morphology
import skimage.color
import scipy.ndimage

from cvastrophoto.image import rgb
from cvastrophoto.util import srgb

from .base import BaseTrackingRop
from .util import find_transform
from . import extraction
from cvastrophoto.util import gaussian

logger = logging.getLogger(__name__)

class OrbFeatureTrackingRop(BaseTrackingRop):

    nfeatures = 5000
    keep_matches = 0.9
    WTA_K = 3
    distance_method = cv2.NORM_HAMMING2
    fast_threshold = 5
    mask_feature_size = 64
    mask_prefilter = 2
    mask_sigma = 2.0
    mask_opening = 1
    mask_closing = 8
    mask_dilation = 20
    gamma = 3.0
    downsample = 1
    show = False

    add_bias = False
    min_sim = None
    sim_prefilter_size = 64
    median_shift_limit = 2.0
    force_pass = False
    track_roi = (0, 0, 0, 0)  # (t-margin, l-margin, b-margin, r-margin), normalized
    tracking_cache = None
    deglow = None
    reference = None
    save_tracks = False

    _POPKW = (
        'add_bias',
        'min_sim',
        'sim_prefilter_size',
        'median_shift_limit',
        'force_pass',
        'track_roi',
        'deglow',
    )

    def __init__(self, raw, pool=None,
            transform_type='euclidean',
            order=3,
            mode='reflect',
            median_shift_limit=None,
            track_roi=None,
            track_distance=None,
            **kw):
        super(OrbFeatureTrackingRop, self).__init__(raw, **kw)
        if pool is None:
            pool = raw.default_pool
        self.pool = pool
        self.transform_type = transform_type
        self.order = order
        self.mode = mode
        self.lxscale = self.lyscale = None

        for k in self._POPKW:
            kw.pop(k, None)

        pp_rop = kw.get('luma_preprocessing_rop', None)
        if pp_rop is False:
            pp_rop = extraction.ExtractStarsRop(rgb.Templates.LUMINANCE, copy=False, star_size=128, pre_demargin=False)
        self.luma_preprocessing_rop = pp_rop

        if median_shift_limit is not None:
            self.median_shift_limit = median_shift_limit

        if track_roi is not None:
            self.track_roi = track_roi

    def get_state(self):
        return {
            'cache': self.tracking_cache,
        }

    def load_state(self, state):
        self.tracking_cache = state.get('cache')

    def set_reference(self, data):
        # Does nothing
        pass

    def _tracking_key(self, data):
        return getattr(data, 'name', id(data))

    def detect(self, data, bias=None, img=None, save_tracks=None, set_data=True, luma=None, **kw):
        if isinstance(data, list):
            data = data[0]

        if save_tracks is None:
            save_tracks = self.save_tracks

        if self.tracking_cache is None:
            self.tracking_cache = {}

        tracking_key = self._tracking_key(img or data)
        cached = self.tracking_cache.get(tracking_key)

        if cached is None:
            if set_data:
                if self.deglow is not None:
                    data = self.deglow.correct(data.copy())

                self.lraw.set_raw_image(data, add_bias=self.add_bias)

                # Initialize postprocessed image in the main thread
                self.lraw.postprocessed

            if luma is None:
                luma = self.lraw.postprocessed_luma(copy=True)

                if self.luma_preprocessing_rop is not None:
                    luma = self.luma_preprocessing_rop.correct(luma)

            vshape = self.lraw.rimg.raw_image_visible.shape
            lshape = self.lraw.postprocessed.shape
            lyscale = vshape[0] // lshape[0]
            lxscale = vshape[1] // lshape[1]

            tmargin, lmargin, bmargin, rmargin = self.track_roi
            t = int(tmargin * lshape[1])
            l = int(lmargin * lshape[0])
            b = max(0, lshape[1] - int(bmargin * lshape[1]))
            r = max(0, lshape[0] - int(rmargin * lshape[0]))
            luma = luma[t:b, l:r]

            downsample = self.downsample
            if downsample > 1:
                luma = skimage.transform.downscale_local_mean(
                    luma, (downsample,) * len(luma.shape))

            # Transform to srgb normalized
            luma = luma.astype(numpy.float32)
            maxval = luma.max()
            if maxval > 0:
                luma *= (1.0 / luma.max())
            luma = numpy.clip(luma, 0, 1, out=luma)
            luma = srgb.encode_srgb(luma, gamma=self.gamma)
            luma = numpy.clip(luma, 0, 1, out=luma)
            luma *= 255
            luma = luma.astype(numpy.uint8)

            mluma = gaussian.fast_gaussian(luma, self.mask_prefilter)
            mluma = scipy.ndimage.white_tophat(mluma, self.mask_feature_size)
            luma_floor = numpy.median(mluma)
            luma_std = numpy.std(mluma)
            luma_std = numpy.std(mluma[mluma <= (luma_floor + luma_std * self.mask_sigma)])
            mask = mluma > int(luma_floor + luma_std * self.mask_sigma)
            if self.mask_closing:
                mask = scipy.ndimage.binary_closing(mask, skimage.morphology.disk(self.mask_closing))
            if self.mask_opening:
                mask = scipy.ndimage.binary_opening(mask, skimage.morphology.disk(self.mask_opening))
            if self.mask_dilation:
                mask = scipy.ndimage.binary_dilation(mask, skimage.morphology.disk(self.mask_dilation))
            mask = mask.astype(numpy.uint8)

            orb = cv2.ORB_create(self.nfeatures, fastThreshold=self.fast_threshold, WTA_K=self.WTA_K)

            if bias is None:
                if self.reference is None:
                    kp, descr = self.reference = bias = orb.detectAndCompute(luma, mask)
                else:
                    kp, descr = bias = self.reference

            curbias = orb.detectAndCompute(luma, mask)

            matcher = cv2.BFMatcher(self.distance_method, crossCheck=True)
            matches = matcher.match(curbias[1], bias[1])
            matches.sort(key=operator.attrgetter('distance'))
            best_matches = matches[:int(len(matches) * self.keep_matches)]

            if self.show or save_tracks:
                img2 = cv2.drawKeypoints(
                    skimage.color.gray2rgb(luma),
                    curbias[0],
                    skimage.color.gray2rgb(luma),
                    color=(0, 255, 0),
                    flags=0)
                img2[:,:,0][~mask.astype(numpy.bool8)] += 100

                if save_tracks:
                    try:
                        PIL.Image.fromarray(img2).save('Tracks/orb_%s.jpg' % os.path.basename(img.name))
                    except Exception:
                        logger.exception("Can't save tracks due to error")

                if self.show:
                    from matplotlib import pyplot as plt
                    plt.imshow(img2)
                    plt.show()

                del img2

            logger.info("Matched %d/%d features, kept %d", len(matches), len(curbias[0]), len(best_matches))
            matches = best_matches

            if len(matches) < 3:
                logger.warning("Rejecting frame %s due to poor tracking", img)
                return None

            kp1 = bias[0]
            kp2 = curbias[0]

            translations = numpy.array([
                [
                    kp2[m.queryIdx].pt[1] * downsample + t,
                    kp2[m.queryIdx].pt[0] * downsample + l,
                    kp1[m.trainIdx].pt[1] * downsample + t,
                    kp1[m.trainIdx].pt[0] * downsample + l,
                    0,
                    0,
                ]
                for m in matches
            ], dtype=numpy.double)
            translations[:,4:6] = translations[:,2:4] - translations[:,0:2]
            self.tracking_cache[tracking_key] = (translations, vshape, lshape)
            luma = None
        else:
            translations, vshape, lshape = cached

        translations = translations.copy()
        self.lyscale = lyscale = vshape[0] // lshape[0]
        self.lxscale = lxscale = vshape[1] // lshape[1]

        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape

        # Translate luma space to raw space
        translations[:, [0, 2]] /= ysize // lyscale
        translations[:, [1, 3]] /= xsize // lxscale

        transform = find_transform(translations, self.transform_type, self.median_shift_limit, self.force_pass)

        if transform is None:
            logger.warning("Rejecting frame %s due to poor tracking", img)
            return None

        return transform, lyscale, lxscale

    def translate_coords(self, bias, y, x):
        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape
        transform, lyscale, lxscale = bias
        lyscale *= ysize
        lxscale *= xsize
        x, y = transform([[x / lxscale, y / lyscale]])
        return y * lyscale, x * lxscale

    def correct_with_transform(self, data, bias=None, img=None, save_tracks=None, **kw):
        if save_tracks is None:
            save_tracks = self.save_tracks

        dataset = rvdataset = data
        if isinstance(data, list):
            data = data[0]
        else:
            dataset = [data]

        if bias is None:
            bias = self.detect(data, img=img, save_tracks=save_tracks)
            if bias is None:
                # Frame rejected
                return None, None

        transform, lyscale, lxscale = bias

        rvdataset = self.apply_transform(dataset, transform, img=img, **kw)

        return rvdataset, transform

    def clear_cache(self):
        for rop in self.trackers:
            rop.clear_cache()
