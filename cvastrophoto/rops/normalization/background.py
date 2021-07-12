from __future__ import absolute_import

import numpy

from . import base
from ..measures import stats

class FullStatsNormalizationRop(stats.StatsMeasureBase, base.PerChannelNormalizationRop):

    reference = None

    bg_mode = 'sub'
    signal_mode = 'mul'

    def set_reference(self, ref):
        self.reference = ref

    def process_channel(self, channel_data, detected=None, channel=None):
        if isinstance(detected, dict):
            detected = detected.get(channel)
        if detected is None:
            detected = self.measure_channel(channel_data.copy(), channel=channel)

        if self.reference is None:
            self.reference = {}
        if channel not in self.reference:
            self.reference[channel] = detected

        bg_avg, bg_std, signal_avg, signal_std = detected
        ref_bg_avg, ref_bg_std, ref_signal_avg, ref_signal_std = self.reference.get(channel, detected)

        signal_mode = self.signal_mode
        bg_mode = self.bg_mode

        if signal_mode is not None:
            if signal_mode == 'mul':
                channel_data = channel_data.astype(numpy.float32, copy=False)
                if signal_avg != 0 and ref_signal_avg != 0:
                    scale_factor = ref_signal_avg / signal_avg
                    channel_data *= scale_factor
                    signal_std *= scale_factor
                    signal_avg *= scale_factor
                    bg_std *= scale_factor
                    bg_avg *= scale_factor
            else:
                raise ValueError("Unsupported signal mode %r" % (signal_mode,))

        if bg_mode is not None:
            if bg_mode == 'sub':
                offset = bg_avg - ref_bg_avg
                channel_data = channel_data.astype(numpy.float32, copy=False)
                channel_data -= offset
            elif bg_mode == 'mul':
                channel_data = channel_data.astype(numpy.float32, copy=False)
                if bg_avg != 0 and ref_bg_avg != 0:
                    scale_factor = ref_bg_avg / bg_avg
                    channel_data *= scale_factor
            else:
                raise ValueError("Unsupported background mode %r" % (bg_mode,))

        return channel_data


class BackgroundNormalizationRop(FullStatsNormalizationRop):
    signal_mode = None


class SignalNormalizationRop(FullStatsNormalizationRop):
    bg_mode = None
