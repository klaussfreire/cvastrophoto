# -*- coding: utf-8 -*-
from __future__ import division

import logging
import time
import math
import numpy
from past.builtins import xrange
from functools import partial

from sklearn import linear_model

from cvastrophoto.util import imgscale


logger = logging.getLogger(__name__)


def dot(a, b):
    ay, ax = a
    by, bx = b
    return ay*by + ax*bx


def norm(a):
    return math.sqrt(dot(a, a))


def norm2(a):
    return dot(a, a)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul(v, factor):
    return (v[0] * factor, v[1] * factor)


class CalibrationSequence(object):

    SIDERAL_SPEED = 360 * 3600 / 86400.0

    guide_exposure = 4.0

    master_dark = None

    guider_fl = None
    imaging_fl = None
    ccd_pixel_size = None
    image_scale = None
    guiding_speed = 1.0

    backlash_cycles = 3
    drift_cycles = 2
    drift_steps = 10
    save_tracks = False
    save_snaps = False
    snap_gamma = 2.4
    snap_bright = 1.0

    stabilization_time = 5.0

    calibration_min_move_px = 20
    calibration_ra_attempts = 6
    calibration_dec_attempts = 6
    calibration_pulse_s_ra = 0.3
    calibration_pulse_s_dec = 0.3
    calibration_max_pulse_s = 6.0

    clear_backlash_pulse_ra = 5.0
    clear_backlash_pulse_dec = 30.0

    min_overlap = 0.5

    img_header = None

    def __init__(self, telescope, controller, ccd, ccd_name, tracker_class,
            phdlogger=None, dark_library=None, bias_library=None, backlash_tracker_class):
        self.tracker_class = tracker_class
        self.backlash_tracker_class = backlash_tracker_class or tracker_class
        self.telescope = telescope
        self.ccd = ccd
        self.ccd_name = ccd_name
        self.controller = controller
        self.phdlogger = phdlogger
        self.dark_library = dark_library
        self.bias_library = bias_library

        self.state = 'uncalibrated'
        self.state_detail = None

        self._snap_listeners = []

        self.eff_calibration_pulse_s_ra = self.calibration_pulse_s_ra
        self.eff_calibration_pulse_s_dec = self.calibration_pulse_s_dec
        self.wstep = self.nstep = self.wnorm = self.nnorm = None
        self.wbacklash = self.nbacklash = None

    @property
    def is_ready(self):
        return self.wstep is not None and self.nstep is not None

    @property
    def is_sane(self):
        if not self.is_ready:
            return False

        wstep_norm = norm(self.wstep)
        nstep_norm = norm(self.nstep)

        if wstep_norm < 0.01 or nstep_norm < 0.01:
            return False
        if wstep_norm < 0.1 * nstep_norm or nstep_norm < 0.1 * wstep_norm:
            return False

        ra_dec_cos_angle = dot(self.wstep, self.nstep) / (wstep_norm * nstep_norm)
        if ra_dec_cos_angle > 0.7:
            return False

        return True

    @property
    def guide_speed(self):
        if self.image_scale and self.wstep:
            return norm(self.wstep) * self.image_scale / self.SIDERAL_SPEED

    @property
    def dec_handedness(self):
        wstep = self.wstep
        canonical_nstep = (wstep[1], -wstep[0])
        n_dot_canonical = dot(self.nstep, canonical_nstep)
        return 1 if n_dot_canonical >= 0 else -1

    def add_snap_listener(self, listener):
        self._snap_listeners.append(listener)

    def run(self, img=None):
        self.state = 'calibrating'

        self.ccd.setLight()
        if img is None:
            # Get a reference picture out of the guide_ccd to use on the tracker_class
            self.ccd.expose(self.guide_exposure)
            img = self.ccd.pullImage(self.ccd_name)
            self.img_header = getattr(img, 'fits_header', None)

        logger.info("Resetting controller")
        self.controller.reset()

        if self.phdlogger is not None:
            try:
                self.phdlogger.start_calibration(self)
            except Exception:
                logger.exception("Error writing to PHD log")

        # First quick drift measurement to allow precise RA/DEC calibration
        logger.info("Performing quick drift and ecuatorial calibration")
        drift, wdrift, ndrift, ra_pulse_s, dec_pulse_s = self.calibrate_axes(img, 'pre', 1)
        self.eff_calibration_pulse_s_ra = ra_pulse_s
        self.eff_calibration_pulse_s_dec = dec_pulse_s

        # Force orthogonal if close enough
        ndrift = self.orthogonalize_n(ndrift, wdrift)

        # Compute RA/DEC drift and set the controller to compensate, then re-calibrate
        driftwe, driftns = self.project_ec(drift, wdrift, ndrift)

        logger.info("Setting constant drift at %.4f N/S %.4f W/E", driftns, driftwe)
        logger.info("Preliminar N/S (DEC) axis speed at: X=%.4f Y=%.4f (%.4f px/s)",
            ndrift[1], ndrift[0], norm(ndrift))
        logger.info("Preliminar W/E (RA) axis speed at: X=%.4f Y=%.4f (%.4f px/s)",
            wdrift[1], wdrift[0], norm(wdrift))
        self.controller.set_constant_drift(-driftns, -driftwe)

        # Store RA/DEC axes for guiding
        self.wstep = wdrift
        self.nstep = ndrift
        self.wnorm = norm(self.wstep)
        self.nnorm = norm(self.nstep)

        if self.phdlogger is not None:
            try:
                self.phdlogger.info("Drift speed RA %.3f, DEC %.3f", driftwe, driftns)
                self.phdlogger.finish_calibration(self)
            except Exception:
                logger.exception("Error writing to PHD log")

        logger.info("Performing final drift and ecuatorial calibration")
        self._update(img, 'final', ra_pulse_s, dec_pulse_s)
        self.eff_calibration_pulse_s_ra = ra_pulse_s
        self.eff_calibration_pulse_s_dec = dec_pulse_s

        # Measure backlash
        self.wbacklash, self.nbacklash = self.measure_backlash(img)

        # Inform controller params
        self.controller.set_backlash(
            self.nbacklash, self.wbacklash,
            self.SIDERAL_SPEED / (self.image_scale * self.wnorm) if self.image_scale and self.wnorm else None)

        self.state = 'calibrated'
        self.state_detail = None

    def update(self, img=None):
        self.state = 'recalibrating'
        self.state_detail = None

        if img is None:
            # Get a reference picture out of the guide_ccd to use on the tracker_class
            self.ccd.expose(self.guide_exposure)
            img = self.ccd.pullImage(self.ccd_name)
            self.img_header = getattr(img, 'fits_header', None)

        logger.info("Adjusting drift and ecuatorial calibration")
        self._update(img, 'update', self.eff_calibration_pulse_s_ra, self.eff_calibration_pulse_s_dec)

    def orthogonalize_n(self, ndrift, wdrift):
        nwe, _ = self.project_ec(ndrift, wdrift, ndrift)
        ortho_ndrift = (ndrift[0] - nwe * wdrift[0], ndrift[1] - nwe * wdrift[1])
        if norm(ortho_ndrift) >= 0.25 * norm(ndrift):
            ndrift = ortho_ndrift
        return ndrift

    def _update(self, img, name, ra_pulse_s=0, dec_pulse_s=0):
        if self.phdlogger is not None:
            try:
                self.phdlogger.start_calibration(self)
            except Exception:
                logger.exception("Error writing to PHD log")

        drift, wdrift, ndrift, ra_pulse_s, dec_pulse_s = self.calibrate_axes(
            img, name, self.drift_cycles, ra_pulse_s, dec_pulse_s, False)

        # Adjust RA/DEC drift and set the controller to compensate
        driftwe, driftns = self.project_ec(drift, wdrift, ndrift)

        # Force orthogonal if close enough
        ndrift = self.orthogonalize_n(ndrift, wdrift)

        logger.info("Setting constant drift at %.4f N/S %.4f W/E", driftns, driftwe)
        logger.info("Final N/S (DEC) axis speed at: X=%.4f Y=%.4f (%.4f px/s)",
            ndrift[1], ndrift[0], norm(ndrift))
        logger.info("Final W/E (RA) axis speed at: X=%.4f Y=%.4f (%.4f px/s)",
            wdrift[1], wdrift[0], norm(wdrift))
        self.controller.set_constant_drift(-driftns, -driftwe)

        # Store RA/DEC axes for guiding
        self.wstep = wdrift
        self.nstep = ndrift
        self.wnorm = norm(self.wstep)
        self.nnorm = norm(self.nstep)

        if self.phdlogger is not None:
            try:
                self.phdlogger.finish_calibration(self)
            except Exception:
                logger.exception("Error writing to PHD log")

    def project_ec(self, drift, wstep=None, nstep=None):
        if wstep is None:
            wstep = self.wstep
        if nstep is None:
            nstep = self.nstep

        driftwe = dot(drift, wstep) / dot(wstep, wstep)
        driftns = dot(drift, nstep) / dot(nstep, nstep)
        return driftwe, driftns

    @property
    def eff_telescope_info(self):
        info_source = self.telescope or self.controller.st4
        return info_source.properties.get('TELESCOPE_INFO', [None]*4)

    @property
    def eff_telescope_coords(self):
        info_source = self.telescope or self.controller.st4

        value = None
        for coord_attr in filter(None, [
                    getattr(info_source, 'COORD_J2000', None),
                    getattr(info_source, 'COORD_EOD', None),
                    "EQUATORIAL_COORD",
                    "EQUATORIAL_EOD_COORD",
                ]):
            value = info_source.properties.get(coord_attr)
            if value is not None:
                break

        return value

    @property
    def eff_telescope_hcoords(self):
        info_source = self.telescope or self.controller.st4

        value = None
        for coord_attr in filter(None, [
                    "HORIZONTAL_COORD",
                ]):
            value = info_source.properties.get(coord_attr)
            if value is not None:
                break

        return value

    @property
    def eff_guider_fl(self):
        telescope_info = self.eff_telescope_info
        return self.guider_fl or telescope_info[3] or None

    @property
    def eff_imaging_fl(self):
        telescope_info = self.eff_telescope_info
        return self.imaging_fl or telescope_info[1] or None

    @property
    def eff_guider_pixel_size(self):
        ccd_info = self.ccd.properties.get('CCD_INFO', [None]*5)
        return self.ccd_pixel_size or ccd_info[2] or None

    def measure_backlash(self, ref_img):
        wbacklashs = []
        nbacklashs = []
        pulse_ra = self.clear_backlash_pulse_ra
        pulse_dec = self.clear_backlash_pulse_dec

        last_cycle = self.backlash_cycles - 1
        pulse_ns_forth = self.controller.pulse_north
        pulse_ns_back = self.controller.pulse_south
        pulse_we_forth = self.controller.pulse_west
        pulse_we_back = self.controller.pulse_east
        direction = 1
        for i in xrange(self.backlash_cycles):
            wbacklash = max(0, self._measure_backlash(
                ref_img, 'w', i,
                pulse_we_forth,
                pulse_we_back,
                self.controller.wait_pulse,
                pulse_ra,
                i == 0,
                i == last_cycle,
                direction)[0])

            nbacklash = max(0, self._measure_backlash(
                ref_img, 'n', i,
                pulse_ns_forth,
                pulse_ns_back,
                self.controller.wait_pulse,
                pulse_dec,
                i == 0,
                i == last_cycle,
                direction)[1])

            logger.info("Measured backlash at: RA=%.4f s DEC=%.4f s", wbacklash, nbacklash)

            wbacklashs.append(wbacklash)
            nbacklashs.append(nbacklash)
            pulse_ra = min(pulse_ra, wbacklash * 2)
            pulse_dec = min(pulse_dec, nbacklash * 2)

            # Flip direction
            direction = -direction
            pulse_ns_forth, pulse_ns_back = pulse_ns_back, pulse_ns_forth
            pulse_we_forth, pulse_we_back = pulse_we_back, pulse_we_forth

        wbacklash = float(min(wbacklashs))
        nbacklash = float(min(nbacklashs))
        logger.info("Measured final backlash at: RA=%.4f s DEC=%.4f s", wbacklash, nbacklash)

        return wbacklash, nbacklash

    def _measure_backlash(self,
            ref_img, which, cycle, pulse_method, pulse_back_method, wait_method, pulse_length,
            clear_initial, clear_final, direction):
        tracker = self.backlash_tracker_class(ref_img)

        self.state_detail = 'backlash-%s (1/4 cycle %d/%d)' % (which, cycle+1, self.backlash_cycles)

        if clear_initial:
            # Clear any initial backlash
            pulse_back_method(pulse_length)
            wait_method(pulse_length * 4)
            time.sleep(0.25)
            pulse_method(pulse_length)
            wait_method(pulse_length * 4)

        # Wait a tiny bit, to let it settle
        time.sleep(0.25)

        offsets = []

        for step in xrange(2):
            # Take reference image
            self.state_detail = 'backlash-%s (%d/4 cycle %d/%d)' % (which, step+1, cycle+1, self.backlash_cycles)
            self.ccd.expose(self.guide_exposure)
            img = self.ccd.pullImage(self.ccd_name)
            img.name = 'calibration_backlash_%s_%d_%d' % (which, step, cycle)
            self.img_header = getattr(img, 'fits_header', None)

            master_dark = self.master_dark
            if master_dark is None and self.dark_library is not None:
                dark_key = self.dark_library.classify_frame(img)
                if dark_key is not None:
                    master_dark = self.dark_library.get_master(dark_key, raw=img)
            if master_dark is None and self.bias_library is not None:
                dark_key = self.bias_library.classify_frame(img)
                if dark_key is not None:
                    master_dark = self.bias_library.get_master(dark_key, raw=img)
            if master_dark is not None:
                img.denoise([master_dark], entropy_weighted=False)

            if self._snap_listeners:
                for listener in self._snap_listeners:
                    listener(img)
            if self.save_snaps:
                bright = 65535.0 * self.snap_bright / max(1, img.rimg.raw_image.max())
                img.save('calibration_snap.jpg', bright=bright, gamma=self.snap_gamma)

            offset = tracker.detect(img.rimg.raw_image, img=img, save_tracks=self.save_tracks)
            offset = tracker.translate_coords(offset, 0, 0)
            offsets.append(mul(offset, direction))

            img.close()
            tracker.clear_cache()

            if step:
                break

            # Induce backlash to measure it
            pulse_method(pulse_length)
            wait_method(pulse_length * 4)
            time.sleep(0.25)
            pulse_back_method(pulse_length)
            wait_method(pulse_length * 4)
            time.sleep(0.25)

        self.state_detail = 'backlash-%s (4/4 cycle %d/%d)' % (which, cycle+1, self.backlash_cycles)

        if clear_final:
            # Clear any leftover backlash again
            pulse_back_method(pulse_length)
            wait_method(pulse_length * 4)
            time.sleep(0.25)
            pulse_method(pulse_length)

        # Compute and log backlash
        backlash = sub(offsets[1], offsets[0])
        backlash_ec = self.project_ec(backlash)

        logger.info("Measured %s backlash at: RA=%.4f s DEC=%.4f s", which, backlash_ec[0], backlash_ec[1])

        if clear_final:
            # Wait for mount to settle
            wait_method(pulse_length * 4)
        time.sleep(0.25)

        return backlash_ec

    def calibrate_axes(self, img, name_prefix, drift_cycles, ra_pulse_s=0, dec_pulse_s=0, subtract_drift=True):
        # Measure constant drift
        logger.info("Measuring drift at rest")
        try:
            self.controller.paused_drift = True
            drifty, driftx = drift = self.measure_drift(img, drift_cycles, name_prefix, self.combine_drift_avg)
        finally:
            self.controller.paused_drift = False

        # Estimate intial pulse lengths from guiding FOV if available
        telescope_fl = self.eff_guider_fl
        ccd_pixel_size = self.eff_guider_pixel_size
        if telescope_fl and ccd_pixel_size:
            # Turn into pulse length using current calibration data and image scale
            self.image_scale = img_scale = imgscale.compute_image_scale(telescope_fl, ccd_pixel_size)

            speed = self.guiding_speed * self.SIDERAL_SPEED
            ra_pulse_s = min(self.calibration_max_pulse_s, max(self.calibration_pulse_s_ra, ra_pulse_s,
                self.calibration_min_move_px * 1.25 * img_scale / speed / self.drift_steps))
            dec_pulse_s = min(self.calibration_max_pulse_s, max(self.calibration_pulse_s_dec, dec_pulse_s,
                self.calibration_min_move_px * 1.25 * img_scale / speed / self.drift_steps))
        else:
            ra_pulse_s = max(ra_pulse_s, self.calibration_pulse_s_ra)
            dec_pulse_s = max(dec_pulse_s, self.calibration_pulse_s_dec)

        if subtract_drift:
            sdrifty, sdriftx = drift
        else:
            sdrifty = sdriftx = 0

        # Measure west movement direction to get RA axis
        logger.info("Measuring RA axis velocity")
        wdrift, ra_pulse_s = self.measure_axis(
            img, sdriftx, sdrifty,
            self.calibration_ra_attempts, ra_pulse_s,
            self.calibration_max_pulse_s, self.clear_backlash_pulse_ra,
            self.calibration_min_move_px,
            self.controller.pulse_west,
            self.controller.wait_pulse,
            self.controller.pulse_east,
            name_prefix + '-w')

        # Measure north movement direction to get DEC axis
        logger.info("Measuring DEC axis velocity")
        ndrift, dec_pulse_s = self.measure_axis(
            img, sdriftx, sdrifty,
            self.calibration_dec_attempts, dec_pulse_s,
            self.calibration_max_pulse_s, self.clear_backlash_pulse_dec,
            self.calibration_min_move_px,
            self.controller.pulse_north,
            self.controller.wait_pulse,
            self.controller.pulse_south,
            name_prefix + '-n')

        return drift, wdrift, ndrift, ra_pulse_s, dec_pulse_s

    def measure_axis(self,
            img, driftx, drifty,
            attempts, pulse_s, max_pulse_s, backlash_pulse_s, min_move_px,
            pulse_method, wait_method, restore_method,
            name):
        def restore(pulses):
            logger.info("Recentering")
            self.state_detail = 'recentering'
            restore_method(pulses * pulse_s)
            wait_method(pulses * pulse_s * 4)
            logger.info("Recentered")

        def clear_backlash():
            logger.info("Clearing backlash")
            self.state_detail = 'clearing backlash'
            restore_method(backlash_pulse_s)
            wait_method(backlash_pulse_s * 4)
            pulse_method(backlash_pulse_s)
            wait_method(backlash_pulse_s * 4)

        for i in xrange(attempts):
            logger.info("Measuring %s drift rate, pulse %dms", name, int(pulse_s * 1000))
            clear_backlash()
            wdrifty, wdriftx, dt, nsteps = self._measure_drift_base(
                img, 1, name, self.combine_drift_avg,
                step_callback=partial(pulse_method, pulse_s),
                post_step_callback=partial(wait_method, 5 * pulse_s),
                total_steps_callback=restore)
            wdrifty = ((wdrifty - drifty) * dt) / (pulse_s * (nsteps - 1))
            wdriftx = ((wdriftx - driftx) * dt) / (pulse_s * (nsteps - 1))
            mag = math.sqrt(wdrifty*wdrifty + wdriftx*wdriftx)
            abs_mag = mag * pulse_s * self.drift_steps
            if abs_mag < min_move_px and pulse_s < max_pulse_s:
                logger.info("Unreliable %s at X=%.4f Y=%.4f (%.4f px/s - %.4f px sampled)",
                    name, wdriftx, wdrifty, mag, abs_mag)
                pulse_s *= min(min_move_px * 1.2 / max(0.01, abs_mag), 4)
                pulse_s = min(pulse_s, max_pulse_s)
                logger.info("Changing pulse to %dms", int(pulse_s * 1000))
            else:
                logger.info("Measured %s at X=%.4f Y=%.4f (%.4f px/s - %.4f px sampled)",
                    name, wdriftx, wdrifty, mag, abs_mag)
                break
        return (wdrifty, wdriftx), pulse_s

    def combine_drift_avg(self, drifts):
        X = numpy.concatenate([d[0] for d in drifts])
        Y = numpy.concatenate([d[1] for d in drifts])
        X = X.reshape((len(X), 1))

        y = Y[:,0]
        ythreshold = numpy.median(numpy.abs(y - numpy.median(y))) + 1
        ransac = linear_model.RANSACRegressor(residual_threshold=ythreshold)
        ransac.fit(X, y)
        dy0, dy1 = ransac.predict([[0], [1]])

        y = Y[:,1]
        ythreshold = numpy.median(numpy.abs(y - numpy.median(y))) + 1
        ransac = linear_model.RANSACRegressor(residual_threshold=ythreshold)
        ransac.fit(X, y)
        dx0, dx1 = ransac.predict([[0], [1]])

        n = sum(d[2] for d in drifts) / len(drifts)
        dy = dy1 - dy0
        dx = dx1 - dx0
        dt = X.ptp()

        return (dy, dx, dt, n)

    def estimate_drift(self, offsets):
        t0 = offsets[0][1]
        X = numpy.array([t - t0 for offset, t in offsets], dtype=numpy.double)
        Y = numpy.array([list(offset) for offset, t in offsets])
        return X, Y, len(offsets)

    def measure_drift(self, ref_img, cycles, which, combine_mode):
        return self._measure_drift_base(ref_img, cycles, which, combine_mode)[:2]

    def _measure_drift_base(self,
            ref_img, cycles, which, combine_mode,
            step_callback=None, post_step_callback=None, total_steps_callback=None):
        drifts = []
        for cycle in xrange(cycles):
            tracker = self.tracker_class(ref_img)

            zero_point = (0, 0)
            latest_point = zero_point
            offsets = []
            nsteps = 0
            prev_img = None
            for step in xrange(self.drift_steps):
                self.state_detail = '%s (cycle %d/%d step %d/%d)' % (
                    which, cycle, cycles, step, self.drift_steps)
                t0 = time.time()
                self.ccd.expose(self.guide_exposure)
                img = self.ccd.pullImage(self.ccd_name)
                img.name = 'calibration_drift_%s_%d_%d' % (which, cycle, step)
                self.img_header = img_header = getattr(img, 'fits_header', None)

                last_step = (step+1) >= self.drift_steps
                if not last_step:
                    nsteps += 1
                    if step_callback is not None:
                        step_callback()

                if self.master_dark is not None:
                    img.denoise([self.master_dark], entropy_weighted=False)
                if self._snap_listeners:
                    for listener in self._snap_listeners:
                        listener(img)
                if self.save_snaps:
                    bright = 65535.0 * self.snap_bright / max(1, img.rimg.raw_image.max())
                    img.save('calibration_snap.jpg', bright=bright, gamma=self.snap_gamma)

                offset = tracker.detect(img.rimg.raw_image, img=img, save_tracks=self.save_tracks)
                offset = tracker.translate_coords(offset, 0, 0)

                if norm(offset) > tracker.track_distance * (1.0 - self.min_overlap):
                    # Recenter tracker
                    logger.info("Offset too large, recentering tracker")
                    tracker = self.tracker_class(ref_img)
                    tracker.detect(prev_img.rimg.raw_image, img=prev_img)
                    offset = tracker.detect(img.rimg.raw_image, img=img, save_tracks=self.save_tracks)
                    offset = tracker.translate_coords(offset, 0, 0)
                    zero_point = latest_point

                prev_img = img
                img.close()
                tracker.clear_cache()

                latest_point = offset = add(offset, zero_point)
                offsets.append((offset, t0))

                if self.phdlogger is not None:
                    try:
                        if which.endswith('-n'):
                            direction = 'North'
                        elif which.endswith('-w'):
                            direction = 'West'
                        else:
                            direction = 'East'
                        self.phdlogger.calibration_step(direction, step, offset[1], offset[0])
                    except Exception:
                        logger.exception("Error writing to PHD log")

                logger.info("Offset for %s cycle %d/%d step %d/%d at X=%.4f Y=%.4f (d=%.4f px)",
                    which, cycle+1, cycles, step+1, self.drift_steps,
                    offset[1], offset[0], norm(offset))

                if post_step_callback is not None and not last_step:
                    post_step_callback()

            # First offset will always be 0 since it's the calibration offset
            offsets = offsets[1:]

            # Compute average drift in pixels/s
            drifts.append(self.estimate_drift(offsets))

            if total_steps_callback is not None:
                total_steps_callback(nsteps)

        self.state_detail = 'stabilize after %s' % (which,)
        time.sleep(self.stabilization_time)

        return combine_mode(drifts)
