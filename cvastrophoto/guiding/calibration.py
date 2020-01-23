# -*- coding: utf-8 -*-
import logging
import time
import math
from functools import partial


logger = logging.getLogger(__name__)


def dot(a, b):
    ay, ax = a
    by, bx = b
    return ay*by + ax*bx


def norm(a):
    return math.sqrt(dot(a, a))


class CalibrationSequence(object):

    guide_exposure = 4.0

    drift_cycles = 2
    drift_steps = 10
    save_tracks = False

    calibration_min_move_px = 20
    calibration_ra_attempts = 4
    calibration_dec_attempts = 5
    calibration_pulse_s_ra = 0.15
    calibration_pulse_s_dec = 0.15
    calibration_max_pulse_s = 4.0

    clear_backlash_pulse_ra = 5.0
    clear_backlash_pulse_dec = 10.0

    def __init__(self, telescope, controller, ccd, ccd_name, tracker_class):
        self.tracker_class = tracker_class
        self.telescope = telescope
        self.ccd = ccd
        self.ccd_name = ccd_name
        self.controller = controller

        self.wstep = self.nstep = None

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

    def run(self, img=None):
        if img is None:
            # Get a reference picture out of the guide_ccd to use on the tracker_class
            self.ccd.expose(self.guide_exposure)
            img = self.ccd.pullImage(self.ccd_name)

        logger.info("Resetting controller")
        self.controller.reset()

        # First quick drift measurement to allow precise RA/DEC calibration
        logger.info("Performing quick drift and ecuatorial calibration")
        drift, wdrift, ndrift = self.calibrate_axes(img, 'pre', 1)

        # Compute RA/DEC drift and set the controller to compensate, then re-calibrate
        driftwe, driftns = self.project_ec(drift, wdrift, ndrift)

        logger.info("Setting constant drift at %.4f N/S %.4f W/E", driftns, driftwe)
        logger.info("Preliminar N/S (DEC) axis speed at: X=%.4f Y=%.4f (%.4f px/s)",
            ndrift[1], ndrift[0], norm(ndrift))
        logger.info("Preliminar W/E (RA) axis speed at: X=%.4f Y=%.4f (%.4f px/s)",
            wdrift[1], wdrift[0], norm(wdrift))
        self.controller.set_constant_drift(-driftns, -driftwe)

        logger.info("Performing final drift and ecuatorial calibration")
        self._update(img, 'final')

    def update(self, img=None):
        if img is None:
            # Get a reference picture out of the guide_ccd to use on the tracker_class
            self.ccd.expose(self.guide_exposure)
            img = self.ccd.pullImage(self.ccd_name)

        logger.info("Adjusting drift and ecuatorial calibration")
        self._update(img, 'update')

    def _update(self, img, name):
        drift, wdrift, ndrift = self.calibrate_axes(img, name, self.drift_cycles)

        # Adjust RA/DEC drift and set the controller to compensate
        driftwe, driftns = self.project_ec(drift, wdrift, ndrift)

        logger.info("Adding constant drift at %.4f N/S %.4f W/E", driftns, driftwe)
        logger.info("Final N/S (DEC) axis speed at: X=%.4f Y=%.4f (%.4f px/s)",
            ndrift[1], ndrift[0], norm(ndrift))
        logger.info("Final W/E (RA) axis speed at: X=%.4f Y=%.4f (%.4f px/s)",
            wdrift[1], wdrift[0], norm(wdrift))
        self.controller.add_drift(-driftns, -driftwe)

        # Store RA/DEC axes for guiding
        self.wstep = wdrift
        self.nstep = ndrift

    def project_ec(self, drift, wstep=None, nstep=None):
        if wstep is None:
            wstep = self.wstep
        if nstep is None:
            nstep = self.nstep

        driftwe = dot(drift, wstep) / dot(wstep, wstep)
        driftns = dot(drift, nstep) / dot(nstep, nstep)
        return driftwe, driftns

    def calibrate_axes(self, img, name_prefix, drift_cycles):
        # Measure constant drift
        logger.info("Measuring drift at rest")
        drifty, driftx = drift = self.measure_drift(img, drift_cycles, name_prefix, self.combine_drift_avg)

        # Clear backlash in RA
        logger.info("Clearing RA backlash")
        self.controller.pulse_east(self.clear_backlash_pulse_ra)
        self.controller.wait_pulse(self.clear_backlash_pulse_ra * 4)
        self.controller.pulse_west(self.clear_backlash_pulse_ra)
        self.controller.wait_pulse(self.clear_backlash_pulse_ra * 4)

        # Measure west movement direction to get RA axis
        logger.info("Measuring RA axis velocity")
        wdrifty, wdriftx = wdrift = self.measure_axis(
            img, driftx, drifty,
            self.calibration_ra_attempts, self.calibration_pulse_s_ra,
            self.calibration_max_pulse_s, self.calibration_min_move_px,
            self.controller.pulse_west,
            self.controller.wait_pulse,
            self.controller.pulse_east,
            name_prefix + '-w')

        # Clear backlash in DEC
        logger.info("Clearing DEC backlash")
        self.controller.pulse_north(self.clear_backlash_pulse_dec)
        self.controller.wait_pulse(self.clear_backlash_pulse_dec * 4)
        self.controller.pulse_south(self.clear_backlash_pulse_dec)
        self.controller.wait_pulse(self.clear_backlash_pulse_dec * 4)

        # Measure north movement direction to get DEC axis
        logger.info("Measuring DEC axis velocity")
        ndrifty, ndriftx = ndrift = self.measure_axis(
            img, driftx, drifty,
            self.calibration_dec_attempts, self.calibration_pulse_s_dec,
            self.calibration_max_pulse_s, self.calibration_min_move_px,
            self.controller.pulse_north,
            self.controller.wait_pulse,
            self.controller.pulse_south,
            name_prefix + '-n')

        return drift, wdrift, ndrift

    def measure_axis(self,
            img, driftx, drifty,
            attempts, pulse_s, max_pulse_s, min_move_px, pulse_method, wait_method, restore_method,
            name):
        def restore(pulses):
            logger.info("Recentering")
            restore_method(pulses * pulse_s)
            wait_method(pulses * pulse_s * 4)
            logger.info("Recentered")
        for i in xrange(self.calibration_ra_attempts):
            logger.info("Measuring %s drift rate", name)
            wdrifty, wdriftx = self._measure_drift_base(
                img, 1, name, self.combine_drift_avg,
                step_callback=partial(pulse_method, pulse_s),
                post_step_callback=partial(wait_method, 5 * pulse_s),
                total_steps_callback=restore,
                fixed_dt=pulse_s)
            wdrifty -= drifty
            wdriftx -= driftx
            mag = wdrifty*wdrifty + wdriftx*wdriftx
            if mag < min_move_px and pulse_s < max_pulse_s:
                logger.info("Unreliable %s at X=%.4f Y=%.4f (%.4f px/s)", name, wdriftx, wdrifty, mag)
                pulse_s *= min(min_move_px / max(0.01, mag), 4)
                pulse_s = min(pulse_s, max_pulse_s)
                logger.info("Changing pulse to %dms", int(pulse_s * 1000))
            else:
                logger.info("Measured %s at X=%.4f Y=%.4f (%.4f px/s)", name, wdriftx, wdrifty, mag)
                break
        return wdrifty, wdriftx

    def combine_drift_avg(self, drifts):
        dy = sum(dy for dy, dx in drifts) / len(drifts)
        dx = sum(dx for dy, dx in drifts) / len(drifts)
        return (dy, dx)

    def measure_drift(self, ref_img, cycles, which, combine_mode):
        return self._measure_drift_base(ref_img, cycles, which, combine_mode)

    def _measure_drift_base(self,
            ref_img, cycles, which, combine_mode,
            step_callback=None, post_step_callback=None, total_steps_callback=None,
            fixed_dt=None):
        drifts = []
        for cycle in xrange(cycles):
            tracker = self.tracker_class(ref_img)

            offsets = []
            nsteps = 0
            for step in xrange(self.drift_steps):
                t0 = time.time()
                self.ccd.expose(self.guide_exposure)
                img = self.ccd.pullImage(self.ccd_name)
                img.name = 'calibration_drift_%s_%d_%d' % (which, cycle, step)

                if step_callback is not None:
                    step_callback()

                offset = tracker.detect(img.rimg.raw_image, img=img, save_tracks=self.save_tracks)
                offset = tracker.translate_coords(offset, 0, 0)
                offsets.append((offset, t0))

                logger.info("Offset for %s cycle %d/%d step %d/%d at X=%.4f Y=%.4f (d=%.4f px)",
                    which, cycle+1, cycles, step+1, self.drift_steps,
                    offset[1], offset[0], norm(offset))

                if post_step_callback is not None:
                    post_step_callback()
                nsteps += 1

            # First offset will always be 0 since it's the calibration offset
            offsets = offsets[1:]

            # Compute average drift in pixels/s
            dt = offsets[-1][1] - offsets[0][1] if fixed_dt is None else nsteps * fixed_dt
            dy = (offsets[-1][0][0] - offsets[0][0][0])
            dx = (offsets[-1][0][1] - offsets[0][0][1])
            drifts.append((dy/dt, dx/dt))

            if total_steps_callback is not None:
                total_steps_callback(nsteps)

        return combine_mode(drifts)