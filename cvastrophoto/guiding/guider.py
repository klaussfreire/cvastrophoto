# -*- coding: utf-8 -*-
from __future__ import absolute_import

import threading
import time
import logging
import collections
import random
import numpy
import functools

from .calibration import norm, add, sub, neg
from . import backlash
from .config import ConfigHelperMixin
from cvastrophoto.image import base, rgb, Image
from cvastrophoto.util import imgscale
from cvastrophoto.util.constants import SIDERAL_SPEED


logger = logging.getLogger(__name__)


def restore_drift(f):
    @functools.wraps(f)
    def decorated(self, *p, **kw):
        saved_drift = self.controller.drift_enabled
        try:
            return f(self, *p, **kw)
        finally:
            self.controller.drift_enabled = saved_drift
    return decorated


class GuiderProcess(ConfigHelperMixin):

    max_sleep_period = 0.25
    rel_sleep_period = 0.25
    ra_aggressiveness = 0.8
    dec_aggressiveness = 0.6
    backlash_aggressiveness = 0.5
    ra_drift_aggressiveness = 0.05
    dec_drift_aggressiveness = 0.01
    dither_aggressiveness = 0.8
    dither_stable_px = 1
    history_length = 5
    min_overlap = 0.5
    drift_enabled = False
    save_tracks = False
    save_snaps = False
    snap_gamma = 2.4
    snap_bright = 1.0

    # Max guide pulse in pixels considered stable enough for drift measurements
    max_stable_shift = 1.0

    # Ratios relative to exposure length
    max_stable_pulse_ratio = 1.0
    max_unstable_pulse_ratio = 2.0
    max_dither_pulse_ratio = 4.0
    max_backlash_pulse_ratio = 1.0

    # Relative to deflection pulse length
    initial_backlash_pulse_ratio = 1.0
    backlash_stop_threshold = 0.75

    # Growth rate between successive backlash pulses
    backlash_ratio_factor = 2.0

    # After this many star lost events, restart guiding
    max_star_lost = 10

    # Max drift speed considered possible in arcsec/s
    max_drift_speed = 20.0

    master_dark = None
    img_header = None

    def __init__(self, telescope, calibration, controller, ccd, ccd_name, tracker_class,
            phdlogger=None, dark_library=None, bias_library=None, config_file=None,
            pool=None):
        self.telescope = telescope
        self.ccd = ccd
        self.ccd_name = ccd_name
        self.calibration = calibration
        self.controller = controller
        self.tracker_class = tracker_class
        self.phdlogger = phdlogger
        self.dark_library = dark_library
        self.bias_library = bias_library
        self.pool = pool

        self.any_event = threading.Event()
        self.offset_event = threading.Event()
        self.wake = threading.Event()

        self._reference_args = None
        self._stop = False
        self._stop_guiding = False
        self._start_guiding = False
        self._redo_calibration = False
        self._update_calibration = False
        self._req_snap = False
        self._snap_done = False
        self._get_traces = False
        self._trace_accum = None
        self._dither_changed = False
        self._snap_listeners = []

        self.runner_thread = None
        self.state = 'not-running'
        self._state_detail = None
        self.dither_offset = (0, 0)
        self.dither_px = 0
        self.lock_pos = None
        self.lock_region = None
        self.dithering = False
        self.eff_max_pulse = 0

        self.offsets = []
        self.speeds = []
        self.ra_speeds = []
        self.dec_speeds = []
        self.stable_history = []

        if config_file is not None:
            self.load_config(config_file, 'GuidingParams')

    @property
    def sleep_period(self):
        return min(
            self.max_sleep_period,
            self.calibration.guide_exposure * self.rel_sleep_period
        )

    @property
    def state_detail(self):
        if self._state_detail is not None:
            return self._state_detail
        elif self.state == 'calibrating':
            return self.calibration.state_detail

    @state_detail.setter
    def state_detail(self, value):
        self._state_detail = value

    def run(self):
        sleep_period = self.sleep_period

        while not self._stop:
            self.wake.wait(sleep_period)
            self.wake.clear()
            if self._stop:
                break

            if self._update_calibration:
                self.state = 'calibrating'
                self._update_calibration = False
                self.any_event.set()

                try:
                    if self.calibration.is_ready and self.calibration.is_sane:
                        logger.info('Calibration started')
                        self.controller.paused = False
                        self.calibration.update()
                    else:
                        logger.info('Basic calibration not yet done')
                except Exception:
                    logger.exception('Error guiding, attempting to restart guiding')
                finally:
                    logger.info('Calibration finished')
                    self.state = 'idle'
                    self.state_detail = None
                    self.any_event.set()

            elif self._redo_calibration:
                self.state = 'calibrating'
                self._redo_calibration = False
                self.any_event.set()

                try:
                    logger.info('Calibration started')
                    self.controller.paused = False
                    self.run_calibration()
                except Exception:
                    logger.exception('Error guiding, attempting to restart guiding')
                finally:
                    logger.info('Calibration finished')
                    self.state = 'idle'
                    self.state_detail = None
                    self.any_event.set()
            elif (
                    self._req_snap
                    or (
                        self._snap_listeners
                        and all(not getattr(snap, 'paused', False) for snap in self._snap_listeners)
                    )):
                force_save = self._req_snap
                self.state = 'snap'
                self._req_snap = False
                self.any_event.set()

                try:
                    logger.info('Taking snapshot')
                    self.snap(force_save=force_save)
                except Exception:
                    logger.exception('Error taking snapshot')
                finally:
                    logger.info('Snapshot taken')
                    self.state = 'idle'
                    self.state_detail = None
                    self.any_event.set()

            if self._start_guiding:
                sleep_period = self.sleep_period
                self.state = 'start-guiding'
                self._stop_guiding = False
                self.any_event.set()
                try:
                    logger.info('Start guiding')
                    self.controller.paused = False
                    self.guide()
                except Exception:
                    logger.exception('Error guiding, attempting to restart guiding')
                finally:
                    logger.info('Stopped guiding')
                    self.state = 'idle'
                    self.state_detail = None
                    self.any_event.set()
            elif self._get_traces:
                self.state = 'trace'
                self.any_event.set()

                try:
                    logger.info('Taking trace snapshot')
                    self.take_trace()
                except Exception:
                    logger.exception('Error taking trace snapshot')
                finally:
                    self.state = 'idle'
                    self.state_detail = None
                    self.any_event.set()

                sleep_period = 0.1
            elif self._snap_listeners:
                sleep_period = self.sleep_period
            else:
                sleep_period = 5

    def snap(self, img_num=0, force_save=False):
        self.ccd.setLight()
        self.ccd.expose(self.calibration.guide_exposure)
        blob = self.ccd.pullBLOB(self.ccd_name)
        img = self.ccd.blob2Image(blob)
        img.default_pool = self.pool
        img.name = 'guide_%d' % (img_num,)
        self.img_header = img_header = getattr(img, 'fits_header', None)

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
            img.denoise([master_dark], entropy_weighted=False, signed=False)

        if self._snap_listeners:
            for listener in self._snap_listeners:
                listener(img)

        if self.save_snaps or force_save or self._req_snap:
            bright = 65535.0 * self.snap_bright / max(1, img.rimg.raw_image.max())
            img.save('guide_snap.jpg', bright=bright, gamma=self.snap_gamma)
            with open('guide_snap.fit', 'wb') as f:
                f.write(blob.getblobdata())
            if master_dark is not None:
                # Denoise saved FIT file
                fit_img = Image.open('guide_snap.fit', mode='update')
                fit_img.denoise([master_dark], entropy_weighted=False, signed=False)
                fit_img.close()
            self._snap_done = True
            self._req_snap = False
            self.any_event.set()

        return img

    def add_snap_listener(self, listener):
        self._snap_listeners.append(listener)
        self.calibration.add_snap_listener(listener)

    def run_calibration(self, *p, **kw):
        self.calibration.run(*p, **kw)

    @restore_drift
    def guide(self):
        # Get a reference picture out of the guide_ccd to use on the tracker_class
        ref_img = self.snap()

        if not self.calibration.is_ready:
            self.state = 'calibrating'
            self.state_detail = None
            self.any_event.set()
            self.run_calibration(ref_img)
        elif not self.calibration.is_sane:
            self.state = 'calibrating'
            self.state_detail = None
            self.any_event.set()
            self.calibration.update(ref_img)

        if not self.calibration.is_sane:
            logger.error("Calibration results not sane, aborting")
            return

        self.state = 'start-guiding'
        self.state_detail = None
        self.any_event.set()

        tracker = self.tracker_class(ref_img)
        img_num = 0
        star_lost_count = 0

        self.controller.drift_enabled = self.drift_enabled

        t1 = time.time()

        self.avg_adus = avg_adus = collections.deque(maxlen=self.history_length)
        self.offsets = offsets = collections.deque(maxlen=self.history_length)
        self.speeds = speeds = collections.deque(maxlen=self.history_length)
        self.ra_speeds = ra_speeds = collections.deque(maxlen=self.history_length)
        self.dec_speeds = dec_speeds = collections.deque(maxlen=self.history_length)
        self.stable_history = stable_history = collections.deque(maxlen=self.history_length)
        zero_point = (0, 0)
        latest_point = zero_point

        prev_img = None
        wait_pulse = None
        imm_n = imm_w = 0
        prev_ec = prev_offset = offset = offset_ec = (0, 0)
        stable = False
        backlash_deadline = None
        had_backlash_ra = had_backlash_dec = False
        shift_ec = None
        backlash_state_dec = backlash.BacklashCompensation.for_controller_dec(self.calibration, self.controller)
        backlash_state_ra = backlash.BacklashCompensation.for_controller_ra(self.calibration, self.controller)
        self.dither_offset = (0, 0)
        self.dithering = dithering = False
        self.dither_stop = False
        phdlogging_started = False

        max_offset = tracker.track_distance * (1.0 - self.min_overlap)
        max_drift_speed = None

        telescope_fl = self.calibration.eff_guider_fl
        ccd_pixel_size = self.calibration.eff_guider_pixel_size
        if telescope_fl and ccd_pixel_size:
            img_scale = imgscale.compute_image_scale(telescope_fl, ccd_pixel_size)
            max_drift_speed = self.max_drift_speed * img_scale
        else:
            logger.info("Image scale unknown, won't be able to validate tracking data")

        while not self._stop_guiding and not self._stop:
            self.wake.wait(self.sleep_period)
            self.wake.clear()
            if self._stop:
                break

            if wait_pulse:
                self.controller.wait_pulse(None, imm_n, imm_w)
                wait_pulse = False

            if self._reference_args is not None:
                logger.info("Resetting tracking reference")
                refp, refkw = self._reference_args
                self._reference_args = None
                tracker.set_reference(*refp, **refkw)
                del refp, refkw

            # We're about to snap, any unexecuted pulse will be moot once we compute a correction
            self.controller.clear_pulse()

            t0 = t1
            t1 = time.time()
            dt = t1 - t0
            prev_ec = offset_ec
            prev_offset = offset
            img = self.snap(img_num)
            img_num += 1
            if self._get_traces:
                self.take_trace(img)

            avg_adu = float(numpy.average(img.rimg.raw_image_visible))
            self.avg_adus.append(avg_adu)

            noffset = tracker.detect(img.rimg.raw_image, img=img, save_tracks=self.save_tracks)
            noffset = tracker.translate_coords(noffset, 0, 0)

            lock_pos = tracker.get_lock_pos()
            lock_region = tracker.get_lock_region()
            if lock_pos is not None:
                self.lock_pos = sub(lock_pos, self.dither_offset)
            if lock_region is not None:
                self.lock_region = lock_region

            if not phdlogging_started and self.phdlogger is not None:
                # Deferred until tracker.detect has been invoked to be able to log the lock pos
                phdlogging_started = True
                try:
                    self.phdlogger.start_guiding(self)
                except Exception:
                    logger.exception("Error writing to PHD log")

            if norm(noffset) > max_offset:
                # Recenter tracker
                logger.info("Offset too large, recentering tracker")
                ntracker = self.tracker_class(ref_img)
                ntracker.detect(prev_img.rimg.raw_image, img=prev_img)
                noffset = tracker.detect(img.rimg.raw_image, img=img, save_tracks=self.save_tracks)
                noffset = tracker.translate_coords(noffset, 0, 0)
                nzero_point = latest_point
            else:
                ntracker = nzero_point = None

            if max_drift_speed is not None and dt > 0 and not dithering and not self._dither_changed:
                max_local_offset = min(max_offset, max_drift_speed * dt)
            else:
                max_local_offset = max_offset

            if norm(noffset) > max_offset or norm(sub(noffset, prev_offset)) > max_local_offset:
                # Recenter tracker
                star_lost_count += 1
                logger.warning("Star lost %d/%d", star_lost_count, self.max_star_lost)
                logger.info("Offset %.3f px, max expected %.3f", norm(noffset), max_local_offset)

                if self.phdlogger is not None:
                    try:
                        self.phdlogger.info("Star lost")
                    except Exception:
                        logger.exception("Error writing to PHD log")

                if star_lost_count > self.max_star_lost:
                    logger.error("Unable to reacquire star, resetting")
                    break

                continue
            else:
                star_lost_count = 0
                if ntracker is not None and nzero_point is not None:
                    tracker = ntracker
                    zero_point = nzero_point
            offset = noffset

            prev_img = img
            img.close()
            tracker.clear_cache()

            if self.dither_stop and norm(offset) < self.dither_px:
                # Force-sync to current offset, reset dither_px to avoid recurring if the second part
                # of dither_stop cannot be triggered yet
                self.dither_offset = neg(offset)
                self.dither_px = 0

            latest_point = offset = add(offset, zero_point)
            offset = add(offset, self.dither_offset)
            offsets.append(offset)

            if self._dither_changed:
                stable = False
                self.dithering = dithering = True
                self._dither_changed = False

                if self.phdlogger is not None:
                    try:
                        self.phdlogger.dither_start(self.dither_offset[1], self.dither_offset[0])
                    except Exception:
                        logger.exception("Error writing to PHD log")

            if dt > 0:
                calibration = self.calibration
                controller = self.controller
                wnorm = calibration.wnorm
                nnorm = calibration.nnorm
                min_pulse_ra = controller.min_pulse_ra
                min_pulse_dec = controller.min_pulse_dec
                max_stable_shift = self.max_stable_shift
                offset_ec = calibration.project_ec(offset)
                diff_ec = sub(offset_ec, prev_ec)
                ign_n, ign_w = controller.pull_ignored()
                diff_ec = add(diff_ec, (ign_w, ign_n))

                if dithering:
                    ra_agg = dec_agg = self.dither_aggressiveness
                    max_pulse = self.max_dither_pulse_ratio
                else:
                    ra_agg = self.ra_aggressiveness
                    dec_agg = self.dec_aggressiveness
                    if stable:
                        max_pulse = self.max_stable_pulse_ratio
                    else:
                        max_pulse = self.max_unstable_pulse_ratio
                self.eff_max_pulse = max_pulse
                ra_dagg = self.ra_drift_aggressiveness
                dec_dagg = self.dec_drift_aggressiveness
                exec_ms = self.sleep_period
                max_pulse = max(exec_ms, max_pulse * calibration.guide_exposure)

                imm_w, imm_n = offset_ec
                diff_w, diff_n = diff_ec
                speed_n = diff_n / dt
                speed_w = diff_w / dt

                getting_backlash_ra = controller.getting_backlash_ra
                getting_backlash_dec = controller.getting_backlash_dec
                getting_backlash = getting_backlash_ra or getting_backlash_dec
                can_drift_update_ra = (stable and not dithering and not getting_backlash_ra
                    and not had_backlash_ra and not ign_w)
                can_drift_update_dec = (stable and not dithering and not getting_backlash_dec
                    and not had_backlash_dec and not ign_n and not (shift_ec and shift_ec[0]))
                can_drift_update = can_drift_update_ra or can_drift_update_dec
                had_backlash_ra = had_backlash_dec = False

                speed_tuple = (speed_w, speed_n, dt, t1)

                if can_drift_update:
                    speeds.append(speed_tuple)
                    if can_drift_update_ra:
                        ra_speeds.append(speed_tuple)
                    if can_drift_update_dec:
                        dec_speeds.append(speed_tuple)

                if can_drift_update and min(len(speeds), len(dec_speeds), len(ra_speeds)) >= self.history_length:
                    logger.info("Measured drift N/S=%.4f%% W/E=%.4f%%", -speed_n, -speed_w)
                    speed_w, _ = self.predict_drift(ra_speeds)
                    _, speed_n = self.predict_drift(dec_speeds)
                    logger.info("Predicted drift N/S=%.4f%% W/E=%.4f%%", -speed_n, -speed_w)

                    if can_drift_update_ra:
                        add_drift_w = -speed_w * ra_dagg
                    else:
                        add_drift_w = 0
                    if can_drift_update_dec:
                        add_drift_n = -speed_n * dec_dagg
                    else:
                        add_drift_n = 0

                    logger.info("Update drift N/S=%.4f%% W/E=%.4f%%", add_drift_n, add_drift_w)
                    controller.add_drift(add_drift_n, add_drift_w)
                    self.adjust_history(speeds, (add_drift_w, add_drift_n))
                    self.adjust_history(ra_speeds, (add_drift_w, add_drift_n))
                    self.adjust_history(dec_speeds, (add_drift_w, add_drift_n))
                    logger.info("New drift N/S=%.4f%% W/E=%.4f%%", controller.ns_drift, controller.we_drift)

                imm_w *= ra_agg if abs(imm_w) >= min_pulse_ra else 0
                imm_n *= dec_agg if abs(imm_n) >= min_pulse_dec else 0

                had_backlash_comp = False
                if getting_backlash:
                    max_backlash_pulse = self.max_backlash_pulse_ratio * calibration.guide_exposure

                    if getting_backlash_ra and imm_w and not ign_w:
                        bcomp = backlash_state_ra.compute_pulse(-imm_w, max_backlash_pulse)
                        imm_w -= bcomp
                        if bcomp:
                            had_backlash_comp = True
                        had_backlash_ra = True
                    else:
                        backlash_state_ra.reset()

                    if getting_backlash_dec and imm_n and not ign_n:
                        bcomp = backlash_state_dec.compute_pulse(-imm_n, max_backlash_pulse)
                        imm_n -= bcomp
                        if bcomp:
                            had_backlash_comp = True
                        had_backlash_dec = True
                    else:
                        backlash_state_dec.reset()

                    if had_backlash_comp:
                        # Reset backlash timeout if we've applied backlash compensation
                        backlash_deadline = None
                else:
                    backlash_state_ra.reset()
                    backlash_state_dec.reset()

                max_imm = max(abs(imm_w), abs(imm_n))
                if max_imm > max_pulse:
                    # Shrink pulse
                    imm_n *= max_pulse / max_imm
                    imm_w *= max_pulse / max_imm

                if max_imm > 0:
                    logger.info("Guide pulse N/S=%.4f W/E=%.4f", -imm_n, -imm_w)
                    controller.set_pulse(-imm_n, -imm_w)
                    wait_pulse = True
                    max_shift = max(abs(imm_n * nnorm), abs(imm_w * wnorm))
                    stable = max_imm < (0.5 * dt) and max_shift < max_stable_shift
                    shift_ec = (-imm_w, -imm_n)
                else:
                    stable = True
                    shift_ec = None

                stable_history.append(stable)

                logger.info("Guide step X=%.4f Y=%.4f N/S=%.4f W/E=%.4f d=%.4f px (%s)",
                    -offset[1], -offset[0], -offset_ec[1], -offset_ec[0],
                    norm(offset),
                    'stable' if stable else 'unstable')

                if self.phdlogger is not None:
                    try:
                        self.phdlogger.guide_step(
                            self, img_num, offset[1], offset[0], offset_ec[0]*wnorm, offset_ec[1]*nnorm,
                            shift_ec[0] if shift_ec else 0, shift_ec[1] if shift_ec else 0,
                            avg_adu)
                    except Exception:
                        logger.exception("Error writing to PHD log")

                if shift_ec:
                    # Reflect added pulse to current offset for a better speed measure later
                    offset_ec = add(offset_ec, shift_ec)
                    logger.info("Recentered offset N/S=%.4f W/E=%.4f", -offset_ec[1], -offset_ec[0])

                if stable and (max_imm < exec_ms or norm(offset) <= self.dither_stable_px or self.dither_stop):
                    if (not getting_backlash or self.dither_stop
                            or (backlash_deadline is not None and time.time() > backlash_deadline)
                            or (self.state == 'guiding' and not had_backlash_comp)):
                        if self.phdlogger is not None and dithering:
                            try:
                                self.phdlogger.dither_finish(self.dither_stop)
                            except Exception:
                                logger.exception("Error writing to PHD log")
                        self.dithering = self.dither_stop = dithering = False
                        self.state = 'guiding'
                        if not getting_backlash:
                            backlash_deadline = None
                    else:
                        self.state = 'guiding-backlash'
                        if backlash_deadline is None:
                            max_backlash = max(calibration.wbacklash, calibration.nbacklash)
                            backlash_deadline = time.time() + (calibration.guide_exposure + max_backlash) * 4
                else:
                    self.state = 'guiding-stabilizing'

            self.offset_event.set()
            self.any_event.set()

        if wait_pulse:
            self.controller.wait_pulse(None, imm_n, imm_w)
            wait_pulse = False

        if self.phdlogger is not None:
            try:
                self.phdlogger.finish_guiding(self)
            except Exception:
                logger.exception("Error writing to PHD log")

        self.lock_pos = None
        self.lock_region = None

    def predict_drift(self, speeds):
        speed_n = sorted([speed[1] for speed in speeds])[len(speeds)//2]
        speed_w = sorted([speed[0] for speed in speeds])[len(speeds)//2]
        return speed_w, speed_n

    def adjust_history(self, speeds, delta):
        lspeeds = list(speeds)
        delta_w, delta_n = delta
        lspeeds = [
            (speed_w + delta_w, speed_n + delta_n, dt, t1)
            for speed_w, speed_n, dt, t1 in lspeeds
        ]
        speeds.clear()
        speeds.extend(lspeeds)

    def wait_stable(self, px, stable_s, stable_s_max, stable_pulses=2):
        # Wait for an offset to be reported, to have some data
        self.offset_event.clear()
        if not self.offset_event.wait(stable_s_max):
            return
        self.offset_event.clear()

        # Wait for dithering to finish, if it's ongoing
        # This will already provide some initial stabilization
        max_deadline = time.time() + stable_s_max
        while ((self._dither_changed or self.dithering
                    or (self.state != 'guiding' and self.controller.getting_effective_backlash))
                and time.time() < max_deadline):
            self.any_event.wait(min(1, max_deadline + 1 - time.time()))
            self.any_event.clear()

        # Wait for it to remain stable for stable_s seconds
        deadline = time.time() + stable_s
        while time.time() < min(deadline, max_deadline):
            if norm(sub(self.offsets[-1], self.offsets[0])) > px:
                deadline = time.time() + stable_s
            elif len(self.stable_history) > stable_pulses and all(list(self.stable_history)[-stable_pulses:]):
                break
            self.offset_event.wait(min(1, max_deadline + 1 - time.time()))
            self.offset_event.clear()

    def start(self):
        if self.runner_thread is None:
            self.runner_thread = threading.Thread(target=self.run)
            self.runner_thread.daemon = True
            self.runner_thread.start()

    def stop(self, wait=True):
        self._stop = True
        self.wake.set()
        if wait:
            self.join()

    def stop_guiding(self, wait=True):
        self._start_guiding = False
        self._stop_guiding = True
        self.wake.set()
        if wait:
            while self.state != 'idle':
                self.any_event.wait(5)
                self.any_event.clear()

    def start_guiding(self, wait=True):
        self._start_guiding = True
        self._stop_guiding = False
        self.wake.set()
        if wait:
            while self.state != 'guiding':
                self.any_event.wait(5)
                self.any_event.clear()

    def request_snap(self, wait=True):
        self._snap_done = False
        self._req_snap = True
        self.wake.set()
        if wait:
            while not self._snap_done:
                self.any_event.wait(5)
                self.any_event.clear()

    def calibrate(self, wait=True):
        self._redo_calibration = True
        self._stop_guiding = True
        self.wake.set()
        if wait:
            while self.state != 'calibrating':
                self.any_event.wait(5)
                self.any_event.clear()
            self._stop_guiding = False
            while self.state == 'calibrating':
                self.any_event.wait(5)
                self.any_event.clear()

    def update_calibration(self, wait=True):
        self._update_calibration = True
        self._stop_guiding = True
        self.wake.set()
        if wait:
            while self.state != 'calibrating':
                self.any_event.wait(5)
                self.any_event.clear()
            self._stop_guiding = False
            while self.state == 'calibrating':
                self.any_event.wait(5)
                self.any_event.clear()

    def join(self):
        if self.runner_thread is not None:
            self.runner_thread.join()
            self.runner_thread = None
            self._stop = False

    def move(self, ns, we, speed=None, seconds=False):
        telescope_fl = self.calibration.eff_guider_fl
        ccd_pixel_size = self.calibration.eff_guider_pixel_size
        if seconds:
            # No translation needed, just typecast in case we get numpy scalars
            ns = float(ns)
            we = float(we)
        elif speed is not None:
            # Turn into pulse length assuming calibration.wstep is "speed" times sideral
            ns = ns / (speed * SIDERAL_SPEED) * (
                norm(self.calibration.wstep) / norm(self.calibration.nstep))
            we = we / float(speed)
        elif telescope_fl and ccd_pixel_size:
            # Turn into pulse length using current calibration data and image scale
            img_scale = imgscale.compute_image_scale(telescope_fl, ccd_pixel_size)
            ns /= img_scale * norm(self.calibration.nstep)
            we /= img_scale * norm(self.calibration.wstep)
        else:
            raise ValueError("Need telescope/ccd information or guiding speed to execute move")

        logger.info("Move will require a guide pulse %.2fs N/S and %.2fs W/E", ns, we)
        self.controller.add_pulse(ns, we)

        return ns, we

    def shift(self, ns, we, speed=None, seconds=False):
        is_guiding = self.state and self.state.startswith('guiding')
        if is_guiding:
            self.stop_guiding(wait=True)
        ns_s, we_s = self.move(float(ns), float(we), float(speed), seconds)
        self.controller.wait_pulse(None, ns_s, we_s)
        if is_guiding:
            self.start_guiding(wait=False)

    def dither(self, px):
        self.dither_px = px
        self.dither_stop = False
        self.dither_offset = (
            (random.random() * 2 - 1) * px,
            (random.random() * 2 - 1) * px,
        )
        self._dither_changed = True

    def stop_dither(self):
        if self._dither_changed or self.dithering:
            self.dither_stop = True

    def set_reference(self, *refp, **refkw):
        self._reference_args = (refp, refkw)

    def start_trace(self):
        self._trace_accum = base.ImageAccumulator()
        self._get_traces = True

    def add_trace(self, img):
        get_traces, trace_accum = self._get_traces, self._trace_accum
        if get_traces and trace_accum is not None:
            trace_accum += img

    def take_trace(self, img=None):
        if img is None:
            img = self.snap()
        self.add_trace(img)
        self.save_trace(img)

    def save_trace(self, img):
        trace_accum = self._trace_accum
        if trace_accum is not None:
            channels = img.rimg.raw_pattern.shape[1]
            rimg = trace_accum.average
            if channels > 1:
                rimg = rimg.reshape((rimg.shape[0], rimg.shape[1]/channels, channels))
            bright = 65535.0 * self.snap_bright / max(1, rimg.max())
            timg = rgb.RGB(None, img=rimg, linear=True)
            timg.save('guide_trace.jpg', bright=bright, gamma=self.snap_gamma)

    def stop_trace(self):
        self._trace_accum = None
        self._get_traces = False
