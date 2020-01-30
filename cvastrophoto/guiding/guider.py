# -*- coding: utf-8 -*-
from __future__ import absolute_import

import threading
import time
import logging
import collections
import numpy

from .calibration import norm, add
from cvastrophoto.image import base, rgb


logger = logging.getLogger(__name__)


class GuiderProcess(object):

    sleep_period = 0.25
    aggressiveness = 0.8
    drift_aggressiveness = 0.3
    history_length = 5
    min_overlap = 0.5
    save_tracks = False
    save_snaps = False
    snap_gamma = 2.4
    snap_bright = 1.0

    master_dark = None
    img_header = None

    SIDERAL_SPEED = 360 * 3600 / 86400.0

    def __init__(self, telescope, calibration, controller, ccd, ccd_name, tracker_class):
        self.telescope = telescope
        self.ccd = ccd
        self.ccd_name = ccd_name
        self.calibration = calibration
        self.controller = controller
        self.tracker_class = tracker_class

        self.any_event = threading.Event()
        self.wake = threading.Event()

        self._stop = False
        self._stop_guiding = False
        self._start_guiding = False
        self._redo_calibration = False
        self._update_calibration = False
        self._req_snap = False
        self._snap_done = False
        self._get_traces = False
        self._trace_accum = None

        self.runner_thread = None
        self.state = None

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
                    if not self.calibration.is_ready and self.calibration.is_sane:
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
                    self.any_event.set()

            elif self._redo_calibration:
                self.state = 'calibrating'
                self._redo_calibration = False
                self.any_event.set()

                try:
                    logger.info('Calibration started')
                    self.controller.paused = False
                    self.calibration.run()
                except Exception:
                    logger.exception('Error guiding, attempting to restart guiding')
                finally:
                    logger.info('Calibration finished')
                    self.state = 'idle'
                    self.any_event.set()
            elif self._req_snap:
                self.state = 'snap'
                self._req_snap = False
                self.any_event.set()

                try:
                    logger.info('Taking snapshot')
                    self.snap(force_save=True)
                except Exception:
                    logger.exception('Error guiding, attempting to restart guiding')
                finally:
                    logger.info('Snapshot taken')
                    self.state = 'idle'
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
                    self.any_event.set()

                sleep_period = 0.1
            else:
                sleep_period = 5

    def snap(self, img_num=0, force_save=False):
        self.ccd.setLight()
        self.ccd.expose(self.calibration.guide_exposure)
        img = self.ccd.pullImage(self.ccd_name)
        img.name = 'guide_%d' % (img_num,)
        if self.master_dark is not None:
            img.denoise([self.master_dark], entropy_weighted=False)
        self.img_header = img_header = getattr(img, 'fits_header', None)
        if self.save_snaps or force_save or self._req_snap:
            bright = 65535.0 * self.snap_bright / max(1, img.rimg.raw_image.max())
            img.save('guide_snap.jpg', bright=bright, gamma=self.snap_gamma)
            self._snap_done = True
            self._req_snap = False
            self.any_event.set()
        return img

    def guide(self):
        # Get a reference picture out of the guide_ccd to use on the tracker_class
        ref_img = self.snap()

        if not self.calibration.is_ready:
            self.calibration.run(ref_img)
        elif not self.calibration.is_sane:
            self.calibration.update(ref_img)

        if not self.calibration.is_sane:
            logger.error("Calibration results not sane, aborting")
            return

        tracker = self.tracker_class(ref_img)
        img_num = 0

        t1 = time.time()

        offsets = collections.deque(maxlen=self.history_length)
        speeds = collections.deque(maxlen=self.history_length)
        zero_point = (0, 0)
        latest_point = zero_point

        prev_img = None
        wait_pulse = None
        imm_n = imm_w = 0
        stable = False

        while not self._stop_guiding and not self._stop:
            self.wake.wait(self.sleep_period)
            self.wake.clear()
            if self._stop:
                break

            if wait_pulse:
                self.controller.wait_pulse(None, imm_n, imm_w)
                wait_pulse = False

            t0 = t1
            t1 = time.time()
            dt = t1 - t0
            img = self.snap(img_num)
            img_num += 1
            if self._get_traces:
                self.take_trace(img)

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
            offsets.append(offset)

            if dt > 0:
                offset_ec = self.calibration.project_ec(offset)

                agg = self.aggressiveness
                dagg = self.drift_aggressiveness
                exec_ms = self.sleep_period

                imm_w, imm_n = offset_ec
                speed_n = imm_n / dt
                speed_w = imm_w / dt
                imm_w *= agg
                imm_n *= agg

                if stable:
                    speeds.append((speed_w, speed_n, dt, t1))

                max_imm = max(abs(imm_w), abs(imm_n))

                if stable and len(speeds) >= self.history_length:
                    logger.info("Measured drift N/S=%.4f%% W/E=%.4f%%", -speed_n, -speed_w)
                    speed_w, speed_n = self.predict_drift(speeds)
                    logger.info("Predicted drift N/S=%.4f%% W/E=%.4f%%", -speed_n, -speed_w)
                    max_speed = max(abs(speed_n), abs(speed_w))
                    if max_speed < 0.5 or max_imm <= exec_ms:
                        add_drift_w = -speed_w * dagg
                        add_drift_n = -speed_n * dagg
                        logger.info("Update drift N/S=%.4f%% W/E=%.4f%%", add_drift_n, add_drift_w)
                        self.controller.add_drift(add_drift_n, add_drift_w)
                        self.adjust_history(speeds, (add_drift_w, add_drift_n))
                        logger.info("New drift N/S=%.4f%% W/E=%.4f%%",
                            self.controller.ns_drift, self.controller.we_drift)

                if max_imm > exec_ms:
                    # Can't do that correction smoothly
                    self.controller.add_pulse(-imm_n, -imm_w)
                    wait_pulse = True
                    stable = max_imm < (0.5 * dt)
                else:
                    self.controller.add_spread_pulse(-imm_n, -imm_w, exec_ms)
                    stable = True

                logger.info("Guide step X=%.4f Y=%.4f N/S=%.4f W/E=%.4f d=%.4f px",
                    -offset[1], -offset[0], -offset_ec[1], -offset_ec[0],
                    norm(offset))

                self.state = 'guiding'
                self.any_event.set()

        if wait_pulse:
            self.controller.wait_pulse(None, imm_n, imm_w)
            wait_pulse = False

    def predict_drift(self, speeds):
        speed_n = sorted([speed[1] for speed in speeds])[len(speeds)/2]
        speed_w = sorted([speed[0] for speed in speeds])[len(speeds)/2]
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

    def move(self, ns, we, speed):
        # Turn into pulse length assuming calibration.wstep is "speed" times sideral
        ns = ns / (speed * self.SIDERAL_SPEED) * (
            norm(self.calibration.wstep) / norm(self.calibration.nstep))
        we = we / float(speed)

        logger.info("Move will require a guide pulse %.2fs N/S and %.2fs W/E", ns, we)
        self.controller.add_pulse(ns, we)

        return ns, we

    def shift(self, ns, we, speed):
        is_guiding = self.state == 'guiding'
        if is_guiding:
            self.stop_guiding(wait=True)
        ns_s, we_s = self.move(float(ns), float(we), float(speed))
        self.controller.wait_pulse(None, ns_s, we_s)
        if is_guiding:
            self.start_guiding(wait=False)

    def dither(self, px):
        we = px / norm(self.calibration.wstep)
        ns = px / norm(self.calibration.nstep)
        is_guiding = self.state == 'guiding'
        if is_guiding:
            self.stop_guiding(wait=True)
        self.controller.add_pulse(ns, we)
        self.controller.wait_pulse(None, ns, we)
        if is_guiding:
            self.start_guiding(wait=True)

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
        self.save_trace()

    def save_trace(self):
        trace_accum = self._trace_accum
        if trace_accum is not None:
            rimg = trace_accum.average
            rimg = rimg.reshape((rimg.shape[0], rimg.shape[1]/3, 3))
            bright = 65535.0 * self.snap_bright / max(1, rimg.max())
            img = rgb.RGB(None, img=rimg, linear=True)
            img.save('guide_trace.jpg', bright=bright, gamma=self.snap_gamma)

    def stop_trace(self):
        self._trace_accum = None
        self._get_traces = False
