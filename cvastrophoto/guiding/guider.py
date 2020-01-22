# -*- coding: utf-8 -*-
from __future__ import absolute_import

import threading
import time
import logging
import collections

from .calibration import norm


logger = logging.getLogger(__name__)


class GuiderProcess(object):

    sleep_period = 0.05
    aggressiveness = 1.0
    history_length = 5
    save_tracks = False

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

        self.runner_thread = None
        self.state = None

    def run(self):
        sleep_period = self.sleep_period

        while not self._stop:
            self.wake.wait(sleep_period)
            self.wake.clear()
            if self._stop:
                break

            if self._start_guiding:
                sleep_period = self.sleep_period
                self.state = 'start-guiding'
                self.any_event.set()
                try:
                    self.guide()
                except Exception:
                    logger.exception('Error guiding, attempting to restart guiding')
                finally:
                    self.state = 'idle'
                    self.any_event.set()
            else:
                sleep_period = 5

    def guide(self):
        # Get a reference picture out of the guide_ccd to use on the tracker_class
        self.ccd.expose(self.calibration.guide_exposure)
        ref_img = self.ccd.pullImage(self.ccd_name)

        if not self.calibration.is_ready:
            self.calibration.run(ref_img)
        elif not self.calibration.is_sane:
            self.calibration.update(ref_img)

        tracker = self.tracker_class(ref_img)
        img_num = 0

        t1 = time.time()

        offsets = collections.deque(self.history_length)

        while not self._stop_guiding and not self._stop:
            self.wake.wait(self.sleep_period)
            self.wake.clear()
            if self._stop:
                break

            t0 = t1
            t1 = time.time()
            dt = t1 - t0
            self.ccd.expose(self.guide_exposure)
            img = self.ccd.pullImage(self.ccd_name)
            img.name = 'guide_%s' % (img_num,)

            offset = tracker.detect(img.rimg.raw_image, img=img, save_tracks=self.save_tracks)
            offset = tracker.transform_coords(offset, 0, 0)
            offsets.append(offset)

            if dt > 0:
                offset_ec = self.calibration.project_ec(offset)

                agg = self.aggressiveness
                self.controller.add_spread_pulse(
                    offset_ec[1] * agg, offset_ec[0] * agg,
                    dt * 0.8)

                logger.info("Guide step X=%.4f Y=%.4f N/S=%.4f W/E=%.4f d=%.4f px",
                    offset[1], offset[0], offset_ec[1], offset_ec[0],
                    norm(offset))

                self.state = 'guiding'
                self.any_event.set()

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

    def start_guiding(self, wait=True):
        self._start_guiding = True
        self._stop_guiding = False
        self.wake.set()
        if wait:
            while self.state != 'guiding':
                self.any_event.wait(5)

    def join(self):
        if self.runner_thread is not None:
            self.runner_thread.join()
            self.runner_thread = None
            self._stop = False

