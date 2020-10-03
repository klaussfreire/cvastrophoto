# -*- coding: utf-8 -*-
from __future__ import division

import threading
import time
import logging

from cvastrophoto.util.signedmag import min_directed

logger = logging.getLogger(__name__)


class GuiderController(object):

    target_pulse = 0.05
    min_pulse = 0.025
    max_pulse = 1.0

    pulse_period = 0.5

    dec_switch_resistence = 0.05
    ra_switch_resistence = 0.05
    dec_max_switch_resistence = 0.125
    ra_max_switch_resistence = 0.125
    backlash_detection_magin = 0.98

    # max_gear_state is 1/2 of backlash, we limit resistence to twice the backlash
    # If we have to switch for a pulse twice as long as the backlash, it's viable and worth it
    resistence_backlash_ratio = 4

    def __init__(self, telescope, st4):
        self.reset()
        self._stop = False
        self.runner_thread = None
        self.wake = threading.Event()
        self.pulse_event = threading.Event()
        self.spread_pulse_event = threading.Event()
        self.telescope = telescope
        self.st4 = st4

    def reset(self):
        self.ns_drift = 0
        self.we_drift = 0
        self.ns_pulse = 0
        self.we_pulse = 0
        self.ns_ignored = self.total_ns_ignored = 0
        self.we_ignored = self.total_we_ignored = 0
        self.ns_drift_extra = 0
        self.we_drift_extra = 0
        self.drift_extra_deadline = 0
        self.drift_extra_time = 0
        self.gear_state_ns = self.unsync_gear_state_ns = 0
        self.gear_state_we = self.unsync_gear_state_we = 0
        self.max_gear_state_ns = self.eff_max_gear_state_ns = 0
        self.max_gear_state_we = self.eff_max_gear_state_we = 0
        self.backlash_measured = False
        self.gear_rate_we = 1.0
        self.paused = False
        self.paused_drift = False

    def set_backlash(self, nbacklash, wbacklash, gear_rate_we):
        self.max_gear_state_ns = self.eff_max_gear_state_ns = abs(nbacklash or 0) / 2
        self.max_gear_state_we = self.eff_max_gear_state_we = abs(wbacklash or 0) / 2
        if gear_rate_we is not None:
            self.gear_rate_we = gear_rate_we
        self.backlash_measured = (nbacklash or wbacklash) is not None

    def sync_gear_state_ra(self, direction, max_shrink=1):
        if direction:
            sign = 1 if direction > 0 else -1

            # Auto-shrink
            shrunk_max_gear_state_we = (
                self.max_gear_state_we - abs(sign * self.max_gear_state_we - self.gear_state_we) * 0.5
            )
            new_max_gear_state_we = max(self.max_gear_state_we * max_shrink, shrunk_max_gear_state_we)
            if new_max_gear_state_we != self.max_gear_state_we:
                logger.info("Sync RA state from %.2f max %.2f dir %d",
                    self.gear_state_we, self.max_gear_state_we, sign)
                logger.info("Shrinking RA backlash to %.2f", new_max_gear_state_we * 2)
                self.max_gear_state_we = new_max_gear_state_we
            self.eff_max_gear_state_we = min(self.eff_max_gear_state_we, shrunk_max_gear_state_we)

            self.gear_state_we = sign * self.max_gear_state_we

    def sync_gear_state_dec(self, direction, max_shrink=1):
        if direction:
            sign = 1 if direction > 0 else -1

            # Auto-shrink
            shrunk_max_gear_state_ns = (
                self.max_gear_state_ns - abs(sign * self.max_gear_state_ns - self.gear_state_ns) * 0.5
            )
            new_max_gear_state_ns = max(self.max_gear_state_ns * max_shrink, shrunk_max_gear_state_ns)
            if new_max_gear_state_ns != self.max_gear_state_ns:
                logger.info("Sync DEC state from %.2f max %.2f dir %d",
                    self.gear_state_ns, self.max_gear_state_ns, sign)
                logger.info("Shrinking DEC backlash to %.2f", new_max_gear_state_ns * 2)
                self.max_gear_state_ns = new_max_gear_state_ns
            self.eff_max_gear_state_ns = min(self.eff_max_gear_state_ns, shrunk_max_gear_state_ns)

            self.gear_state_ns = sign * self.max_gear_state_ns

    def _eff_switch_resistence(self, resistence, max_resistence, max_gear_state, max_other_gear_state):
        if self.backlash_measured or max_gear_state or max_other_gear_state:
            return max(min(max_resistence, max_gear_state * self.resistence_backlash_ratio), resistence)
        else:
            return resistence

    @property
    def _eff_ra_switch_resistence(self):
        return self._eff_switch_resistence(
            self.ra_switch_resistence, self.ra_max_switch_resistence,
            self.eff_max_gear_state_we, self.eff_max_gear_state_ns)

    @property
    def _eff_dec_switch_resistence(self):
        return self._eff_switch_resistence(
            self.dec_switch_resistence, self.dec_max_switch_resistence,
            self.eff_max_gear_state_ns, self.eff_max_gear_state_we)

    @property
    def eff_drift(self):
        if self.paused_drift:
            return (0, 0)
        else:
            return (self.ns_drift, self.we_drift)

    @property
    def getting_backlash(self):
        return (
            abs(self.unsync_gear_state_ns) < abs(self.max_gear_state_ns * self.backlash_detection_magin)
            or abs(self.unsync_gear_state_we) < abs(self.max_gear_state_we * self.backlash_detection_magin)
        )

    @property
    def getting_backlash_ra(self):
        return abs(self.unsync_gear_state_we) < abs(self.max_gear_state_we * self.backlash_detection_magin)

    @property
    def getting_backlash_dec(self):
        return abs(self.unsync_gear_state_ns) < abs(self.max_gear_state_ns * self.backlash_detection_magin)

    def backlash_compensation_ra(self, pulse):
        return self._backlash_compensation(pulse, self.gear_state_we, self.max_gear_state_we)

    def backlash_compensation_dec(self, pulse):
        return self._backlash_compensation(pulse, self.gear_state_ns, self.max_gear_state_ns)

    def _backlash_compensation(self, pulse, state, max_state):
        if not pulse:
            return 0

        sign = 1 if pulse > 0 else -1
        return sign * max(0, min(2*max_state, abs(sign * max_state - state)))

    def set_constant_drift(self, ns, we):
        """ Set a constant, smooth drift

        ns and we are in "percent duty cycle", where 0 means
        no movement, and 1 means a constant pulse (maximum
        movement) in the direction, -1 being the opposite direction.
        """
        self.ns_drift = ns
        self.we_drift = we
        self.wake.set()

    def add_pulse(self, ns_s, we_s):
        """ Immediately schedule a guiding pulse of a limited duration

        The pulse will be executed by adding ns_s and we_s time
        to the constant drift in each direction (or subtracting if
        in the opposite direction), which results in a movement that
        is added to that of the constant drift. If current constant
        drift is high, execution can take considerably longer than
        the requested pulse duration.
        """
        self.pulse_event.clear()
        self.ns_pulse += ns_s
        self.we_pulse += we_s
        self.wake.set()

    def set_gear_state(self, ns_state, we_state):
        self.gear_state_ns = self.unsync_gear_state_ns = ns_state
        self.gear_state_we = self.unsync_gear_state_we = we_state

    def add_gear_state(self, ns_pulse, we_pulse):
        self.gear_state_ns = max(min(
            self.gear_state_ns + ns_pulse, self.max_gear_state_ns), -self.max_gear_state_ns)
        self.gear_state_we = max(min(
            self.gear_state_we + we_pulse, self.max_gear_state_we), -self.max_gear_state_we)
        self.unsync_gear_state_ns = max(min(
            self.unsync_gear_state_ns + ns_pulse, self.max_gear_state_ns), -self.max_gear_state_ns)
        self.unsync_gear_state_we = max(min(
            self.unsync_gear_state_we + we_pulse, self.max_gear_state_we), -self.max_gear_state_we)

    def wait_pulse(self, timeout=None, ns=None, we=None):
        """ Wait until the current pulse has been fully executed """
        if timeout is None and (ns or we):
            timeout = max(abs(ns), abs(we)) * 4
        if self.pulse_event.wait(timeout):
            self.pulse_event.clear()

    def pulse_north(self, ms):
        self.add_pulse(ms, 0)

    def pulse_south(self, ms):
        self.add_pulse(-ms, 0)

    def pulse_west(self, ms):
        self.add_pulse(0, ms)

    def pulse_east(self, ms):
        self.add_pulse(0, -ms)

    def add_drift(self, ns, we):
        """ Change the constant drift gradually by the amount given

        Add ns and we to the constant drift. Note that the constant
        drift cannot exceed 1 in magnitude, so it will be clamped
        to the [-1, 1] range.
        """
        self.ns_drift = max(-1, min(1, self.ns_drift + ns))
        self.we_drift = max(-1, min(1, self.we_drift + we))
        self.wake.set()

    def add_spread_pulse(self, ns_s, we_s, exec_s):
        """ Schedule a spread pulse - an momentary increase in drift rate

        Schedule a momentary increase in constant drift that executes
        a guiding pulse of ns_s/we_s smoothly during the next exec_s.

        Assuming no constant drift is currently being executed, this
        sets the drift to ns_s / exec_s for exec_s (similarly for we).

        In the presence of a pre-existing constant drift, maximum drift
        speed limits may lengthen execution.

        Only one such pulse may be active at any time, any call while
        one spread pulse is being executed will abort that one and be
        replaced by the new pulse request.
        """
        exec_s = max(exec_s, we_s, ns_s)
        ns_speed = ns_s / exec_s
        we_speed = we_s / exec_s

        if abs(ns_speed + self.ns_drift) > 1 or abs(we_speed + self.we_drift) > 1:
            slow_factor = max(abs(ns_speed + self.ns_drift), abs(we_speed + self.we_drift))
            exec_s *= slow_factor
            we_speed /= slow_factor
            ns_speed /= slow_factor

        self.ns_drift_extra = ns_speed
        self.we_drift_extra = we_speed
        self.drift_extra_time = exec_s
        self.spread_pulse_event.clear()
        self.wake.set()

    def pull_ignored(self):
        ns_ignored = self.ns_ignored
        we_ignored = self.we_ignored
        self.ns_ignored = self.we_ignored = 0
        return ns_ignored, we_ignored

    def wait_spread_pulse(self, timeout=None):
        self.spread_pulse_event.wait(timeout)

    def run(self):
        cur_ns_duty = 0
        cur_we_duty = 0
        self.pulse_period = cur_period = 0.5
        sleep_period = cur_period

        pulse_deadline = last_pulse = time.time()

        self.doing_pulse = doing_pulse = False

        ns_dir = we_dir = 0

        while not self._stop:
            self.wake.wait(sleep_period)
            self.wake.clear()
            if self._stop:
                break
            elif self.paused:
                sleep_period = self.pulse_period
                continue

            now = time.time()
            if pulse_deadline > now:
                # Do not interfere with running pulses
                sleep_period = max(pulse_deadline - now, self.min_pulse)
                continue

            min_pulse = self.min_pulse
            if doing_pulse and abs(cur_ns_duty) < min_pulse and abs(cur_we_duty) < min_pulse:
                self.doing_pulse = doing_pulse = False
                self.pulse_event.set()

            now = time.time()
            delta = now - last_pulse
            drift_delta = delta

            ns_drift, we_drift = self.eff_drift

            direct_ns_pulse = self.ns_pulse
            direct_we_pulse = self.we_pulse
            self.ns_pulse -= direct_ns_pulse
            self.we_pulse -= direct_we_pulse
            cur_ns_duty += ns_drift * drift_delta + direct_ns_pulse
            cur_we_duty += we_drift * drift_delta + direct_we_pulse
            if direct_ns_pulse or direct_we_pulse:
                self.doing_pulse = doing_pulse = True

            drift_extra_time = self.drift_extra_time
            if drift_extra_time:
                self.drift_extra_deadline = now + drift_extra_time
                self.drift_extra_time = 0

            drift_extra_deadline = self.drift_extra_deadline
            if last_pulse < drift_extra_deadline:
                ns_drift_extra = self.ns_drift_extra
                we_drift_extra = self.we_drift_extra
                extra_delta = min(now, drift_extra_deadline) - last_pulse

                cur_ns_duty += ns_drift_extra * extra_delta
                cur_we_duty += we_drift_extra * extra_delta

            target_pulse = self.target_pulse
            if cur_ns_duty >= min_pulse:
                ns_pulse = min(cur_ns_duty, cur_period)
            elif cur_ns_duty <= -min_pulse:
                ns_pulse = max(cur_ns_duty, -cur_period)
            else:
                ns_pulse = 0

            if cur_we_duty >= min_pulse:
                we_pulse = min(cur_we_duty, cur_period)
            elif cur_we_duty <= -min_pulse:
                we_pulse = max(cur_we_duty, -cur_period)
            else:
                we_pulse = 0

            if ns_pulse and ns_dir and direct_ns_pulse and (ns_pulse < 0) != (ns_dir < 0):
                # Direction switch - resist it
                total_ns_ignored = self.total_ns_ignored
                if (ns_pulse < 0) != (total_ns_ignored < 0):
                    self.total_ns_ignored = total_ns_ignored = 0
                switch_potential = max(
                    abs(ns_pulse + total_ns_ignored),
                    abs(cur_ns_duty + total_ns_ignored),
                )
                if switch_potential < self._eff_dec_switch_resistence:
                    ign = min_directed(direct_ns_pulse, ns_pulse)

                    if (ns_pulse < 0) == (ns_drift < 0):
                        # Pulse and drift move together, next time, do it
                        ns_dir = -1 if ns_drift < 0 else 1

                    self.ns_ignored += ign
                    self.total_ns_ignored += ign
                    cur_ns_duty -= ign
                    ns_pulse -= ign
                else:
                    ns_dir = -1 if ns_pulse < 0 else 1
                    self.total_ns_ignored = self.ns_ignored
            elif ns_pulse:
                ns_dir = -1 if ns_pulse < 0 else 1

            if we_pulse and we_dir and direct_we_pulse and (we_pulse < 0) != (we_dir < 0):
                # Direction switch - resist it
                total_we_ignored = self.total_we_ignored
                if (ns_pulse < 0) != (total_we_ignored < 0):
                    self.total_we_ignored = total_we_ignored = 0
                switch_potential = max(
                    abs(we_pulse + total_we_ignored),
                    abs(cur_we_duty + total_we_ignored),
                )
                if switch_potential < self._eff_ra_switch_resistence:
                    ign = min_directed(direct_we_pulse, we_pulse)

                    if (we_pulse < 0) == (we_drift < 0):
                        # Pulse and drift move together, next time, do it
                        we_dir = -1 if we_drift < 0 else 1

                    self.we_ignored += ign
                    self.total_we_ignored += ign
                    cur_we_duty -= ign
                    we_pulse -= ign
                else:
                    we_dir = -1 if we_pulse < 0 else 1
                    self.total_we_ignored = self.we_ignored
            elif we_pulse:
                we_dir = -1 if we_pulse < 0 else 1

            rate_we = delta * self.gear_rate_we
            last_pulse = now
            longest_pulse = max(abs(we_pulse), abs(ns_pulse))
            if we_pulse or ns_pulse:
                if longest_pulse > 2 * target_pulse:
                    cur_period *= 0.7
                elif longest_pulse < 0.5 * target_pulse:
                    cur_period *= 1.4
                cur_period = max(min(cur_period, self.max_pulse), self.min_pulse)

                # No need to wake up before the pulse is done
                ins_pulse = int(ns_pulse * 1000)
                iwe_pulse = int(we_pulse * 1000)
                fns_pulse = ins_pulse / 1000.0
                fwe_pulse = iwe_pulse / 1000.0
                self.st4.pulseGuide(ins_pulse, iwe_pulse)
                cur_ns_duty -= fns_pulse
                cur_we_duty -= fwe_pulse
                pulse_deadline = time.time() + longest_pulse

                self.add_gear_state(fns_pulse, fwe_pulse + rate_we)
            else:
                self.add_gear_state(0, rate_we)

            self.pulse_period = cur_period
            sleep_period = max(cur_period, longest_pulse, 0.05)

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

    def join(self):
        if self.runner_thread is not None:
            self.runner_thread.join()
            self.runner_thread = None
            self._stop = False

