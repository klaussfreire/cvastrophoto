# -*- coding: utf-8 -*-
import threading
import time

class GuiderController(object):

    target_pulse = 0.15
    min_pulse = 0.05
    max_pulse = 1.0

    pulse_period = 0.5

    dec_switch_resistence = 0.5
    ra_switch_resistence = 0.25

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
        self.ns_ignored = 0
        self.we_ignored = 0
        self.ns_drift_extra = 0
        self.we_drift_extra = 0
        self.drift_extra_deadline = 0
        self.drift_extra_time = 0
        self.paused = False
        self.paused_drift = False

    @property
    def eff_drift(self):
        return (self.ns_drift, self.we_drift)

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

            if self.paused_drift:
                drift_delta = 0
            else:
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
            if cur_ns_duty > min_pulse:
                ns_pulse = min(cur_ns_duty, cur_period)
            elif cur_ns_duty < -min_pulse:
                ns_pulse = max(cur_ns_duty, -cur_period)
            else:
                ns_pulse = 0

            if cur_we_duty > min_pulse:
                we_pulse = min(cur_we_duty, cur_period)
            elif cur_we_duty < -min_pulse:
                we_pulse = max(cur_we_duty, -cur_period)
            else:
                we_pulse = 0

            if ns_pulse and ns_dir and (ns_pulse < 0) != (ns_dir < 0):
                # Direction switch - resist it
                if abs(ns_pulse) < self.dec_switch_resistence:
                    self.ns_ignored += ns_pulse
                    ns_pulse = 0
                    if (ns_pulse < 0) == (ns_drift < 0):
                        # Pulse and drift move together, next time, do it
                        ns_dir = -1 if ns_drift < 0 else 1
                else:
                    ns_dir = -1 if ns_pulse < 0 else 1

            if we_pulse and we_dir and (we_pulse < 0) != (we_dir < 0):
                # Direction switch - resist it
                if abs(we_pulse) < self.ra_switch_resistence:
                    self.we_ignored += we_pulse
                    we_pulse = 0
                    if (we_pulse < 0) == (we_drift < 0):
                        # Pulse and drift move together, next time, do it
                        we_dir = -1 if we_drift < 0 else 1
                else:
                    we_dir = -1 if we_pulse < 0 else 1

            longest_pulse = max(abs(we_pulse), abs(ns_pulse))
            if we_pulse or ns_pulse:
                if longest_pulse > 2 * target_pulse:
                    cur_period *= 0.7
                elif longest_pulse < 0.5 * target_pulse:
                    cur_period *= 1.4
                cur_period = max(min(cur_period, self.max_pulse), self.min_pulse)

                # No need to wake up before the pulse is done
                pulse_deadline = now + longest_pulse

                ins_pulse = int(ns_pulse * 1000)
                iwe_pulse = int(we_pulse * 1000)
                self.st4.pulseGuide(ins_pulse, iwe_pulse)
                cur_ns_duty -= ins_pulse / 1000.0
                cur_we_duty -= iwe_pulse / 1000.0

            self.pulse_period = cur_period
            last_pulse = now
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

