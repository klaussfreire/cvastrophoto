# -*- coding: utf-8 -*-
import threading
import time

class GuiderController(object):

    target_pulse = 0.15
    min_pulse = 0.05
    max_pulse = 1.0

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
        self.ns_drift_extra = 0
        self.we_drift_extra = 0
        self.drift_extra_deadline = 0
        self.paused = False

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

    def wait_pulse(self, timeout=None):
        """ Wait until the current pulse has been fully executed """
        return self.pulse_event.wait(timeout)

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
        self.drift_extra_deadline = time.time() + exec_s
        self.spread_pulse_event.clear()
        self.wake.set()

    def wait_spread_pulse(self, timeout=None):
        self.spread_pulse_event.wait(timeout)

    def run(self):
        cur_ns_duty = 0
        cur_we_duty = 0
        cur_period = 0.5
        sleep_period = cur_period

        last_pulse = time.time()

        doing_pulse = False

        while not self._stop:
            self.wake.wait(sleep_period)
            self.wake.clear()
            if self._stop:
                break
            elif self.paused:
                continue

            now = time.time()
            delta = now - last_pulse

            ns_drift, we_drift = self.eff_drift

            direct_ns_pulse = self.ns_pulse
            direct_we_pulse = self.we_pulse
            cur_ns_duty += ns_drift * delta + direct_ns_pulse
            cur_we_duty += we_drift * delta + direct_we_pulse
            self.ns_pulse -= direct_ns_pulse
            self.we_pulse -= direct_we_pulse
            if direct_ns_pulse or direct_we_pulse:
                doing_pulse = True

            drift_extra_deadline = self.drift_extra_deadline
            if last_pulse < drift_extra_deadline:
                ns_drift_extra = self.ns_drift_extra
                we_drift_extra = self.we_drift_extra
                extra_delta = min(now, drift_extra_deadline) - last_pulse

                cur_ns_duty += ns_drift_extra * extra_delta
                cur_we_duty += we_drift_extra * extra_delta

            min_pulse = self.min_pulse
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

            longest_pulse = max(abs(we_pulse), abs(ns_pulse))
            if we_pulse or ns_pulse:
                if longest_pulse > 2 * target_pulse:
                    cur_period *= 0.7
                elif longest_pulse < 0.5 * target_pulse:
                    cur_period *= 1.4
                cur_period = max(min(cur_period, self.max_pulse), self.min_pulse)

                ins_pulse = int(ns_pulse * 1000)
                iwe_pulse = int(we_pulse * 1000)
                self.st4.pulseGuide(ins_pulse, iwe_pulse)
                cur_ns_duty -= ins_pulse / 1000.0
                cur_we_duty -= iwe_pulse / 1000.0

                if doing_pulse and abs(cur_ns_duty) < min_pulse and abs(cur_we_duty) < min_pulse:
                    self.pulse_event.set()

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

