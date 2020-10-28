
class BacklashCompensation(object):

    backlash_aggressiveness = 0.5

    # Ratios relative to exposure length
    max_backlash_pulse_ratio = 1.0

    # Relative to deflection pulse length
    initial_backlash_pulse_ratio = 1.0
    backlash_stop_threshold = 0.75
    backlash_sync_threshold = 0.25

    # Growth rate between successive backlash pulses
    backlash_ratio_factor = 2.0

    # Backlash calibration shrink rate when early stop is triggered
    shrink_rate = 1.0

    def __init__(self, calibration, backlash_method, sync_method):
        self.calibration = calibration
        self._backlash_compensation = backlash_method
        self._sync_method = sync_method
        self.reset(0)

    @classmethod
    def for_controller_ra(cls, calibration, controller):
        return cls(calibration, controller.backlash_compensation_ra, controller.sync_gear_state_ra)

    @classmethod
    def for_controller_dec(cls, calibration, controller):
        return cls(calibration, controller.backlash_compensation_dec, controller.sync_gear_state_dec)

    def reset(self, pulse=None):
        self.backlash_ratio = self.initial_backlash_pulse_ratio
        if pulse is not None:
            self.prev_backlash_pulse = pulse
        self.prev_max_backlash_pulse = 0

    def compute_pulse(self, imm, max_pulse):
        backlash_pulse = self._backlash_compensation(imm)
        if backlash_pulse:
            if (self.prev_backlash_pulse < 0) != (backlash_pulse < 0):
                self.backlash_ratio = self.initial_backlash_pulse_ratio

                if self.prev_backlash_pulse:
                    # Direction switch without a reset in-between requires a gear state sync
                    # Clearly, backlash was cleared in-between so we have to record it
                    self._sync_method(self.prev_backlash_pulse, self.shrink_rate)
                    backlash_pulse = self._backlash_compensation(imm)

                self.reset(backlash_pulse)

        if backlash_pulse:
            max_backlash_pulse = min(max_pulse, abs(imm))
            prev_max_backlash_pulse = self.prev_max_backlash_pulse
            if max_backlash_pulse < prev_max_backlash_pulse * self.backlash_stop_threshold:
                # Significant move along the desired direction is a sign that backlash was finally cleared
                # Record the fact in the gear state and let the controller adjust max gear state accordingly
                backlash_pulse = 0
                self.reset(0)
                if max_backlash_pulse < prev_max_backlash_pulse * self.backlash_sync_threshold:
                    self._sync_method(imm, self.shrink_rate)
            else:
                self.prev_max_backlash_pulse = max_backlash_pulse
                max_backlash_pulse = min(
                    self.max_backlash_pulse_ratio * self.calibration.guide_exposure,
                    max_backlash_pulse * self.backlash_ratio)
                backlash_pulse = backlash_pulse * self.backlash_aggressiveness
                backlash_pulse = max(min(backlash_pulse, max_backlash_pulse), -max_backlash_pulse)

                self.backlash_ratio *= self.backlash_ratio_factor
        else:
            self.reset(0)

        return backlash_pulse
