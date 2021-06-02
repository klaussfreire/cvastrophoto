from . import base


class GuideProfile(base.BaseProfile):

    backlash_dec = base.ProfileProperty('backlash_dec', None, float)
    backlash_ra = base.ProfileProperty('backlash_ra', None, float)


class GuideScopeProfile(base.BaseProfile):

    calibration_pulse_s_ra = base.ProfileProperty('calibration_pulse_s_ra', None, float)
    calibration_pulse_s_dec = base.ProfileProperty('calibration_pulse_s_dec', None, float)
    clear_backlash_pulse_ra = base.ProfileProperty('clear_backlash_pulse_ra', None, float)
    clear_backlash_pulse_dec = base.ProfileProperty('clear_backlash_pulse_dec', None, float)


class MountProfile(base.BaseProfile):

    def get_guide_profile(self, guide_speed=None):
        if guide_speed is None:
            name = 'dflt'
        else:
            name = '%.2f' % guide_speed
        return self.equipment_subprofile('guide', name, GuideProfile)

    def get_guidescope_profile(self, guide_speed=None, guide_fl=None, guide_ccd_name=None):
        name = 's=%s_fl=%s_ccd=%s' % (
            guide_speed or 'deflt',
            guide_fl or 'unk',
            guide_ccd_name or 'unk',
        )
        return self.equipment_subprofile('guidescope', name, GuideScopeProfile)


base.RootProfile.register_device_type('mount', MountProfile)
