from . import base


class CCDProfile(base.BaseProfile):

    def get_focusing_profile(self, focuser_name, cfw_name, configuration='default'):
        """CCD focusing profile contains base focusing position for this equipment combination"""
        return self.equipment_subprofile('focuser', "%s,%s/%s" % (focuser_name, cfw_name, configuration), CCDFocuserProfile)

    def list_focusing_configurations(self):
        return self.equipment_subprofile('focuser').list_subprofiles()


class CCDFocuserProfile(base.BaseProfile):

    base_focus_pos = base.ProfileProperty('base_focus_pos', None, float)
    base_focus_filter_name = base.ProfileProperty('base_focus_filter_filter', None)


base.RootProfile.register_device_type('ccd', CCDProfile)
