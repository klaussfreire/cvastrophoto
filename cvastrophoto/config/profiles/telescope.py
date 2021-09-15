from . import base


class TelescopeProfile(base.BaseProfile):

    def get_ccd_profile(self, device_name):
        return self.equipment_subprofile('ccd', device_name)

    def get_focusing_profile(self, focuser_name, cfw_name):
        """Telescope focusing profile contains focus offsets for any applicable filters"""
        return self.equipment_subprofile('focuser', "%s,%s" % (focuser_name, cfw_name), FilterFocuserProfile)


class FilterFocuserProfile(base.BaseProfile):

    def get_filter_offset(self, filter_name, from_filter=None):
        offset = self.get_value("filter_offset[%s]" % (filter_name,), None, float)
        if offset is not None and from_filter is not None:
            ref_offset = self.get_filter_offset(from_filter)
            if ref_offset is None:
                return None
            else:
                offset -= ref_offset
        return offset

    def set_filter_offset(self, filter_name, offset, from_filter=None):
        offset = float(offset)
        if from_filter is not None:
            ref_offset = self.get_filter_offset(from_filter)
            if ref_offset is not None:
                offset += ref_offset
        return self.set_value("filter_offset[%s]" % (filter_name,), offset)

    def del_filter_offset(self, filter_name):
        return self.del_value("filter_offset[%s]" % (filter_name,),)

    @property
    def filter_offsets(self):
        return {
            k[14:-1]: v
            for k, v in self.values.items()
            if k.startswith("filter_offset[") and k.endswith("]")
        }

    def reset_filter_offsets(self):
        for filter_name in self.filter_offsets:
            self.del_filter_offset(filter_name)


base.RootProfile.register_device_type('telescope', TelescopeProfile)
