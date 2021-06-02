from . import base


class TelescopeProfile(base.BaseProfile):
    pass


base.RootProfile.register_device_type('telescope', TelescopeProfile)
