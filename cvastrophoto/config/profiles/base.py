import logging


logger = logging.getLogger(__name__)


class BaseProfile(object):

    def __init__(self, store, name, autoflush=True):
        self.store = store
        self.name = name
        self.autoflush = autoflush

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.name)

    def __str__(self):
        return self.name

    def list_subprofiles(self, kind=None):
        store = self.store
        if kind is not None:
            store = store.get_section(kind)
        return store.list_sections()

    def equipment_subprofile(self, equipment_class, equipment_name, klass=None):
        if klass is None:
            klass = RootProfile.profile_class_for(equipment_class)
            if klass is None:
                klass = type(self)
        return klass(
            self.store.get_section(equipment_class).get_section(equipment_name),
            '/'.join([self.name, equipment_class, equipment_name])
        )

    def get_value(self, name, deflt=None, typ=None):
        return self.store.get_value(name, deflt, typ)

    def set_value(self, name, value):
        self.store.set_value(name, value)

    def del_value(self, name):
        self.store.del_value(name)

    def flush(self):
        self.store.flush()

    def __del__(self):
        if self.autoflush:
            self.flush()


class RootProfile(BaseProfile):

    device_type_map = {}

    def get_device_profile(self, device_type, device_name, klass=None):
        if klass is None:
            klass = self.device_type_map[device_type]
        return self.equipment_subprofile(device_type, device_name, klass)

    def get_mount_profile(self, device_name):
        return self.get_device_profile('mount', device_name)

    def get_telescope_profile(self, device_name, fl, ap):
        return self.get_device_profile('telescope', '%s,%s,%s' % (device_name, fl, ap))

    @classmethod
    def register_device_type(cls, device_type, klass):
        cls.device_type_map[device_type] = klass

    @classmethod
    def profile_class_for(cls, device_type):
        return cls.device_type_map.get(device_type)


class ProfileProperty(object):

    def __init__(self, name, deflt=None, typ=None):
        self.name = name
        self.deflt = deflt
        self.typ = typ

    def __get__(self, obj, typ=None):
        if obj is None:
            return self

        return obj.get_value(self.name, self.deflt, self.typ)

    def __set__(self, obj, value):
        logger.debug("Profile %r set %s=%r", obj, self.name, value)
        obj.set_value(self.name, value)

    def __delete__(self, obj):
        logger.debug("Profile %r delete %s=%r", obj, self.name)
        obj.del_value(self.name)
