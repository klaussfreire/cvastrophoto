

class ConfigStore(object):

    def __init__(self):
        self._values = None
        self._values_modified = False

    @property
    def values(self):
        if self._values is None:
            self._values = self.get_values()
        return self._values

    def list_sections(self):
        raise NotImplementedError()

    def get_section(self, name):
        raise NotImplementedError()

    def get_values(self):
        raise NotImplementedError()

    def save_values(self, values):
        raise NotImplementedError()

    def get_file(self, name):
        raise NotImplementedError()

    def get_value(self, name, deflt=None, typ=None):
        v = self.values.get(name, deflt)
        if v is not deflt and typ is not None:
            v = typ(v)
        return v

    def set_value(self, name, value):
        self.values[name] = value
        self._values_modified = True

    def del_value(self, name):
        rv = self.values.pop(name, None)
        if rv is not None:
            self._values_modified = True

    def flush(self):
        if self._values_modified:
            self.save_values(self.values)
            self._values_modified = False
