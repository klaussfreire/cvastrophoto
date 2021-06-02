from __future__ import absolute_import

import os.path
import errno
import json

from .base import ConfigStore


class FilesStore(ConfigStore):

    VALUES_FILE = 'values.json'

    def __init__(self, base_path):
        try:
            os.makedirs(base_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.base_path = base_path
        super(FilesStore, self).__init__()

    def get_section(self, name):
        return FilesStore(os.path.join(self.base_path, name))

    def get_values(self):
        if not self.has_file(self.VALUES_FILE):
            return {}
        with self.get_file(self.VALUES_FILE) as f:
            return json.load(f)

    def save_values(self, values):
        with self.get_file(self.VALUES_FILE, 'w') as f:
            return json.dump(values, f, indent=2, separators=(',', ': '))

    def has_file(self, name):
        return os.path.isfile(os.path.join(self.base_path, name))

    def get_file(self, name, mode='r'):
        return open(os.path.join(self.base_path, name), mode)
