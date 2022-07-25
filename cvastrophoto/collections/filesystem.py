from __future__ import absolute_import

import os.path
import shutil

from .base import CollectionBase


class FilesystemCollection(CollectionBase):

    enable_hardlinks = True

    def _walk(self, path, recurse):
        for dirpath, dirnames, filenames in os.walk(path):
            if not recurse:
                del dirnames[:]
            for filename in filenames:
                yield dirpath, filename

    def _subdirs(self, path, recurse):
        for dirpath, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                if dirname == '.':
                    continue
                yield dirpath, dirname
            if not recurse:
                del dirnames[:]

    def _contains(self, path, filename):
        if filename is None:
            return os.path.exists(path) and os.path.isdir(path)
        else:
            return os.path.exists(os.path.join(path, filename))

    def _add(self, path, filename, img):
        name = getattr(img, "name", None)
        if not os.path.exists(path):
            os.makedirs(path)
        if name is not None and os.path.exists(name):
            # Compute hash of file contents
            st1 = os.stat(name)
            st2 = os.stat(path)
            dstpath = os.path.join(path, filename)
            if self.enable_hardlinks and st1.st_dev == st2.st_dev:
                # Can hardlink
                os.link(name, dstpath)
            else:
                try:
                    shutil.copyfile(name, dstpath)
                except:
                    if os.path.exists(dstpath):
                        os.unlink(dstpath)
                    raise
        else:
            raise NotImplementedError("Can't add memory-only images to collection")
