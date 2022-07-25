# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os.path
import collections
import logging
import itertools
import hashlib
import uuid
import multiprocessing.pool

from six import iteritems, itervalues

import cvastrophoto.image


logger = logging.getLogger(__name__)


class CollectionBase(object):

    default_base_path = None

    def __init__(self, base_path=None):
        if base_path is None:
            if self.default_base_path is None:
                raise ValueError("Need a base_path")
            base_path = os.path.expanduser(self.default_base_path)
        else:
            base_path = os.path.expanduser(base_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        self.base_path = base_path

    def get_path_for(self, key, subcollection=None):
        parts = [self.base_path] + list(map(str, key))
        if subcollection:
            parts.extend(map(str, subcollection))
        return os.path.join(*parts)

    def get_filename_for(self, path, img):
        name = getattr(img, "name", None)
        if name is not None and os.path.exists(name):
            # Compute hash of file contents
            csum = hashlib.sha1()
            buf = True
            bufsize = 1 << 20
            with open(name, "rb") as f:
                while buf:
                    buf = f.read(bufsize)
                    if buf:
                        csum.update(buf)
            hashpart = csum.hexdigest()
            basename, ext = os.path.splitext(name)
            if ext == '.gz':
                basename, ext = os.path.splitext(basename)
                ext += '.gz'
            basename = os.path.basename(basename)
            if ext:
                ext = ext[1:]
        else:
            # Generate a random filename
            hashpart = uuid.uuid4()
            basename = 'nn'
            ext = getattr(img, 'preferred_ext', None)
        return '%s_%s.%s' % (basename, hashpart, ext)

    def _walk(self, path, recurse):
        raise NotImplementedError()

    def _subdirs(self, path, recurse):
        raise NotImplementedError()

    def _contains(self, path, filename):
        raise NotImplementedError()

    def _add(self, path, filename, img):
        raise NotImplementedError()

    def contains(self, key, subcollection=None, img=None, filename=None):
        path = self.get_path_for(key, subcollection)
        if filename is None and img is not None:
            filename = self.get_filename_for(path, img)
        return self._contains(path, filename)

    def list_paths(self, key, subcollection=None, recurse=False):
        path = self.get_path_for(key, subcollection)
        if os.path.exists(path) and os.path.isdir(path):
            for dirname, filename in self._walk(path, recurse):
                yield os.path.join(dirname, filename)

    def list_subcollections(self, key, subcollection=None, recurse=False):
        path = self.get_path_for(key)
        if os.path.exists(path) and os.path.isdir(path):
            for dirname, subdirname in self._subdirs(path, recurse):
                yield self._path_to_key(os.path.join(path, dirname, subdirname))

    def _path_to_key(self, path):
        base_path = self.base_path
        if not base_path.endswith(os.path.sep):
            base_path += os.path.sep

        assert path.startswith(base_path)
        path = path[len(base_path):]
        path = os.path.normpath(path)
        return path.split(os.path.sep)

    def list(self, key, subcollection=None, recurse=False, **open_kw):
        for path in self.list_paths(key, subcollection, recurse):
            yield cvastrophoto.image.Image.open(path, **open_kw)

    def add(self, key, subcollection, img):
        path = self.get_path_for(key, subcollection)
        filename = self.get_filename_for(path, img)

        if not self.contains(key, subcollection, filename):
            logger.info("Adding %r to collection at path %r", img.name, os.path.join(path, filename))
            self._add(path, filename, img)


class CollectionSortingBase(object):

    classification_semantic = None

    def classify_frame(self, img_path):
        raise NotImplementedError

    def reclassify_frame(self, img_path, **semantic_changes):
        cur = self.classify_frame(img_path)
        cur_sem = self.semantic(cur)
        new_sem = cur_sem.copy()
        new_sem.update(semantic_changes)
        if new_sem != cur_sem:
            new = self.from_semantic(new_sem)
            if new != cur:
                self._reclassify_frame(img_path, new, cur)

    def semantic(self, classification):
        return dict(zip(self.classification_semantic, classification))

    def from_semantic(self, semantic, deflt='NA'):
        return tuple([semantic.get(k, deflt) for k in self.classification_semantic])
