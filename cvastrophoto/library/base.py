# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os.path
import collections
import imageio
import logging
import itertools
import multiprocessing.pool

import cvastrophoto.image


logger = logging.getLogger(__name__)


class LibraryBase(object):

    save_meta = dict(compress=6)
    default_base_path = None
    min_subs = 1

    def __init__(self, base_path=None, default_pool=None, cache_size=4):
        if base_path is None:
            if self.default_base_path is None:
                raise ValueError("Need a base_path")
            base_path = os.path.expanduser(self.default_base_path)
        else:
            base_path = os.path.expanduser(base_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if default_pool is None:
            default_pool = multiprocessing.pool.ThreadPool()

        self.base_path = base_path
        self.pool = default_pool
        self.cache = {}
        self.cache_queue = []
        self.cache_size = cache_size

    def classify_frame(self, img_path):
        raise NotImplementedError

    def vary(self, key, for_build=False):
        return [key]

    def build_master(self, key, frames):
        raise NotImplementedError

    def _cache_add(self, key, value):
        ovalue = self.cache.setdefault(key, value)
        if ovalue is value:
            self.cache[key] = value
            self.cache_queue.append(key)
            if len(self.cache_queue) > self.cache_size:
                for popkey in self.cache_queue[:-self.cache_size]:
                    self.cache.pop(popkey, None)
                del self.cache_queue[:-self.cache_size]
        return ovalue

    def get_master(self, key, vary=True, raw=None):
        if vary:
            keys = self.vary(key)
        else:
            keys = [key]

        for key in keys:
            img = self._get_master(key, raw=raw)
            if img is not None:
                return img

    def contains(self, key, vary=True):
        if vary:
            keys = self.vary(key)
        else:
            keys = [key]

        for key in keys:
            if self._contains(key):
                return True
        return False

    def _get_master(self, key, raw=None):
        img = self.cache.get(key)
        if img is None:
            path = self.get_path_for(key)
            if os.path.exists(path):
                img = self._cache_add(key, self.open_impl(path))
                if raw is not None and hasattr(img, 'set_raw_template'):
                    img.set_raw_template(raw)
        else:
            self.cache_queue.remove(key)
            self.cache_queue.append(key)
        return img

    def _contains(self, key):
        if key in self.cache:
            return True

        path = self.get_path_for(key)
        return os.path.exists(path)

    def add(self, key, frames):
        img = self.build_master(key, frames)

        path = self.get_path_for(key)
        logger.info("Adding %r to library at path %r", key, path)

        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != 17:
                    raise

        with imageio.get_writer(path, mode='i', software='cvastrophoto') as writer:
            writer.append_data(img, self.save_meta)

        logger.info("Added %r to library", key)

    def open_impl(self, img_path):
        return cvastrophoto.image.Image.open(img_path, default_pool=self.pool, autoscale=False, linear=True)

    def get_path_for(self, key):
        parts = [self.base_path] + map(str, key)
        parts[-1] = 'temp_%s.tiff' % (parts[-1],)
        return os.path.join(*parts)

    def build(self, img_paths, refresh=False):
        img_sets = collections.defaultdict(set)

        def classify(img_path):
            try:
                return img_path, self.classify_frame(img_path)
            except Exception:
                logger.exception("Error classifying %r", img_path)

        if self.pool is not None:
            map_ = self.pool.imap_unordered
        else:
            map_ = itertools.imap

        for img_path, key in map_(classify, img_paths):
            keys = self.vary(key, for_build=True)
            logger.info("Classified %r into %r", img_path, keys[0])
            for key in keys:
                img_sets[key].add(img_path)

        if not img_sets:
            return

        def build(entry):
            try:
                key, img_paths = entry
                self.add(key, img_paths)
            except Exception:
                logger.exception("Error adding master for %r", key)
                return False
            else:
                return True

        up2date = 0
        buildable = 0
        to_build = []
        min_subs = self.min_subs
        min_subs = min(min_subs, max(map(len, img_sets.itervalues())))
        for key, img_paths in img_sets.iteritems():
            if len(img_paths) >= min_subs:
                buildable += 1
                if refresh or not self.contains(key, vary=False):
                    to_build.append((key, img_paths))
                else:
                    up2date += 1

        failed = 0
        for i, success in enumerate(map_(build, to_build)):
            if not success:
                failed += 1
            logger.info("Built %d/%d masters", i+1, len(to_build))

        if failed:
            logger.info("Failed %d/%d masters", failed, len(to_build))
        if up2date:
            logger.info("%d/%d masters already up to date", up2date, buildable)

    def build_recursive(self, basepath, refresh=False, **kw):
        self.build(self.list_recursive(basepath, **kw), refresh=refresh)

    def list_recursive(self, basepath, filter_fn=None):
        if filter_fn is None:
            filter_fn = self.default_filter
        for dirpath, dirnames, filenames in os.walk(basepath):
            if filter_fn is not None:
                dirnames[:] = [
                    dirname for dirname in dirnames
                    if filter_fn(dirpath, dirname, None)
                ]

            for filename in filenames:
                fullpath = os.path.join(dirpath, filename)
                if filter_fn is not None and not filter_fn(dirpath, None, filename):
                    continue
                elif os.path.isfile(fullpath) and cvastrophoto.image.Image.supports(fullpath):
                    yield os.path.realpath(fullpath)
