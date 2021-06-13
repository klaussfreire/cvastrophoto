# -*- coding: utf-8 -*-
from __future__ import absolute_import

import time
import multiprocessing.pool
import itertools
import operator

from .base import BaseWizard
from ..rops.measures import focus
from ..rops.denoise import median
from ..library import darks

import logging

logger = logging.getLogger(__name__)

class SubSelectionWizard(BaseWizard):

    def __init__(self,
            selection_class=focus.FocusMeasureRop,
            selection_kwargs=dict(quick=True),
            cleanup_class=median.MedianFilterRop,
            cleanup_kwargs={},
            best_ratio=0.7,
            pool=None):

        if pool is None:
            self.pool = pool = multiprocessing.pool.ThreadPool()

        self.selection_class = selection_class
        self.selection_kwargs = selection_kwargs
        self.cleanup_class = cleanup_class
        self.cleanup_kwargs = cleanup_kwargs
        self.best_ratio = best_ratio
        self.dark_library = None

    def load_set(self, subs, dark_library=None):
        subs = iter(subs)

        if dark_library is None:
            dark_library = darks.DarkLibrary(default_pool=self.pool)

        sub = next(subs)
        self.dark_library = dark_library
        self.selection = self.selection_class(sub, **self.selection_kwargs)

        if self.cleanup_class is not None:
            self.cleanup = self.cleanup_class(sub, **self.cleanup_kwargs)
        else:
            self.cleanup = None

    def _rank_subs(self, subs, nsubs=None):
        if self.pool is not None:
            imap = self.pool.imap_unordered
        else:
            imap = itertools.imap

        selection = self.selection
        cleanup = self.cleanup
        dark_library = self.dark_library

        # Initialize rop lazy props that shouldn't be initialized inside the thread pool
        selection.init_pattern()
        cleanup.init_pattern()

        def rank(item):
            try:
                i, sub = item

                do_remove_bias = True
                if dark_library is not None:
                    dark = dark_library.get_master(dark_library.classify_frame(sub.name), raw=sub)
                    if dark is not None:
                        sub.denoise([dark], quick=True, master_bias=None, entropy_weighted=False)
                        do_remove_bias = False
                if do_remove_bias:
                    sub.remove_bias()

                raw_image = sub.rimg.raw_image

                if cleanup is not None:
                    raw_image = cleanup.correct(raw_image, pool=None)

                rank = selection.measure_scalar(raw_image)
                sub.close()
                return i, sub, rank
            except Exception:
                logger.exception("Error ranking sub")

        if nsubs is None:
            try:
                nsubs = len(subs)
            except Exception:
                nsubs = None
        if nsubs is None:
            nsubs = '?'

        last_status = time.time()
        ranked = 0
        for i, sub, sub_rank in imap(rank, enumerate(subs)):
            yield i, sub, sub_rank
            ranked += 1
            if time.time() > last_status + 10:
                logger.info("Ranked %d/%s subs", ranked, nsubs)
                last_status = time.time()

        logger.info("Ranked %d/%s subs", ranked, nsubs)

    def rank(self, subs, nsubs=None):
        ranked = []
        for i, sub, sub_rank in self._rank_subs(subs, nsubs=nsubs):
            ranked.append((sub_rank, i, sub))

        ranked.sort(key=operator.itemgetter(0))

        ranks = list(map(operator.itemgetter(0), ranked))
        logger.info("Rank stats: avg=%s best=%s worst=%s", sum(ranks) / len(ranks), max(ranks), min(ranks))

        return ranked

    def select(self, subs, nsubs=None):
        ranked = self.rank(subs, nsubs=nsubs)

        ranked.sort(key=operator.itemgetter(0))
        nselected = int(max(1, len(ranked) * self.best_ratio))

        logger.info("Selection rank threshold: %s", ranked[-nselected][0])

        selected = ranked[-nselected:]
        ranks = list(map(operator.itemgetter(0), selected))

        logger.info("Rank stats (selection): avg=%s best=%s worst=%s", sum(ranks) / len(ranks), max(ranks), min(ranks))

        # Yield in original order
        selected.sort(key=operator.itemgetter(1))
        for sub_rank, i, sub in selected:
            yield i, sub
