# -*- coding: utf-8 -*-
from __future__ import absolute_import

import time
import multiprocessing.pool
import itertools
import operator

from .base import BaseWizard
from ..rops.measures import focus
from ..library import darks

import logging

logger = logging.getLogger(__name__)

class SubSelectionWizard(BaseWizard):

    def __init__(self,
            selection_class=focus.FocusMeasureRop,
            selection_kwargs=dict(quick=True),
            best_ratio=0.7,
            pool=None):

        if pool is None:
            self.pool = pool = multiprocessing.pool.ThreadPool()

        self.selection_class = selection_class
        self.selection_kwargs = selection_kwargs
        self.best_ratio = best_ratio
        self.dark_library = None

    def load_set(self, subs, dark_library=None):
        subs = iter(subs)

        if dark_library is None:
            dark_library = darks.DarkLibrary(default_pool=self.pool)

        self.dark_library = dark_library
        self.selection = self.selection_class(next(subs), **self.selection_kwargs)

    def select(self, subs, nsubs=None):
        if self.pool is not None:
            imap = self.pool.imap_unordered
        else:
            imap = itertools.imap

        selection = self.selection
        dark_library = self.dark_library

        def rank(item):
            try:
                i, sub = item

                sub.remove_bias()

                if dark_library is not None:
                    dark = dark_library.get_master(dark_library.classify_frame(sub.name), raw=sub)
                    if dark is not None:
                        sub.denoise([dark], quick=True, master_bias=None, entropy_weighted=False)

                rank = selection.measure_scalar(sub.rimg.raw_image)
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
        ranked = []
        for i, sub, sub_rank in imap(rank, enumerate(subs)):
            ranked.append((sub_rank, i, sub))
            if time.time() > last_status + 10:
                logger.info("Ranked %d/%s subs", len(ranked), nsubs)
                last_status = time.time()

        logger.info("Ranked %d/%s subs", len(ranked), nsubs)

        ranked.sort(key=operator.itemgetter(0))
        nselected = int(max(1, len(ranked) * self.best_ratio))

        logger.info("Selection rank threshold: %s", ranked[-nselected][0])

        selected = ranked[-nselected:]
        ranks = list(map(operator.itemgetter(0), selected))

        logger.info("Rank stats: avg=%s best=%s worst=%s", sum(ranks) / len(ranks), max(ranks), min(ranks))

        # Yield in original order
        selected.sort(key=operator.itemgetter(1))
        for sub_rank, i, sub in selected:
            yield i, sub
