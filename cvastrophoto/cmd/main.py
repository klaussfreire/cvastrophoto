# -*- coding: utf-8 -*-
from __future__ import division

import logging
import multiprocessing.pool

from .commands import ALL_COMMANDS

from cvastrophoto.util.nullpool import NullPool

def main(opts):
    level = logging.INFO
    if opts.verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    if opts.parallel is None:
        pool = multiprocessing.pool.ThreadPool(multiprocessing.cpu_count())
    elif opts.parallel:
        pool = multiprocessing.pool.ThreadPool(opts.parallel)
    else:
        pool = NullPool()

    if opts.parallel_input is None:
        input_pool = multiprocessing.pool.ThreadPool(min(4, multiprocessing.cpu_count() // 2))
    elif opts.parallel_input:
        input_pool = multiprocessing.pool.ThreadPool(opts.parallel_input)
    else:
        input_pool = NullPool()
    opts.input_pool = input_pool

    ALL_COMMANDS[opts.command].main(opts, pool)