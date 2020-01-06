# -*- coding: utf-8 -*-
import logging
import multiprocessing.pool

from .commands import ALL_COMMANDS

def main(opts):
    level = logging.INFO
    if opts.verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    if opts.parallel:
        pool = multiprocessing.pool.ThreadPool(opts.parallel)
    else:
        pool = None

    ALL_COMMANDS[opts.command].main(opts, pool)
