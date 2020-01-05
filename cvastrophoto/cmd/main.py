# -*- coding: utf-8 -*-
import logging

from .commands import ALL_COMMANDS

def main(opts):
    level = logging.INFO
    if opts.verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    ALL_COMMANDS[opts.command].main(opts)
