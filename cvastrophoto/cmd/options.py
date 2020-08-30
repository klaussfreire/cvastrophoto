# -*- coding: utf-8 -*-
import argparse
from six import itervalues

from .commands import ALL_COMMANDS

def parse():
    ap = argparse.ArgumentParser(
        prog='cvastrophoto',
        description='Astrophotography workflow library',
    )
    ap.add_argument('--verbose', '-v', help='Log more verbosely', action='store_true', default=False)
    ap.add_argument('--parallel', '-j', default=None, type=int, metavar='N',
        help='Run with up to N threads. By default, it uses as many processors as available.')

    subp = ap.add_subparsers(dest='command')

    for subcommand in itervalues(ALL_COMMANDS):
        subcommand.add_opts(subp)

    return ap.parse_args()
