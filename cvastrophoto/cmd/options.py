# -*- coding: utf-8 -*-
import argparse

from .commands import ALL_COMMANDS

def parse():
    ap = argparse.ArgumentParser(
        prog='cvastrophoto',
        description='Astrophotography workflow library',
    )
    ap.add_argument('--verbose', '-v', help='Log more verbosely', action='store_true', default='False')

    subp = ap.add_subparsers(dest='command')

    for subcommand in ALL_COMMANDS.itervalues():
        subcommand.add_opts(subp)

    return ap.parse_args()
