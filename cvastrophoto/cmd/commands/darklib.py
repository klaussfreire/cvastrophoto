# -*- coding: utf-8 -*-
import multiprocessing.pool
import itertools
import os.path

def add_opts(subp):
    ap = subp.add_parser('darklib', help='Manage the dark library')

    ap.add_argument('--path', '-p',
        help='Location of the dark library', default='~/.cvastrophoto/darklib')
    ap.add_argument('--sources', '-s',
        help='Location of darks to be added', nargs='+', default=['Darks', 'Dark Flats'])
    ap.add_argument('--refresh', action='store_true', default=False,
        help=(
            'Reconstruct library. When given, any darks found will be added to the '
            'library unconditionally. The default is to skip darks for which a valid '
            'master dark can be found already within the library.'
        ))
    ap.add_argument('--local', action='store_true', default=False,
        help='Manage the local dark library in "./darklib". Overrides --path')
    ap.add_argument('--parallel', '-j', default=None, type=int, metavar='N',
        help='Run with up to N threads. By default, it uses as many processors as available.')

    ap.add_argument('action', choices=ACTIONS.keys())

def main(opts):
    if opts.local:
        opts.path = 'darklib'
    ACTIONS[opts.action](opts)

def build(opts):
    from cvastrophoto.library import darks

    if opts.parallel:
        pool = multiprocessing.pool.ThreadPool(opts.parallel)
    else:
        pool = None

    lib = darks.DarkLibrary(opts.path, default_pool=pool)
    lib.build(
        itertools.chain(*[
            lib.list_recursive(s)
            for s in opts.sources if os.path.exists(s)
        ]),
        refresh=opts.refresh)

ACTIONS = {
    'build': build,
}
