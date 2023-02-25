# -*- coding: utf-8 -*-
from __future__ import print_function

import itertools
import os.path


def add_opts(subp,
        name='darklib', help='Manage the dark library',
        libname='dark library', subname='dark',
        defpaths=['Darks', 'Dark Flats'],
        defpath='~/.cvastrophoto/darklib', localpath='./darklib',
        actions=None):
    ap = subp.add_parser(name, help=help)

    ap.add_argument('--path', '-p',
        help='Location of the %s' % libname, default=defpath)
    ap.add_argument('--sources', '-s',
        help='Location of %ss to be added' % subname, nargs='+', default=defpaths)
    ap.add_argument('--refresh', action='store_true', default=False,
        help=(
            'Reconstruct library. When given, any %(subname)ss found will be added to the '
            'library unconditionally. The default is to skip %(subname)s for which a valid '
            'master dark can be found already within the library.'
        ) % dict(subname=subname))
    ap.add_argument('--local', action='store_true', default=False,
        help='Manage the local %(libname)s in "%(localpath)s". Overrides --path' % dict(
            localpath=localpath, libname=libname))

    if actions is None:
        actions = ACTIONS

    ap.add_argument('action', choices=actions.keys())

def main(opts, pool, localpath='darklib', actions=None):
    if opts.local:
        opts.path = localpath
    if actions is None:
        actions = ACTIONS
    actions[opts.action](opts, pool)

def build(opts, pool, LibClass=None):
    if LibClass is None:
        from cvastrophoto.library import darks
        LibClass = darks.DarkLibrary

    lib = LibClass(opts.path, default_pool=pool)
    lib.build(
        itertools.chain(*[
            lib.list_recursive(s)
            for s in opts.sources if os.path.exists(s)
        ]),
        refresh=opts.refresh)

def classify(opts, pool, LibClass=None):
    if LibClass is None:
        from cvastrophoto.library import darks
        LibClass = darks.DarkLibrary

    lib = LibClass(opts.path, default_pool=pool)
    classified_paths = lib.classify_all(
        itertools.chain(*[
            lib.list_recursive(s, filter_fn=lambda *p, **kw:True)
            for s in opts.sources if os.path.exists(s)
        ]),
        refresh=opts.refresh)

    for img_path, img_key in classified_paths:
        print(img_path, ":", "/".join(img_key))

ACTIONS = {
    'build': build,
    'classify': classify,
}
