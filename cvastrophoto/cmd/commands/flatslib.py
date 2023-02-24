# -*- coding: utf-8 -*-
from .darklib import add_opts as base_opts, main as base_main, build as base_build, classify as base_classify

def add_opts(subp):
    return base_opts(
        subp,
        name='flatslib', help='Manage the flats library',
        libname='flats library', subname='flats',
        defpaths=['Flats'],
        defpath='~/.cvastrophoto/flatslib', localpath='./flatslib',
        actions=ACTIONS
    )

def main(opts, pool):
    return base_main(opts, pool, localpath='flatslib', actions=ACTIONS)

def build(opts, pool):
    from cvastrophoto.library import flats
    return base_build(opts, pool, LibClass=flats.FlatLibrary)

def classify(opts, pool):
    from cvastrophoto.library import flats
    return base_classify(opts, pool, LibClass=flats.FlatLibrary)

ACTIONS = {
    'build': build,
    'classify': classify,
}
