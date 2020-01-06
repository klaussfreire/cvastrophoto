# -*- coding: utf-8 -*-
from .darklib import add_opts as base_opts, main as base_main, build as base_build

def add_opts(subp):
    return base_opts(
        subp,
        name='biaslib', help='Manage the bias library',
        libname='bias library', subname='bias',
        defpaths=['Bias'],
        defpath='~/.cvastrophoto/biaslib', localpath='./biaslib',
        actions=ACTIONS
    )

def main(opts, pool):
    return base_main(opts, pool, localpath='biaslib', actions=ACTIONS)

def build(opts, pool):
    from cvastrophoto.library import bias
    return base_build(opts, pool, LibClass=bias.BiasLibrary)

ACTIONS = {
    'build': build,
}
