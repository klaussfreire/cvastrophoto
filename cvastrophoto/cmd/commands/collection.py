# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import itertools
import sys
import os.path

import logging


logger = logging.getLogger(__name__)


def add_opts(subp,
        name='collection', help='Manage a collection',
        defpath='~/.cvastrophoto/collections',
        actions=None):
    ap = subp.add_parser(name, help=help)

    ap.add_argument('--path', '-p',
        help='Location of the collections', default=defpath)

    subp = ap.add_subparsers(dest='action')

    add_sort(subp.add_parser('sort', help='Sort out subs into existing or new collections'))
    add_list(subp.add_parser('list', help='List collections'))

def add_sort(ap):
    ap.add_argument('--auto-add', '-a', action='store_true',
        help='Automatically add items to matching existing collections')
    ap.add_argument('--auto-create', '-A', action='store_true',
        help='Automatically create new collections with fully identified items')
    ap.add_argument('--subcollection', '-s', help='Subcollection to add to (when autoadding)')
    ap.add_argument('files', nargs='+')

def add_list(ap):
    ap.add_argument('--recursive', '-r', action='store_true')
    ap.add_argument('selectors', nargs='*')

def main(opts, pool):
    from cvastrophoto.collections import filesystem
    if opts.path is not None:
        collections = filesystem.FilesystemCollection(opts.path)
    else:
        logger.error("No collection specified")
        sys.exit(1)
    ACTIONS[opts.action](opts, pool, collections)

def get_sorter(opts):
    from cvastrophoto.collections import tagsort
    return tagsort.CollectionTagSorting()

def cls_str(cls_sem):
    return ' '.join('%s=%r' % (k, v) for k, v in cls_sem.items())

AUTOCREATE_MANDATORY_FIELDS = {
    'OBJECT',
    'FRAME',
    'DATEOBS',
}

def expandfiles(files):
    from cvastrophoto.image import Image
    for path in files:
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    fullpath = os.path.join(dirpath, filename)
                    if Image.supports(fullpath):
                        yield fullpath
        else:
            yield path

def sort_action(opts, pool, collections):
    from cvastrophoto import image
    sorter = get_sorter(opts)

    def classify(path):
        return path, sorter.classify_frame(path)

    auto_create = opts.auto_create
    auto_add = opts.auto_add or auto_create

    for path, cls in pool.imap_unordered(classify, expandfiles(opts.files)):
        cls_sem = sorter.semantic(cls)
        do_add = False
        fully_defined = all(
            cls_sem.get(k, 'NA').upper() not in ('NA', 'UNKNOWN', '')
            for k in AUTOCREATE_MANDATORY_FIELDS
        )
        logger.info("Classified %r as %s", path, cls_str(cls_sem))
        if auto_add and collections.contains(cls, opts.subcollection):
            do_add = True
        elif auto_create:
            if fully_defined:
                logger.info("New collection %s", cls_str(cls_sem))
                do_add = True
        elif fully_defined and auto_add:
            # New DATEOBS?
            base_cls = cls[:sorter.classification_semantic.index('DATEOBS')]
            if collections.contains(base_cls):
                do_add = True

        if do_add:
            with image.Image.open(path) as img:
                if not collections.contains(cls, opts.subcollection, img):
                    collections.add(cls, opts.subcollection, img)
                    logger.info("Added %r to %s", path, cls_str(cls_sem))
                else:
                    logger.info("Already present %r in %s", path, cls_str(cls_sem))
        elif fully_defined:
            logger.info("Can add %r to %s", path, cls_str(cls_sem))


def parse_selectors(opts, selectors):
    pos_selectors = [None] * len(selectors)
    kw_selectors = {}
    for i, attval in enumerate(selectors):
        if '=' in attval:
            attname, attval = attval.split('=', 1)
            kw_selectors[attname] = attval
        else:
            pos_selectors[i] = attval

    if kw_selectors:
        sorter = get_sorter(opts)
        pos_kw = sorter.semantic(pos_selectors)
        kw_selectors.update({k: v for k, v in pos_kw.items() if v})
        pos_selectors = sorter.from_semantic(kw_selectors, None)

    pos_selectors = list(pos_selectors)
    while pos_selectors and pos_selectors[-1] is None:
        del pos_selectors[-1]

    kwfilter = lambda key: True
    if None in pos_selectors:
        sorter = get_sorter(opts)
        kw_selectors = {k: v for k, v in sorter.semantic(pos_selectors).items() if v is not None}
        pos_selectors = pos_selectors[:pos_selectors.index(None)]

        def kwfilter(key):
            key_sem = sorter.semantic(key)
            for k, v in kw_selectors.items():
                if key_sem.get(k) != v:
                    return False
            else:
                return True

    return pos_selectors, kwfilter


def list_action(opts, pool, collections):
    baseclass, kwfilter = parse_selectors(opts, opts.selectors)

    print("Subcollections:\n")

    for key in collections.list_subcollections(baseclass, recurse=opts.recursive):
        if not kwfilter(key):
            continue
        print(' '.join(map(repr, key)))

    print("\nItems:\n")

    for path in collections.list_paths(baseclass):
        dirname, basename = os.path.split(path)
        print(basename)


ACTIONS = {
    'sort': sort_action,
    'list': list_action,
}
