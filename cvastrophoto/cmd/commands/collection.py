# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import itertools
import sys
import os.path

import logging

from .collection_utils import cls_str, get_sorter, parse_selectors


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
    add_add(subp.add_parser('add', help='Add images to a collection'))

def add_sort(ap):
    ap.add_argument('--auto-add', '-a', action='store_true',
        help='Automatically add items to matching existing collections')
    ap.add_argument('--auto-create', '-A', action='store_true',
        help='Automatically create new collections with fully identified items')
    ap.add_argument('--subcollection', '-s', help='Subcollection to add to (when autoadding)')
    ap.add_argument('--overrides', '-o', nargs='+',
        help='Override sorting attributes, in the form of collection selectors, will take precedence over tags.')
    ap.add_argument('files', nargs='+')

def add_list(ap):
    ap.add_argument('--recursive', '-r', action='store_true')
    ap.add_argument('--complete', '-c', action='store_true',
        help='Only list complete collections, ie, those that have all basic semantic attributes')
    ap.add_argument('--items', '-i', action='store_true',
        help='Only list items, no subcollections')
    ap.add_argument('--subcollections', '-s', action='store_true',
        help='Only list subcollections, no items')
    ap.add_argument('--no-empty', '-n', action='store_true',
        help='Do not include empty collections in the listing, ie those without itmes, even if they have subcollections')
    ap.add_argument('--full', '-f', action='store_true',
        help='List item full paths')
    ap.add_argument('selectors', nargs='*')

def add_add(ap):
    ap.add_argument('--recursive', '-r', action='store_true')
    ap.add_argument('--collection', '-c', nargs='+',
        help=(
            'Collection selector. Each following argument will be selecting a subcollection, the last '
            'of which will be the one the images will be added to'
        ))
    ap.add_argument('paths', nargs='+',
        help=(
            'List of files or directories whose files will be added to the collection, recursively if '
            'recursive was given'
        ))

def main(opts, pool):
    from cvastrophoto.collections import filesystem
    if opts.path is not None:
        collections = filesystem.FilesystemCollection(opts.path)
    else:
        logger.error("No collection specified")
        sys.exit(1)
    ACTIONS[opts.action](opts, pool, collections)

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

def is_fully_defined(sem):
    return all(
        sem.get(k, 'NA').upper() not in ('NA', 'UNKNOWN', '')
        for k in AUTOCREATE_MANDATORY_FIELDS
    )

def sort_action(opts, pool, collections):
    from cvastrophoto import image
    sorter = get_sorter(opts)

    if opts.overrides:
        overrides = parse_selectors(opts, opts.overrides, get_semantic=True)
    else:
        overrides = None

    def classify(path):
        return path, sorter.classify_frame(path)

    auto_create = opts.auto_create
    auto_add = opts.auto_add or auto_create

    for path, cls in pool.imap_unordered(classify, expandfiles(opts.files)):
        cls_sem = sorter.semantic(cls)
        if overrides:
            cls_sem.update(overrides)
            cls = sorter.from_semantic(cls_sem)
        do_add = False
        fully_defined = is_fully_defined(cls_sem)
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


def list_action(opts, pool, collections):
    baseclass, kwfilter = parse_selectors(opts, opts.selectors)

    if not opts.items:
        if not opts.subcollections:
            # No need to print this header when listing only subcollections
            print("Subcollections:\n")

        for key in collections.list_subcollections(baseclass, recurse=opts.recursive or kwfilter is not None):
            if kwfilter is not None and not kwfilter(key):
                continue
            if opts.complete:
                sorter = get_sorter(opts)
                if not is_fully_defined(sorter.semantic(key)):
                    continue
            if opts.no_empty:
                for _ in collections.list_paths(key):
                    break
                else:
                    continue

            print(' '.join(map(repr, key)))

    if not opts.subcollections:
        if not opts.items:
            # No need to print this header when listing only items
            print("\nItems:\n")

        if kwfilter is not None:
            bases = [
                key
                for key in collections.list_subcollections(baseclass, recurse=opts.recursive or kwfilter is not None)
                if kwfilter(key)
            ]
        else:
            bases = [baseclass]

        for baseclass in bases:
            for path in collections.list_paths(baseclass):
                if opts.full:
                    print(path)
                else:
                    dirname, basename = os.path.split(path)
                    print(basename)


def add_action(opts, pool, collections):
    baseclass, kwfilter = parse_selectors(opts, opts.collection)

    if kwfilter:
        targetclass = None
        for key in collections.list_subcollections(baseclass, recurse=opts.recursive):
            if not kwfilter(key):
                continue
            if targetclass is not None:
                print("Collection selector ambiguous")
                sys.exit(1)
            else:
                targetclass = key
        if targetclass is not None:
            baseclass = targetclass

    if not baseclass:
        print("Must specify a destination collection")
        sys.exit(1)

    from cvastrophoto.image import Image

    for pathname in opts.paths:
        if os.path.isdir(pathname):
            if opts.recursive:
                def paths():
                    for root, dirnames, filenames in os.walk(pathname):
                        for filename in filenames:
                            yield os.path.join(root, filename)
            else:
                def paths():
                    for filename in os.listdir(pathname):
                        fullpath = os.path.join(pathname, filename)
                        if os.path.isfile(fullpath):
                            yield fullpath
            paths = paths()

            def addit(path):
                if os.path.isfile(path) and not Image.IGNORE.match(path) and Image.supports(path):
                    with Image.open(path) as img:
                        collections.add(baseclass, None, img)
                        return path
            for path in pool.imap_unordered(addit, paths):
                pass
        else:
            with Image.open(pathname) as img:
                collections.add(baseclass, None, img)
            pass


ACTIONS = {
    'sort': sort_action,
    'list': list_action,
    'add': add_action,
}
