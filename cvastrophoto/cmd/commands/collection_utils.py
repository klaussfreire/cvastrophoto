from __future__ import absolute_import


def get_sorter(opts):
    from cvastrophoto.collections import tagsort
    return tagsort.CollectionTagSorting()


def parse_selectors(opts, selectors, get_semantic=False):
    pos_selectors = [None] * len(selectors)
    kw_selectors = {}
    for i, attval in enumerate(selectors):
        if '=' in attval:
            attname, attval = attval.split('=', 1)
            kw_selectors[attname] = attval
        else:
            pos_selectors[i] = attval

    if get_semantic:
        if any(pos_selectors):
            sorter = get_sorter(opts)
            sem_selectors = sorter.semantic(pos_selectors)
            kw_selectors.update({k: v for k, v in sem_selectors.items() if v})
        return kw_selectors

    if kw_selectors:
        sorter = get_sorter(opts)
        pos_kw = sorter.semantic(pos_selectors)
        kw_selectors.update({k: v for k, v in pos_kw.items() if v})
        pos_selectors = sorter.from_semantic(kw_selectors, None)

    pos_selectors = list(pos_selectors)
    while pos_selectors and pos_selectors[-1] is None:
        del pos_selectors[-1]

    kwfilter = None
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


def cls_str(cls_sem):
    return ' '.join('%s=%r' % (k, v) for k, v in cls_sem.items())
