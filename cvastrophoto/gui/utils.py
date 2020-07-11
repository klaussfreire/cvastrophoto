# -*- coding: utf-8 -*-


def _p(w, *p, **kw):
    w.pack(*p, **kw)
    return w


def _g(w, *p, **kw):
    w.grid(*p, **kw)
    return w


def _focus_get(c):
    try:
        c.focus_get()
    except KeyError:
        return None
