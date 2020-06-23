# -*- coding: utf-8 -*-


def _p(w, *p, **kw):
    w.pack(*p, **kw)
    return w


def _g(w, *p, **kw):
    w.grid(*p, **kw)
    return w
