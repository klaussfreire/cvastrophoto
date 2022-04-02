try:
    import cPickle
except ImportError:
    import pickle as cPickle

from functools import partial
import gzip

from past.builtins import basestring
from six import iteritems


def keyreplace(pat, repl, ts):
    """ Replaces any occurrence of pat with repl within tracking state cache data

    This helper will perform a recursive replace over the tracking state, replacing
    occurrences of the pat string with repl, allowing movement of tracking state data
    from one folder to another, among other possible uses.
    """
    if isinstance(ts, (list, tuple)):
        return type(ts)(map(partial(keyreplace, pat, repl), ts))
    elif isinstance(ts, dict):
        return {keyreplace(pat, repl, k): keyreplace(pat, repl, v) for k, v in iteritems(ts)}
    elif isinstance(ts, basestring):
        return ts.replace(pat, repl)
    else:
        return ts


def load_ts(path, compressed=True):
    if compressed:
        gzf = gzip.GzipFile(path, mode='rb')
    else:
        gzf = open(path, mode='rb')
    with gzf:
        return cPickle.load(gzf)

def save_ts(path, ts, compressed=True):
    if compressed:
        gzf = gzip.GzipFile(path, mode='wb')
    else:
        gzf = open(path, mode='wb')
    with gzf:
        cPickle.dump(ts, gzf, 2)