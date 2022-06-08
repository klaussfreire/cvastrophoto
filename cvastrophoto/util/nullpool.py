
class Result:

    def __init__(self, func, args, kwds):
        self.func = func
        self.args = args
        self.kwds = kwds

    def _do(self):
        if hasattr(self, "rv"):
            return self.rv
        elif hasattr(self, "exc"):
            raise self.exc

        try:
            self.rv = self.func(*self.args, **self.kwds)
            return self.rv
        except Exception as e:
            self.exc = e
            raise

    def get(self):
        return self._do()

    def wait(self):
        return True

    def ready(self):
        return True

    def successfull(self):
        return not hasattr(self, "exc")


class NullPool:

    def apply(self, func, args=(), kwds={}):
        return func(*args, **kwds)

    def map(self, func, iterable, chunksize=None):
        return list(map(func, iterable))

    def imap(self, func, iterable, chunksize=None):
        return map(func, iterable)

    imap_unordered = imap

    def apply_async(self, func, args=(), kwds={}):
        return Result(func, args, kwds)

    def map_async(self, func, iterable, chunksize=None):
        return [self.apply_async(func, (x,)) for x in iterable]

    def starmap(self, func, iterable, chunksize=None):
        return [func(*x) for x in iterable]

    def starmap_async(self, func, iterable, chunksize=None):
        return [self.apply_async(func, x) for x in iterable]

    def _nop(self, *p, **kw):
        pass

    close = terminate = join = _nop
