import numpy
from functools import wraps

try:
    import numba

    try:
        import numba.cuda
        if not len(list(numba.cuda.gpus)):
            raise NotImplementedError()
        import logging
        logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARN)
        with_cuda = True
    except Exception:
        with_cuda = False
    with_numba = True
except ImportError:
    with_numba = False


def auto_vectorize(sigs, big_thresh=1000000, cuda=True, size_arg=0, out_arg=None, cache=True, fastmath=True, **kw):
    if not with_numba:
        raise NotImplementedError("Vectorize only works with numba")

    def decorator(ufunc):
        _sml = numba.vectorize(sigs, target='cpu', cache=cache, fastmath=fastmath, **kw)(ufunc)
        if big_thresh is not None:
            if cuda and with_cuda:
                _big = numba.vectorize(sigs, target='cuda', fastmath=fastmath, **kw)(ufunc)
            else:
                _big = numba.vectorize(sigs, target='parallel', cache=cache, fastmath=fastmath, **kw)(ufunc)

        @wraps(ufunc)
        def decorated(*p, **kw):
            size = p[size_arg].size
            if out_arg is None:
                out = kw.pop('out', None)
            else:
                out = p[out_arg]

            force_cuda = cuda and with_cuda and kw.pop('force_cuda', None)

            if force_cuda or (big_thresh is not None and size >= big_thresh):
                if cuda and with_cuda and out is not None and not numba.cuda.is_cuda_array(out):
                    # out doesn't help
                    out[:] = _big(*p, **kw)
                    return out
                else:
                    if out is not None:
                        kw['out'] = out
                    return _big(*p, **kw)
            else:
                if out is not None:
                    kw['out'] = out
                return _sml(*p, **kw)
        decorated.cuda = cuda and with_cuda
        return decorated
    return decorator


def auto_guvectorize(sigs, layout, big_thresh=1000000, cuda=True, size_arg=0, out_arg=None):
    if not with_numba:
        raise NotImplementedError("Vectorize only works with numba")

    def decorator(ufunc):
        _sml = numba.guvectorize(sigs, layout, target='cpu')(ufunc)
        if big_thresh is not None:
            if cuda and with_cuda:
                _big = numba.guvectorize(sigs, layout, target='cuda')(ufunc)
            else:
                _big = numba.guvectorize(sigs, layout, target='parallel')(ufunc)

        @wraps(ufunc)
        def decorated(*p, **kw):
            size = p[size_arg].size
            if out_arg is None:
                out = kw.pop('out', None)
            else:
                out = p[out_arg]

            if big_thresh is not None and size >= big_thresh:
                if with_cuda and out is not None and not numba.cuda.is_cuda_array(out):
                    # out doesn't help
                    out[:] = _big(*p, **kw)
                    return out
                else:
                    if out is not None:
                        kw['out'] = out
                    return _big(*p, **kw)
            else:
                if out is not None:
                    kw['out'] = out
                return _sml(*p, **kw)
        return decorated
    return decorator
