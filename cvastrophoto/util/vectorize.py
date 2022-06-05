import numpy
import os
from functools import wraps

CUDA_ERRORS = None

try:
    import numba

    with_cuda = False
    try_cuda = os.environ.get('NUMBA_CUDA', 'yes') != 'no'
    if try_cuda:
        try:
            import numba.cuda
            if not len(list(numba.cuda.gpus)):
                raise NotImplementedError()
            import logging
            for logname in [
                    'numba.cuda.cudadrv.driver',
                    'numba.core.byteflow',
                    'numba.core.ssa',
                    'numba.core.interpreter']:
                logging.getLogger(logname).setLevel(logging.WARN)
            with_cuda = True
        except Exception:
            pass
        else:
            from numba.cuda import CudaSupportError
            from numba.cuda.cudadrv.error import CudaDriverError, CudaRuntimeError
            CUDA_ERRORS = (CudaSupportError, CudaDriverError, CudaRuntimeError)

            dev = numba.cuda.get_current_device()
            CUDA_MAX_THREADS_PER_BLOCK = dev.MAX_THREADS_PER_BLOCK
            CUDA_MAX_BLOCK_DIM_X = dev.MAX_BLOCK_DIM_X
            CUDA_MAX_BLOCK_DIM_Y = dev.MAX_BLOCK_DIM_Y
            CUDA_MAX_SHMEM = dev.MAX_SHARED_MEMORY_PER_BLOCK
            CUDA_CAN_MAP = dev.CAN_MAP_HOST_MEMORY
            CUDA_NPROC = dev.MULTIPROCESSOR_COUNT
            CUDA_WARP_SIZE = dev.WARP_SIZE
    with_numba = True
    with_parallel = os.environ.get('NUMBA_PARALLEL', 'yes') != 'no'
except ImportError:
    with_cuda = False
    with_numba = False
    with_parallel = False

if CUDA_ERRORS is None:
    # Bogus class so the CUDA_ERRORS tuple can still be used in context
    class NoError(Exception):
        pass
    CUDA_ERRORS = (NoError,)


def auto_vectorize(sigs, big_thresh=1000000, cuda=True, size_arg=0, out_arg=None, cache=True, fastmath=True, **kw):
    if not with_numba:
        raise NotImplementedError("Vectorize only works with numba")

    def decorator(ufunc):
        _sml = numba.vectorize(sigs, target='cpu', cache=cache, fastmath=fastmath, **kw)(ufunc)
        if big_thresh is not None:
            if cuda and with_cuda:
                _big = numba.vectorize(sigs, target='cuda', cache=cache, fastmath=fastmath, **kw)(ufunc)
            else:
                big_target = 'parallel' if with_parallel else 'cpu'
                _big = numba.vectorize(sigs, target=big_target, cache=cache, fastmath=fastmath, **kw)(ufunc)

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
