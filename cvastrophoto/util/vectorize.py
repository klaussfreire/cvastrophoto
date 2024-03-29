from __future__ import division

import multiprocessing
import threading
import numpy
import os
import math
import logging
from functools import wraps
from past.builtins import xrange

from pyrsistent import v

from cvastrophoto.accel.config import numba_cuda as try_cuda
from cvastrophoto.util import nullpool


CUDA_ERRORS = None

logger = logging.getLogger(__name__)


try:
    import numba

    with_cuda = False
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
            CUDA_ERRORS = (CudaSupportError, CudaDriverError, CudaRuntimeError, MemoryError)

            dev = numba.cuda.get_current_device()
            CUDA_MAX_THREADS_PER_BLOCK = dev.MAX_THREADS_PER_BLOCK
            CUDA_MAX_BLOCK_DIM_X = dev.MAX_BLOCK_DIM_X
            CUDA_MAX_BLOCK_DIM_Y = dev.MAX_BLOCK_DIM_Y
            CUDA_MAX_GRID_DIM_X = dev.MAX_GRID_DIM_X
            CUDA_MAX_GRID_DIM_Y = dev.MAX_GRID_DIM_Y
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
                _big_fn = numba.vectorize(sigs, target='cuda', cache=cache, fastmath=fastmath, **kw)(ufunc)
                _big = lambda *p, **kw: in_cuda_pool(None, _big_fn, *p, **kw).get()
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


def auto_guvectorize(sigs, layout, big_thresh=1000000, cuda=True, size_arg=0, out_arg=None,
        tile_param=None, tile_axis=0):
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
            elif out_arg < len(p):
                out = p[out_arg]
            else:
                out = None

            if big_thresh is not None and size >= big_thresh:
                if with_cuda and tile_param is not None:
                    tile_in = p[tile_param]
                    required_mem = tile_in.size * tile_in.dtype.itemsize * 3 # input + output + overhead
                    if cuda_free_mem() < required_mem:
                        stride = max(1, (tile_in.shape[tile_axis] * cuda_free_mem() // required_mem + 1) // 2 + 1)
                        if out is None:
                            out = numpy.empty_like(tile_in)
                        for y in xrange(0, tile_in.shape[tile_axis], stride):
                            slicing = tuple(
                                [slice(None)] * tile_axis
                                + [slice(y, y+stride)]
                                + [slice(None)] * (len(tile_in.shape) - tile_axis - 1)
                            )
                            slicep = p[:tile_param] + (tile_in[slicing],) + p[tile_param+1:]
                            out[slicing] = _big(*slicep, **kw)
                        return out

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


if with_cuda:
    # Various CUDA helpers

    def cuda_block_config(imshape, threadsperblock):
        blockspergrid = tuple([math.ceil(imdim / blockdim) for imdim, blockdim in zip(imshape, threadsperblock)])
        return blockspergrid, threadsperblock

    CUDA_POOL_THREADS = int(os.environ.get('CUDA_POOL_THREADS', '1'))
    if CUDA_POOL_THREADS == -1:
        CUDA_POOL_THREADS = None

    # Minimum amount of RAM expected out of a CUDA device
    MIN_CUDA_MEM = (1 << 30)

    # Any task that requires less than this amount of RAM does not need to run in the CUDA pool
    MAX_INLINE_TASK_MEM = MIN_CUDA_MEM // (multiprocessing.cpu_count() * 4)

    # Any task that requires less than this amount of RAM does not need a pre-emptive memory check
    MEMCHECK_MIN_MEM = MIN_CUDA_MEM // CUDA_POOL_THREADS // 2

    _oom_cleanup = []

    _cuda_pool = None
    _cuda_tl = threading.local()

    def cuda_pool():
        global _cuda_pool
        if _cuda_pool is None:
            import atexit
            from multiprocessing.pool import ThreadPool
            _cuda_pool = ThreadPool(CUDA_POOL_THREADS)
            atexit.register(_cuda_pool.terminate)
        return _cuda_pool

    def close_cuda_pool():
        global _cuda_pool
        if _cuda_pool is not None:
            _cuda_pool.terminate()
            _cuda_pool = None

    def register_oom_cleanup(cb):
        _oom_cleanup.append(cb)
        return cb

    def _call_oom_cleanup():
        for cb in _oom_cleanup:
            try:
                cb()
            except Exception:
                logger.exception("Error ignored in OOM cleanup callback")

    def in_cuda_pool(required_mem, fn, *p, **kw):
        if required_mem is not None:
            if required_mem < MAX_INLINE_TASK_MEM:
                # Trivial amount of RAM, run in-place
                return nullpool.Result(fn, p, kw)
            elif required_mem < MEMCHECK_MIN_MEM:
                # Skip the memory check
                required_mem = None
        if required_mem is not None:
            if cuda_total_mem() < required_mem:
                raise MemoryError("Not enough VRAM for operation")
            if getattr(_cuda_tl, '_is_cuda_pool', False):
                # Call directly, no need to recurse
                logger.warn("Re-entrance in the CUDA pool is frowned upon")
                return nullpool.Result(_fn, p, kw)
            _fn = fn
            def wrapfn(*p, **kw):
                _cuda_tl.is_cuda_pool = True
                if cuda_free_mem() < required_mem:
                    _call_oom_cleanup()
                if cuda_free_mem() < required_mem:
                    raise MemoryError("Not enough VRAM for operation")
                return _fn(*p, **kw)
            fn = wrapfn
        return cuda_pool().apply_async(fn, p, kw)

    @numba.cuda.jit
    def _cuda_fill_1d(arr, value):
        x = numba.cuda.grid(1)
        if x < arr.shape[0]:
            arr[x] = value

    def cuda_fill(arr, value):
        arr = arr.reshape(arr.size)
        _cuda_fill_1d[cuda_block_config(arr.shape, (128,))](arr, value)

    @numba.cuda.jit
    def _cuda_mul_1d(a, b, out):
        x = numba.cuda.grid(1)
        if x < out.shape[0]:
            out[x] = a[x] * b[x]

    def cuda_mul(a, b, out=None):
        a = a.reshape(a.size)
        b = b.reshape(a.size)
        if out is None:
            out = numba.cuda.device_array_like(a)
        else:
            out = out.reshape(a.size)
        _cuda_mul_1d[cuda_block_config(a.shape, (128,))](a, b, out)
        return out

    @numba.cuda.reduce
    def cuda_sum(a, b):
        return a + b

    @numba.cuda.jit(fastmath=True)
    def _cuda_sum2_partial(ary, out):
        x = numba.cuda.grid(1)

        if x < out.shape[0]:
            partial = 0.0
            for ix in range(x * numba.cuda.blockDim.x, (x+1) * numba.cuda.blockDim.x):
                if ix < ary.shape[0]:
                    partial += ary[ix] * ary[ix]

            out[x] = partial


    from numba.np.numpy_support import from_dtype

    _NUMWARPS = 4

    def _gpu_reduce_factory(fn, combine, nbtype):
        cuda = numba.cuda

        reduce_op = cuda.jit(device=True)(fn)
        combine_op = cuda.jit(device=True)(combine)
        inner_sm_size = CUDA_WARP_SIZE + 1   # plus one to avoid SM collision
        max_blocksize = _NUMWARPS * CUDA_WARP_SIZE

        @cuda.jit(device=True)
        def inner_warp_reduction(sm_partials, init):
            """
            Compute reduction within a single warp
            """
            tid = cuda.threadIdx.x
            warpid = tid // CUDA_WARP_SIZE
            laneid = tid % CUDA_WARP_SIZE

            sm_this = sm_partials[warpid, :]
            sm_this[laneid] = init
            cuda.syncwarp()

            width = CUDA_WARP_SIZE // 2
            while width:
                if laneid < width:
                    old = sm_this[laneid]
                    sm_this[laneid] = combine_op(old, sm_this[laneid + width])
                cuda.syncwarp()
                width //= 2

        @cuda.jit(device=True)
        def device_reduce_full_block(arr, partials, sm_partials, zero):
            """
            Partially reduce `arr` into `partials` using `sm_partials` as working
            space.  The algorithm goes like:
                array chunks of 128:  |   0 | 128 | 256 | 384 | 512 |
                            block-0:  |   x |     |     |   x |     |
                            block-1:  |     |   x |     |     |   x |
                            block-2:  |     |     |   x |     |     |
            The array is divided into chunks of 128 (size of a threadblock).
            The threadblocks consumes the chunks in roundrobin scheduling.
            First, a threadblock loads a chunk into temp memory.  Then, all
            subsequent chunks are combined into the temp memory.
            Once all chunks are processed.  Inner-block reduction is performed
            on the temp memory.  So that, there will just be one scalar result
            per block.  The result from each block is stored to `partials` at
            the dedicated slot.
            """
            tid = cuda.threadIdx.x
            blkid = cuda.blockIdx.x
            blksz = cuda.blockDim.x
            gridsz = cuda.gridDim.x

            # block strided loop to compute the reduction
            start = tid + blksz * blkid
            stop = arr.size
            step = blksz * gridsz

            # load first value
            tmp = zero
            # loop over all values in block-stride
            for i in range(start, stop, step):
                tmp = reduce_op(arr[i], tmp)

            cuda.syncthreads()
            # inner-warp reduction
            inner_warp_reduction(sm_partials, tmp)

            cuda.syncthreads()
            # at this point, only the first slot for each warp in tsm_partials
            # is valid.

            # finish up block reduction
            # warning: this is assuming 4 warps.
            # assert numwarps == 4
            if tid < 2:
                sm_partials[tid, 0] = combine_op(sm_partials[tid, 0], sm_partials[tid + 2, 0])
                cuda.syncwarp()
            if tid == 0:
                partials[blkid] = combine_op(sm_partials[0, 0], sm_partials[1, 0])

        @cuda.jit(device=True)
        def device_reduce_partial_block(arr, partials, sm_partials, zero):
            """
            This computes reduction on `arr`.
            This device function must be used by 1 threadblock only.
            The blocksize must match `arr.size` and must not be greater than 128.
            """
            tid = cuda.threadIdx.x
            blkid = cuda.blockIdx.x
            blksz = cuda.blockDim.x
            warpid = tid // CUDA_WARP_SIZE
            laneid = tid % CUDA_WARP_SIZE

            size = arr.size
            # load first value
            tid = cuda.threadIdx.x
            value = reduce_op(arr[tid], zero)
            sm_partials[warpid, laneid] = value

            cuda.syncthreads()

            if (warpid + 1) * CUDA_WARP_SIZE < size:
                # fully populated warps
                inner_warp_reduction(sm_partials, value)
            else:
                # partially populated warps
                # NOTE: this uses a very inefficient sequential algorithm
                if laneid == 0:
                    sm_this = sm_partials[warpid, :]
                    base = warpid * CUDA_WARP_SIZE
                    for i in range(1, size - base):
                        sm_this[0] = combine_op(sm_this[0], sm_this[i])

            cuda.syncthreads()
            # finish up
            if tid == 0:
                num_active_warps = (blksz + CUDA_WARP_SIZE - 1) // CUDA_WARP_SIZE

                result = sm_partials[0, 0]
                for i in range(1, num_active_warps):
                    result = combine_op(result, sm_partials[i, 0])

                partials[blkid] = result

        def gpu_reduce_block_strided(arr, partials, init, use_init, zero):
            """
            Perform reductions on *arr* and writing out partial reduction result
            into *partials*.  The length of *partials* is determined by the
            number of threadblocks. The initial value is set with *init*.
            Launch config:
            Blocksize must be multiple of warpsize and it is limited to 4 warps.
            """
            tid = cuda.threadIdx.x

            sm_partials = cuda.shared.array((_NUMWARPS, inner_sm_size), dtype=nbtype)
            if cuda.blockDim.x == max_blocksize:
                device_reduce_full_block(arr, partials, sm_partials, zero)
            else:
                device_reduce_partial_block(arr, partials, sm_partials, zero)
            # deal with the initializer
            if use_init and tid == 0 and cuda.blockIdx.x == 0:
                partials[0] = combine_op(partials[0], init)

        return cuda.jit(gpu_reduce_block_strided)


    class Reduce(object):
        """Create a reduction object that reduces values using a given binary
        function. The binary function is compiled once and cached inside this
        object. Keeping this object alive will prevent re-compilation.
        """

        _cache = {}

        def __init__(self, functor, combine=lambda a,b:a+b,
                temp_dtmap={
                    'f': numpy.float64,
                }):
            """
            :param functor: A function implementing a binary operation for
                            reduction. It will be compiled as a CUDA device
                            function using ``cuda.jit(device=True)``.
                            It will be called with (new_value, accumulated).
            :param combine> A function implementing combination of partially
                            accumulated results. By default, it just adds.
            """
            self._functor = functor
            self._combine = combine
            self._temp_dtmap = temp_dtmap

        def _compile(self, dtype):
            key = self._functor, dtype
            if key in self._cache:
                kernels = self._cache[key]
            else:
                kernel = _gpu_reduce_factory(self._functor, self._combine, from_dtype(dtype))
                combinekernel = _gpu_reduce_factory(self._combine, self._combine, from_dtype(dtype))
                self._cache[key] = kernels = (kernel, combinekernel)
            return kernels

        def _temp_dt(self, dtype):
            return self._temp_dtmap.get(dtype.char, dtype)

        def mktemp(self, arr, dtype=None):
            # ensure 1d array
            if arr.ndim != 1:
                arr = arr.reshape(arr.size)
            if dtype is None:
                dtype = self._temp_dt(arr.dtype)

            # Perform the reduction on the GPU
            blocksize = _NUMWARPS * CUDA_WARP_SIZE
            size_full = (arr.size // blocksize) * blocksize
            size_partial = arr.size - size_full
            full_blockct = min(size_full // blocksize, CUDA_WARP_SIZE * 2)

            # allocate size of partials array
            partials_size = full_blockct
            if size_partial:
                partials_size += 1

            return numba.cuda.device_array(shape=partials_size, dtype=dtype)

        def __call__(self, arr, size=None, res=None, partials=None, init=0, zero=0, stream=0):
            """Performs a full reduction.
            :param arr: A host or device array.
            :param partials: A temporary array to use as buffer, created with ``mktemp``.
            :param size: Optional integer specifying the number of elements in
                        ``arr`` to reduce. If this parameter is not specified, the
                        entire array is reduced.
            :param res: Optional device array into which to write the reduction
                        result to. The result is written into the first element of
                        this array. If this parameter is specified, then no
                        communication of the reduction output takes place from the
                        device to the host.
            :param init: Optional initial value for the reduction, the type of which
                        must match ``partials.dtype`` (default ``arr.type``).
            :param zero: Neutral initial value for the reduction, the type of which
                        must match ``partials.dtype``.
            :param stream: Optional CUDA stream in which to perform the reduction.
                        If no stream is specified, the default stream of 0 is
                        used.
            :return: If ``res`` is specified, ``None`` is returned. Otherwise, the
                    result of the reduction is returned.
            """
            cuda = numba.cuda

            # ensure 1d array
            if arr.ndim != 1:
                arr = arr.reshape(arr.size)

            # adjust array size
            if size is not None:
                arr = arr[:size]

            init = arr.dtype.type(init)  # ensure the right type

            # return `init` if `arr` is empty
            if arr.size < 1:
                return init

            # Perform the reduction on the GPU
            blocksize = _NUMWARPS * CUDA_WARP_SIZE
            size_full = (arr.size // blocksize) * blocksize
            size_partial = arr.size - size_full
            full_blockct = min(size_full // blocksize, CUDA_WARP_SIZE * 2)

            # allocate size of partials array
            partials_size = full_blockct
            if size_partial:
                partials_size += 1
            if partials is None:
                partials = self.mktemp(arr)

            kernel, combinekernel = self._compile(partials.dtype)

            if size_full:
                # kernel for the fully populated threadblocks
                kernel[full_blockct, blocksize, stream](
                    arr[:size_full],
                    partials[:full_blockct],
                    init,
                    True,
                    zero,
                )

            if size_partial:
                # kernel for partially populated threadblocks
                kernel[1, size_partial, stream](
                    arr[size_full:],
                    partials[full_blockct:],
                    init,
                    not full_blockct,
                    zero,
                )

            if partials.size > 1:
                # finish up
                combinekernel[1, partials_size, stream](partials, partials, init, False, zero)

            # handle return value
            if res is not None:
                res[:1].copy_to_device(partials[:1], stream=stream)
                return
            else:
                return partials[0]

    cuda_reduce = Reduce

    @cuda_reduce
    def cuda_sum2(a, b):
        return a * a + b

    def cuda_free_mem():
        return numba.cuda.get_current_device().primary_context.get_memory_info().free

    def cuda_total_mem():
        return numba.cuda.get_current_device().primary_context.get_memory_info().total
