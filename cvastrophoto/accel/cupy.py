from __future__ import absolute_import

from .config import cupy_cuda, numba_cuda


if cupy_cuda and numba_cuda:
    try:
        import cupy.fft
        import cupy.cuda
        with_cupy = True
    except ImportError:
        with_cupy = False
else:
    with_cupy = False

if with_cupy and not cupy.cuda.is_available():
    with_cupy = False


if with_cupy:
    from cvastrophoto.util.vectorize import in_cuda_pool, register_oom_cleanup

    @register_oom_cleanup
    def cupy_oom_cleanup():
        pool = cupy.get_default_memory_pool()
        pool.free_all_blocks()
else:
    # no-op
    def cupy_oom_cleanup():
        pass
