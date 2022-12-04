from __future__ import absolute_import

import logging
import numpy as np
import scipy.ndimage as ndi

from ..cupy import with_cupy, cupy_oom_cleanup


logger = logging.getLogger(__name__)


if with_cupy:
    try:
        import cupy as cp
        import cupy.cuda.memory
        import cupyx.scipy.ndimage as cundi
    except ImportError:
        with_cupy = False

if with_cupy:
    from cvastrophoto.util.vectorize import in_cuda_pool

    def _median_filter(input, size=None, footprint=None, **kw):
        input = cp.asarray(input)
        footprint = cp.asarray(footprint) if footprint is not None else None
        return cundi.median_filter(input, size=size, footprint=footprint, **kw).get()

    def median_filter(input, size=None, footprint=None, **kw):
        req_mem = input.size * input.dtype.itemsize
        try:
            return in_cuda_pool(req_mem, _median_filter, input, size, footprint, **kw).get()
        except (MemoryError, cupy.cuda.memory.OutOfMemoryError):
            logger.warning("Out of memory during CUDA median filter, falling back to CPU")
            cupy_oom_cleanup()
            return ndi.median_filter(input, size=size, footprint=footprint, **kw)
else:
    median_filter = ndi.median_filter
