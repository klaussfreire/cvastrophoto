from __future__ import absolute_import

import logging
import numpy as np
import scipy.ndimage as ndi

from .cupy import with_cupy, cupy_oom_cleanup


logger = logging.getLogger(__name__)


if with_cupy:
    try:
        import cupy as cp
        import cupy.cuda.memory
        import cupyx.scipy.ndimage as cundi
    except ImportError:
        with_cupy = False

def _content_mask(cp, ndi, luma, mask_sigma, mask_open, get):
    luma = cp.asarray(luma)
    luma_median = cp.median(luma)
    luma_std = cp.std(luma)
    luma_std = cp.std(luma[luma <= (luma_median + mask_sigma * luma_std)])
    content_mask = luma > (luma_median + mask_sigma * luma_std)
    if mask_open:
        content_mask = ndi.binary_opening(content_mask)
    if get:
        content_mask = content_mask.get()
    return content_mask

if with_cupy:
    from cvastrophoto.util.vectorize import in_cuda_pool

    def content_mask(luma, mask_sigma, mask_open):
        req_mem = luma.size * luma.dtype.itemsize * 2 + luma.size
        try:
            return in_cuda_pool(req_mem, _content_mask, cp, cundi, luma, mask_sigma, mask_open, True).get()
        except (MemoryError, cupy.cuda.memory.OutOfMemoryError):
            logger.warning("Out of memory during CUDA operation, falling back to CPU")
            cupy_oom_cleanup()
            return _content_mask(np, ndi, luma, mask_sigma, mask_open, False)
else:
    def content_mask(luma, mask_sigma, mask_open):
        return _content_mask(np, ndi, luma, mask_sigma, mask_open, False)
