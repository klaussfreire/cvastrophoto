from __future__ import absolute_import

import numpy as np
import logging

from skimage.transform import warp as sk_warp

from ..cupy import with_cupy, cupy_oom_cleanup


logger = logging.getLogger(__name__)


if with_cupy:
    try:
        import cupy as cp
        import cupy.cuda.memory
        import cupyx.scipy.ndimage as ndi
    except ImportError:
        with_cupy = False


if with_cupy:
    from cvastrophoto.util.vectorize import in_cuda_pool
    from skimage.transform._warps import HOMOGRAPHY_TRANSFORMS

    def _clip_warp_output(input_image, output_image, mode, cval):
        min_val = cp.min(input_image)
        if cp.isnan(min_val):
            # NaNs detected, use NaN-safe min/max
            min_func = cp.nanmin
            max_func = cp.nanmax
            min_val = min_func(input_image)
        else:
            min_func = cp.min
            max_func = cp.max
        max_val = max_func(input_image)

        # Check if cval has been used such that it expands the effective input
        # range
        preserve_cval = (
            mode == 'constant'
            and not min_val <= cval <= max_val
            and min_func(output_image) <= cval <= max_func(output_image)
        )

        # expand min/max range to account for cval
        if preserve_cval:
            # cast cval to the same dtype as the input image
            cval = input_image.dtype.type(cval)
            min_val = min(min_val, cval)
            max_val = max(max_val, cval)

        cp.clip(output_image, min_val, max_val, out=output_image)

    def warp(image, inverse_map, order=None,
            mode='constant', cval=0., clip=True, preserve_range=False, output_shape=None,
            **kw):

        matrix = coords = None

        if kw is None and image.size and image.ndim == 2 or order in (0, 1, 2, 3, 4, 5):

            matrix = None

            if isinstance(inverse_map, np.ndarray) and inverse_map.shape == (3, 3):
                # inverse_map is a transformation matrix as numpy array
                matrix = inverse_map

            elif isinstance(inverse_map, HOMOGRAPHY_TRANSFORMS):
                # inverse_map is a homography
                matrix = inverse_map.params

            elif isinstance(inverse_map, np.ndarray):
                # inverse_map is directly given as coordinates
                coords = inverse_map

        if matrix is not None:
            def _warp(image, matrix):
                if matrix.shape[0] == matrix.shape[1]:
                    # Apply coordinate transform since ndimage uses y,x instead of x,y
                    S = np.identity(matrix.shape[0])
                    S[[0,1]] = S[[1,0]]

                matrix = cp.asanyarray(matrix)
                image = cp.asanyarray(image)
                warped = ndi.affine_transform(
                    image, matrix,
                    output_shape=output_shape, order=order, mode=mode, cval=cval)

                if clip:
                    _clip_warp_output(image, warped, mode, cval)

                return cp.asnumpy(warped)
            args = (image, matrix)
            req_mem = image.size * image.dtype.itemsize * 2
        elif coords is not None and output_shape is None:
            def _warp(image, coords):
                coords = cp.asanyarray(coords)
                image = cp.asanyarray(image)
                warped = ndi.map_coordinates(
                    image, coords,
                    order=order, mode=mode, cval=cval)

                if clip:
                    _clip_warp_output(image, warped, mode, cval)

                return cp.asnumpy(warped)
            args = (image, coords)
            req_mem = image.size * image.dtype.itemsize * 2 + coords.size * coords.dtype.itemsize
        else:
            _warp = None

        if _warp is not None:
            try:
                return in_cuda_pool(req_mem, _warp, *args).get()
            except (MemoryError, cupy.cuda.memory.OutOfMemoryError):
                logger.warning("Out of memory during CUDA transform, falling back to CPU")
                cupy_oom_cleanup()

        return sk_warp(
            image, inverse_map,
            order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range,
            output_shape=output_shape,
            **kw)

else:
    warp = sk_warp
