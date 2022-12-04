from __future__ import absolute_import

import numpy as np
import logging

from ..cupy import cupy_oom_cleanup, with_cupy

try:
    from skimage.registration import phase_cross_correlation as _sk_phase_cross_correlation
    from skimage.registration._phase_cross_correlation import (
        _compute_error, _compute_phasediff,
    )
except ImportError:
    from skimage.feature import (
        register_translation as _sk_phase_cross_correlation,
        _compute_error, _compute_phasediff,
    )

logger = logging.getLogger(__name__)


if with_cupy:
    try:
        import cupy.fft
        import cupy.cuda
        import cupy as cp
    except ImportError:
        with_cupy = False


if with_cupy:
    from cvastrophoto.util.vectorize import in_cuda_pool
    from scipy.fft import fftfreq

    def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
        # if people pass in an integer, expand it to a list of equal-sized sections
        if not hasattr(upsampled_region_size, "__iter__"):
            upsampled_region_size = [upsampled_region_size, ] * data.ndim
        else:
            if len(upsampled_region_size) != data.ndim:
                raise ValueError("shape of upsampled region sizes must be equal "
                                "to input data's number of dimensions.")

        if axis_offsets is None:
            axis_offsets = [0, ] * data.ndim
        else:
            if len(axis_offsets) != data.ndim:
                raise ValueError("number of axis offsets must be equal to input "
                                "data's number of dimensions.")

        im2pi = 1j * 2 * np.pi

        dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

        for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
            kernel = ((np.arange(ups_size) - ax_offset)[:, None]
                    * fftfreq(n_items, upsample_factor))
            kernel = cp.asarray(kernel)
            kernel = cp.exp(-im2pi * kernel)
            # use kernel with same precision as the data
            kernel = kernel.astype(data.dtype, copy=False)

            # Equivalent to:
            #   data[i, j, k] = kernel[i, :] @ data[j, k].T
            data = cp.tensordot(kernel, data, axes=(1, -1))
        return data

    def _cupy_phase_cross_correlation(
        reference_image, moving_image,
        upsample_factor=1, space="real",
        return_error=True, reference_mask=None,
        moving_mask=None, overlap_ratio=0.3,
        normalization="phase",
    ):
        reference_image = cp.asarray(reference_image)
        moving_image = cp.asarray(moving_image)

        # assume complex data is already in Fourier space
        if space.lower() == 'fourier':
            src_freq = reference_image
            target_freq = moving_image
        # real data needs to be fft'd.
        elif space.lower() == 'real':
            src_freq = cupy.fft.fftn(reference_image)
            target_freq = cupy.fft.fftn(moving_image)
        else:
            raise ValueError('space argument must be "real" of "fourier"')

        # Whole-pixel shift - Compute cross-correlation by an IFFT
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        float_dtype = image_product.real.dtype
        if normalization == "phase":
            eps = np.finfo(float_dtype).eps
            image_product /= cp.maximum(cp.abs(image_product), 100 * eps)
        elif normalization is not None:
            raise ValueError("normalization must be either phase or None")
        cross_correlation = cupy.fft.ifftn(image_product)

        # Locate maximum
        maxima = np.unravel_index(cp.argmax(cp.abs(cross_correlation)).get(),
                                cross_correlation.shape)
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

        shifts = np.stack(maxima).astype(float_dtype, copy=False)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

        if upsample_factor == 1:
            if return_error:
                src_amp = cp.sum(cp.real(src_freq * src_freq.conj()))
                src_amp /= src_freq.size
                target_amp = cp.sum(cp.real(target_freq * target_freq.conj()))
                target_amp /= target_freq.size
                CCmax = cross_correlation[maxima]
        # If upsampling > 1, then refine estimate with matrix multiply DFT
        else:
            # Initial shift estimate in upsampled grid
            upsample_factor = np.array(upsample_factor, dtype=float_dtype)
            shifts = np.round(shifts * upsample_factor) / upsample_factor
            upsampled_region_size = np.ceil(upsample_factor * 1.5)
            # Center of output array at dftshift + 1
            dftshift = np.fix(upsampled_region_size / 2.0)
            # Matrix multiply DFT around the current shift estimate
            sample_region_offset = dftshift - shifts*upsample_factor
            cross_correlation = _upsampled_dft(image_product.conj(),
                                            upsampled_region_size,
                                            upsample_factor,
                                            sample_region_offset).conj()
            # Locate maximum and map back to original pixel grid
            dftmaxima = np.unravel_index(cp.argmax(cp.abs(cross_correlation)).get(),
                                    cross_correlation.shape)

            maxima = np.stack(dftmaxima).astype(float_dtype, copy=False)
            maxima -= dftshift

            shifts += maxima / upsample_factor

            if return_error:
                src_amp = cp.sum(cp.real(src_freq * src_freq.conj()))
                target_amp = cp.sum(cp.real(target_freq * target_freq.conj()))

            # Keep it after everything else as it syncs with the GPU
            CCmax = cross_correlation[dftmaxima]

        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        for dim in range(src_freq.ndim):
            if shape[dim] == 1:
                shifts[dim] = 0

        if return_error:
            # Redirect user to masked_phase_cross_correlation if NaNs are observed
            if cp.isnan(CCmax) or cp.isnan(src_amp) or cp.isnan(target_amp):
                raise ValueError(
                    "NaN values found, please remove NaNs from your "
                    "input data or use the `reference_mask`/`moving_mask` "
                    "keywords, eg: "
                    "phase_cross_correlation(reference_image, moving_image, "
                    "reference_mask=~np.isnan(reference_image), "
                    "moving_mask=~np.isnan(moving_image))")

            CCmax = CCmax.get()
            return shifts,  _compute_error(CCmax, src_amp.get(), target_amp.get()),\
                _compute_phasediff(CCmax)
        else:
            return shifts

    def phase_cross_correlation(reference_image, moving_image, **kw):
        # images must be the same shape
        if reference_image.shape != moving_image.shape:
            raise ValueError("images must be same shape")

        if kw.get('reference_mask') is not None or kw.get('moving_mask') is not None:
            return _sk_phase_cross_correlation(reference_image, moving_image, **kw)

        def do_cuda(reference_image, moving_image):
            return _cupy_phase_cross_correlation(reference_image, moving_image, **kw)

        try:
            return in_cuda_pool(
                # complex128 * 2
                reference_image.size * 16 * 2,
                do_cuda, reference_image, moving_image,
            ).get()
        except (MemoryError, cupy.cuda.memory.OutOfMemoryError):
            logger.warning("Out of memory during CUDA correlation, falling back to CPU")
            cupy_oom_cleanup()
            return _sk_phase_cross_correlation(reference_image, moving_image, **kw)

else:
    phase_cross_correlation = _sk_phase_cross_correlation
