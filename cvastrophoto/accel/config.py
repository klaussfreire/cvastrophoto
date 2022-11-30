import os

numba_cuda = os.environ.get('NUMBA_CUDA', 'yes') != 'no'
cupy_cuda = os.environ.get('CUPY_CUDA', 'yes') != 'no'
any_cuda = os.environ.get('CV_CUDA', 'yes') != 'no'

if not any_cuda:
    numba_cuda = False
    cupy_cuda = False
