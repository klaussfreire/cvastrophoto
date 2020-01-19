# cython: infer_types=True
from __future__ import absolute_import

import numpy
import cython
import logging
import scipy.ndimage

from skimage import color, measure

from ..base import PerChannelRop


if not cython.compiled:
    globals().update(dict(fabs=abs))


logger = logging.getLogger(__name__)


class DiffusionRop(PerChannelRop):

    T = 0.0025
    L = 0.15
    Lspeck = 1.0
    R = 2
    thr = 3
    steps = 40
    dt = 1.0/7
    prefilter_size = 1.0
    mask = True

    def estimate_noise(self, data):
        hfc = scipy.ndimage.white_tophat(data, int(self.prefilter_size * self.R))

        mask = hfc <= ((numpy.average(hfc) + numpy.max(hfc)) * 0.5)

        L = numpy.std(hfc[mask])

        return L

    def process_channel(self, data, detected=None):
        if data.dtype.char != 'f':
            data = data.astype(numpy.float32)

        L = self.estimate_noise(data)

        dscale = data.max() / 255.0
        if dscale > 0:
            T = dscale
        else:
            T = 1

        T *= self.T
        L *= self.L

        if not self.mask:
            T = 0

        data = Denoise(L, self.Lspeck, accelerated=False).modifiedPMdiffusion(
            data, self.steps, self.R, T, self.thr, self.dt)
        return data


@cython.cfunc
@cython.locals(gradient=cython.double, L=cython.double, g2=cython.double)
@cython.returns(cython.double)
@cython.nogil
def g(L,gradient):
    g2 = fabs(gradient) / L
    return (1.0 / (1.0 + g2*g2))


@cython.cclass
class Denoise(object):

    cython.declare(
        L=cython.double,
        Lspeck=cython.double,
        accelerated=cython.bint,
    )

    def __init__(self, L, Lspeck, accelerated=False):
        self.L = L
        self.Lspeck = Lspeck
        self.accelerated = accelerated

    def modifiedPMdiffusion(self, noisy, steps, nradius, T, thr, dt):
        ut=noisy

        logger.info("Starting %d diffusion iterations", steps)
        for i in range(steps):
            maskt, evolvedut = self.binarymask(ut, nradius, T, thr)

            logger.debug("Starting diffusion iteration %d/%d", i, steps)
            ut, changed = self.iteration(maskt, evolvedut, dt)
            logger.debug("Finished diffusion iteration %d/%d", i, steps)

            if not changed:
                break

        logger.info("Finished %d diffusion iterations", steps)

        return ut

    @cython.locals(
        pevolved='float[:,:]', y=int, x=int, ystart=int, xstart=int, yend=int, width=int, xend=int, nradius=int, pmask='unsigned char[:,:]',
        pnoisy='float[:,:]', i=int, j=int, m=int, u=cython.double, T=cython.double, thr=int, n0sum=cython.double, n0total=int)
    def binarymask(self, noisy, nradius, T, thr):
        # image domain
        ystart = 0
        yend = noisy.shape[0]
        xstart = 0
        xend = noisy.shape[1]
        width = 2 * nradius + 1
        if T > 0:
            mask = numpy.zeros(shape=noisy.shape, dtype=numpy.uint8)
        else:
            mask = numpy.ones(shape=noisy.shape, dtype=numpy.uint8)
        pmask = mask
        pnoisy = noisy

        if T > 0:
            logger.debug("Computing corrupted pixels")
            with cython.nogil:
                for y in range(ystart, yend-width):
                    for x in range(xstart, xend-width):
                        m = 0
                        u=pnoisy[y+nradius, x+nradius]
                        for i in range(y, y+width):
                            for j in range(x, x+width):
                                if fabs(pnoisy[i,j] - u) > T:
                                    m += 1

                        if m>thr:
                            pmask[y+nradius, x+nradius] = 1


        pevolved = evolved = noisy.copy()
        if self.accelerated: 
            logger.debug("Computed corrupted pixels")
            logger.debug("accelerated method")
            with cython.nogil:
                for y in range(ystart, yend-width):
                    for x in range(xstart, xend-width):
                        if pmask[y+nradius, x+nradius]:
                            n0sum = 0
                            n0total = 0
                            for i in range(y, y+width):
                                for j in range(x, x+width):
                                    if not pmask[i,j]:
                                        n0sum += pnoisy[i,j]
                                        n0total += 1
                            if n0total:
                                pevolved[y+nradius, x+nradius] = n0sum/n0total
            logger.debug("Computed accelerated method") 

        return mask, evolved

    @cython.locals(
        pu='float[:,:]', pnu='float[:,:]', pc='unsigned char[:,:]',
        dt=cython.double, y=int, x=int, ysize=int, xsize=int, changed=int,
        puyx=cython.double, dn=cython.double, de=cython.double, dne=cython.double,
        dse=cython.double, dse=cython.double, ds=cython.double, dw=cython.double,
        dsw=cython.double, dnw=cython.double, l=cython.double)
    def iteration(self, c, u, dt):
        pu = u
        pnu = nu = u.copy() 
        pc = c
        L = self.L
        ysize, xsize = u.shape[0:2]
        changed = 0

        with cython.nogil: 
            for y in range(1, ysize-1):
                for x in range(1, xsize-1):
                    if not pc[y,x]:
                        continue
                    changed += 1
                    puyx = pu[y,x]
                    dn = pu[y-1,x] - puyx
                    de = pu[y,x+1] - puyx
                    dne = pu[y-1,x+1] - puyx
                    dse = pu[y+1,x+1] - puyx
                    ds = pu[y+1,x] - puyx
                    dw = pu[y,x-1] - puyx
                    dsw = pu[y+1,x-1] - puyx
                    dnw = pu[y-1,x-1] - puyx
                    if dn > 0 and de > 0 and ds > 0 and dw > 0 and dne > 0 and dse > 0 and dnw > 0 and dsw > 0:
                        # Local speckle
                        l = L * self.Lspeck
                    elif dn < 0 and de < 0 and ds < 0 and dw < 0 and dne < 0 and dse < 0 and dnw < 0 and dsw < 0:
                        # Local dark spot
                        l = L * self.Lspeck
                    else:
                        l = L
                    diffusivity = (
                        g(l,dn)*dn + g(l,de)*de + g(l,ds)*ds + g(l,dw)*dw + (g(l,dne)*dne
                        + g(l,dse)*dse + g(l,dsw)*dsw + g(l,dnw)*dnw)/2)
                    pnu[y,x] += dt*diffusivity
                #logger.debug("Processed row %d/%d", y, u.shape[0])

        logger.debug("Changed %d pixels", changed)
        return nu, changed

