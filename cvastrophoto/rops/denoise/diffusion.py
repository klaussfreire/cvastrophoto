# cython: infer_types=True

import numpy
import cython
import logging
from skimage import color, measure
from ..base import PerChannelRop


if not cython.compiled:
    globals().update(dict(fabs=abs))


logger = logging.getLogger(__name__)


class DiffusionRop(PerChannelRop):

    def __init__(self, raw=None, T=0.0025, L=1.0/180, R=2, thr=3, steps=20, dt=1.0/7):
        super(DiffusionRop,self).__init__(raw)
        self.T = T
        self.L = L
        self.R = R
        self.thr = thr
        self.dt = dt
        self.steps = steps

    def process_channel(self,data,detected=None):
        if data.dtype.char != 'f':
            data = data.astype(numpy.float32)
        return Denoise(self.L, accelerated=False).modifiedPMdiffusion(
            data, self.steps, self.R, self.T, self.thr, self.dt)
 
     
@cython.cfunc
@cython.locals(gradient=cython.double, L=cython.double, g2=cython.double)
@cython.returns(cython.double)
@cython.nogil
def g(L,gradient):
    g2 = fabs(gradient)/L
    return (1.0/(1.0+ g2*g2))


@cython.cclass
class Denoise(object):

    cython.declare(
        L=cython.double,
        accelerated=cython.bint,
    )

    def __init__(self, L, accelerated=False):
        self.L= L
        self.accelerated = accelerated
        
    def modifiedPMdiffusion(self, noisy, steps, nradius, T, thr, dt):
        ut=noisy

        for i in range(steps):
            maskt, evolvedut = self.binarymask(ut, nradius, T, thr)
            ut, changed = self.iteration(maskt, evolvedut, dt)
            if not changed:
                break
        
        return ut

    @cython.locals(
        pevolved='float[:,:]', y=int, x=int, ystart=int, xstart=int, yend=int, width=int, xend=int, nradius=int, pmask='unsigned char[:,:]',
        pnoisy='float[:,:]', i=int, j=int, m=int, u=cython.double, T=cython.double, thr=int, n0sum=cython.double, n0total=int)
    def binarymask(self, noisy, nradius, T, thr):
        # image domain
        logger.info("Computing corrupted pixels")
        ystart = 0
        yend = noisy.shape[0]
        xstart = 0
        xend = noisy.shape[1]
        width = 2 * nradius + 1
        pmask = mask = numpy.zeros(shape=noisy.shape, dtype=numpy.uint8)
        pnoisy = noisy
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
            logger.info("Computed corrupted pixels")
            logger.info("accelerated method")
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
            logger.info("Computed accelerated method") 

        return mask, evolved

    @cython.locals(
        pu='float[:,:]', pnu='float[:,:]', pc='unsigned char[:,:]',
        dt=cython.double, y=int, x=int, ysize=int, xsize=int, changed=int,
        puyx=cython.double, dn=cython.double, de=cython.double, dne=cython.double,
        dse=cython.double, dse=cython.double, ds=cython.double, dw=cython.double,
        dsw=cython.double, dnw=cython.double)
    def iteration(self,c,u,dt):
        logger.info("Starting diffusion iteration")
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
                    diffusivity = (
                        g(L,dn)*dn + g(L,de)*de + g(L,ds)*ds + g(L,dw)*dw + (g(L,dne)*dne
                        + g(L,dse)*dse + g(L,dsw)*dsw + g(L,dnw)*dnw)/2)
                    pnu[y,x] += dt*diffusivity
                #logger.debug("Processed row %d/%d", y, u.shape[0])
         
        logger.debug("Changed %d pixels", changed)
        logger.info("Finished diffusion iteration")
        return nu, changed

