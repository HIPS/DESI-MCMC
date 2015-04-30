import numpy as np
import pylab as plt
from tractor import *
from tractor.galaxy import *
from tractor.sersic import *

# size of image
W,H = 40,40

# PSF size
psfsigma = 1.

# per-pixel noise
noisesigma = 0.01

# create tractor.Image object for rendering synthetic galaxy
# images
tim = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)) / (noisesigma**2),
                    psf=NCircularGaussianPSF([psfsigma], [1.]))

sources = [ ExpGalaxy(PixPos(10,10), Flux(10.), GalaxyShape(3., 0.5, 45.)),
            CompositeGalaxy(PixPos(10,30),
                            Flux(10.), EllipseE(3., 0.5, 0.),
                            Flux(10.), EllipseE(3., 0., -0.5)),
            PointSource(PixPos(20,20), Flux(10.)),
            SersicGalaxy(PixPos(30,10), Flux(10.),
                         EllipseESoft(1., 0.5, 0.5), SersicIndex(3.)),
            FixedCompositeGalaxy(PixPos(30,30), Flux(10.), 0.8,
                                 EllipseE(2., 0., 0.),EllipseE(1., 0., 0.)) ]

tractor = Tractor([tim], sources)

mod = tractor.getModelImage(0)

# Plot
plt.clf()
plt.imshow(np.log(mod + noisesigma),
                   interpolation='nearest', origin='lower', cmap='gray')
plt.title('Galaxies')
plt.savefig('7.png')

