import numpy as np

class SrcParams():
    """ source parameter object


        Input:
          u : 2-d np.array holdin right ascension and declination
          b : Total brightness/flux.  Equal to 

               b = ell / (4 * pi d^2)

               where ell is the luminosity (in Suns) and d is the distance
               to the source (in light years)

          fluxes : python dictionary such that b['r'] = brightness value for 'r'
              band.  Note that this is essentially the expected number
              of photons to enter the lens and be recorded by a given band
              over the length of one exposure (typically 1.25^2 meters^2 size
              lens and 54 second exposure)

              This will be kept in nanomaggies (must be scaled by 
              image calibration and image specific gain). If this is 
              present, it takes priority over the combo of the next few parameters

          t : effective temperature of source (in Kelvin)
          ell : luminosity of source (in Suns)
          d : distance to source (in light years)
    """
    # define source parameter D_type
    src_dtype = [ ('a', 'u1'),
                  ('t', 'f4'),
                  ('b', 'f4'),
                  ('u', 'f4', (2,)), 
                  ('v', 'f4', (2,)),
                  ('theta', 'f4'),
                  ('phi', 'f4'),
                  ('sigma', 'f4'),
                  ('rho', 'f4'),
                  ('fluxes', 'f4', (5,)) ]

    def __init__(self,
                 u,
                 a      = None,
                 # star specific params
                 b      = None,
                 t      = None,
                 # galaxy specific params
                 v      = None,
                 theta  = None,
                 phi    = None,
                 sigma  = None,
                 rho    = None,
                 fluxes = None,
                 # extra, aux params
                 ell    = None,
                 d      = None,
                 header = None):

        ## binary indicator that source is a star (0) or galaxy (1)
        self.a = a

        ## star params
        self.u = u
        self.b = b
        self.t = t

        ## galaxy params
        self.v      = v        # location - different from star location
        self.theta  = theta    # mixture between exponential and devacalours galaxies
        self.phi    = phi      # rotation angle of the galaxy
        self.sigma  = sigma    # scale of the galaxy extent
        self.rho    = rho      # eccentricity = major/minor axis ratio
        self.fluxes = fluxes   # galaxy 5 band fluxes

        ## unused/extra params
        self.ell = ell
        self.d = d
        self.header = header

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.u, other.u) and self.b == other.b
        else:
            return False

    def __str__(self): 
        if self.a == 0: 
            return "StrSrc: u=(%2.2f, %2.2f), b=%.4g, t=%2.2f"%(self.u[0], self.u[1], self.b, self.t)
        elif self.a==1:
            return "GalSrc: u=(%2.2f, %2.2f), theta=%2.2f"%(self.u[0], self.u[1], self.theta)
        else: 
            return "NoType: u=(%2.2f, %2.2f)"%(self.u[0], self.u[1])

    def to_array(self):
        """ returns a structured array """
        src_array = np.zeros(1, dtype = SrcParams.src_dtype)
        src_array['a'][0] = self.a
        if self.a == 0:
            src_array['t'][0] = self.t
            src_array['b'][0] = self.b
            src_array['u'][0] = self.u
        elif self.a == 1:
            src_array['u'][0]      = self.u
            src_array['v'][0]      = self.v
            src_array['theta'][0]  = self.theta
            src_array['phi'][0]    = self.phi
            src_array['sigma'][0]  = self.sigma
            src_array['rho'][0]    = self.rho
            src_array['fluxes'][0] = np.array([self.fluxes[b] for b in ['u', 'g', 'r', 'i', 'z']])
        return src_array

    @staticmethod
    def init_obj(src_array):
        return SrcParams( 
            u = np.array(src_array['u'], dtype=np.float),
            a = src_array['a'],
            b = src_array['b'],
            t = src_array['t'],
            v = src_array['v'],
            theta  = src_array['theta'],
            phi    = src_array['phi'],
            sigma  = src_array['sigma'],
            rho    = src_array['rho'],
            fluxes = dict(zip(['u', 'g', 'r', 'i', 'z'], src_array['fluxes']))
            )

