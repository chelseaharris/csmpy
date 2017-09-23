# A class to help deal with hydro output files - 
# mostly has useful tools built in
#

import numpy           as np
import glob

import Const           as C
import ChevKasen_tools as CKtools

def calc_av(arr):
    return np.mean(arr)


def merge_cells(hi_ray, k, tol, lo_ray, i):
    # hi_ray: high resolution ray we're degrading
    # k     : index in the high resolution ray that we're on
    # tol   : fractional difference tolerance in density and temperature
    # lo_ray: low resolution ray we're writing into
    # i     : index in the low resolution ray that we're on
    GAM = 5./3

    Q1 = hi_ray.rho[k]
    d1 = abs(hi_ray.rho[k:] - Q1)/Q1
    cand = np.where( d1 > tol )[0]
    k_m1 = hi_ray.size()-k-1 if len(cand)==0 else cand[0]

    Q2 = hi_ray.T_gas[k]
    d2 = abs(hi_ray.T_gas[k:] - Q2)/Q2
    cand = np.where(  d2 > tol )[0]
    k_m2 = hi_ray.size()-k-1 if len(cand)==0 else cand[0]

    k_m = min(k_m1,k_m2) + k

    kslice = slice(k,k_m) if k_m>k else k

    lo_ray.r1[i] = hi_ray.r1[k]
    lo_ray.r2[i] = hi_ray.r2[k_m]

    lo_ray.rho  [i] = calc_av(hi_ray.rho  [kslice])
    lo_ray.T_gas[i] = calc_av(hi_ray.T_gas[kslice])
    lo_ray.v    [i] = calc_av(hi_ray.v    [kslice])

    p_gas = C.K_B/C.M_P*lo_ray.rho[i]*lo_ray.T_gas[i]
    lo_ray.p_gas[i] = p_gas
    # u_gas = p_gas/(gam_fac - 1)
    lo_ray.u_gas[i] = p_gas/(GAM - 1)
    # sqrt(gamfac_ * grid->z[i].p_gas/grid->z[i].rho)
    lo_ray.cs   [i] = np.sqrt( GAM * p_gas/lo_ray.rho[i] )

    # if we're not at the end of lo_ray or hi_ray, keep merging
    knext = k_m if k_m > k else k+1
    if i+1 < lo_ray.size() and knext < hi_ray.size():
        i = merge_cells(hi_ray, knext, tol, lo_ray, i+1)

    return i
        

class Ray(object):
    """
    Member variables of this class: time, r1, r2, r, rho, T_gas, p_gas, v, u_gas, cs, n_e
    """

    def __init__(self, a_props, a_time=-1, a_n_e = [], a_n_I = [], u_per_cc = False, a_r=[]):
        """
        INPUTS
        a_props  : 2D array of properties. Rows are
                   0: r1
                   1: r2
                   2: rho
                   3: T_gas
                   4: p_gas
                   5: v
                   6: u_gas (assumed in erg/gram unless u_per_cc set to True)
                   (7): cs  -- optional
        a_n_e    : electron density per cell -- optional input
        a_n_I    : ion density per cell -- optional input
        u_per_cc : whether units of a_props[6] are erg/cc (True) or erg/g (False)
        a_r      : average radius to use per cell 
        """

        self.time = a_time 

        self.r1    = a_props[0]
        self.r2    = a_props[1]

        if np.any( (self.r1-self.r2) == 0.0 ):
            raise RuntimeError('Increased precision in the ray file is required.');

        if len(a_r)!=len(self.r1):
            self.r = CKtools.calc_zone_r(self.r1,self.r2)
        else:
            self.r = np.array(a_r)

        self.rho   = a_props[2]
        self.T_gas = a_props[3]
        self.p_gas = a_props[4]
        self.v     = a_props[5]

        if not u_per_cc:
            self.u_gas = a_props[6] * self.rho
        else:
            self.u_gas = a_props[6]

        if len(a_props)==8:
            self.cs = a_props[7]
        else:
            self.cs = np.zeros(len(self.r1))

        self.n_e = a_n_e
        self.n_I = a_n_I



    @classmethod
    def from_file(cls, model_dir, ray_num, ray_time=-1):
    # INPUTS:
    # model_dir  : (string)  directory the ray is in
    # ray_num    : (int)     the number of the ray
    # ray_time   : (float)   model time corresponding to this ray, in seconds
    #
        ray_fn = model_dir+'/ray_{0:05d}'.format(ray_num);
        assert len(glob.glob(ray_fn))==1;
  
        try:        
            this_ray = Ray( np.loadtxt(ray_fn, usecols=[0,1,2,3,4,5,6,10], unpack=True), ray_time );
        except IndexError:
            this_ray = Ray( np.loadtxt(ray_fn, usecols=[0,1,2,3,4,5,6], unpack=True), ray_time );

        return this_ray


    def __getitem__(self,i):
        props = [ self.r1[i], self.r2[i], self.rho[i], self.T_gas[i], self.p_gas[i], self.v[i], self.u_gas[i], self.cs[i] ]
        return Ray( props, self.time, 
                    (self.n_e[i] if len(self.n_e)>0 else []), 
                    (self.n_I[i] if len(self.n_I)>0 else []), 
                    True, self.r[i] )



    def __len__(self):
        return len(self.r1)
 
    def size(self):
    # Number of cells in the ray
        return len(self.r1);



    def copy(self, i0=0, i1=0 ):
        if i1==0: 
            i1 = self.size()

        if i1>= self.size(): i1=self.size()
        if i0<0               : i0=0

        return Ray( [ self.r1   [i0:i1], self.r2   [i0:i1], self.rho[i0:i1], 
                      self.T_gas[i0:i1], self.p_gas[i0:i1], self.v  [i0:i1], 
                      self.u_gas[i0:i1], self.cs   [i0:i1]                   ],
                    self.time, 
                    (self.n_e[i0:i1] if self.n_e!=None else self.n_e), 
                    (self.n_I[i0:i1] if self.n_I!=None else self.n_I), 
                    u_per_cc=True )
        

    def cell_mass(self,i):    
    # Returns the mass in cell i
        assert i>-1 and i<self.size();
        return 4*np.pi/3 * self.rho[i] * ((self.r2[i])**3 - (self.r1[i])**3);


    def mass_btwn(self,i1,i2,verbose=False):
        assert self.size() > 0;
        assert i1 > -1;
        assert i1 < self.size();
        assert i2 > -1;
        assert i2 < self.size();

        # Make sure i1 < i2                                                                                                                                                                            
        if i2<i1:
            if verbose: print("Switching i1 and i2 in Ray.mass_btwn");
            i_min = i2;
            i2 = i1;
            i1 = i_min;

        if i1==i2:
            if verbose: print("Ray.mass_btwn() called on single cell");
            return self.cell_mass(i1)
        else:
            return sum( [ self.cell_mass(i) for i in range(i1,i2) ] );


    def calc_N_e(self, i=[]):
        if i==[]:
            i = range(len(self.r1))
        return sum(self.n_e[i] * (self.r2[i] - self.r1[i]))

    def calc_N(self, i=[]):
        if i==[]:
            i = range(len(self.r1))

        n = self.n_e[i] + self.n_I[i]
        dr = self.r2[i] - self.r1[i]
        return sum(n * dr)


    def cell_at_v(self,vel,tol=-1):
    # Returns the index of the cell with velocity closest to 'vel'. 
    # If a tolerance 'tol' is set, then there is the requirement 
    # that the cell velocity is within 'tol' of 'vel', and -1 is
    # returned if no cell meets that criterion.
    # 
        diff = abs(self.v - vel);
        if tol==-1:
            return np.argmin( diff );
        else:
            candidate = np.argmin( diff );
            if diff[candidate]<=tol: 
                return candidate;
            else:
                print("No cell found near desired velocity within tolerance.");
                return self.size();


    def cell_at_r(self,rad,tol=-1):
    # Same as cell_at_v but with radius
        diff = abs(self.r - rad);
        if tol==-1:
            return np.argmin( diff );
        else:
            candidate = np.argmin( diff );
            if diff[candidate]<=tol: 
                return candidate;
            else:
                print("No cell found near desired radius within tolerance.");
                return self.size();


    def rmax(self):
        return max(self.r);

    def rmin(self):
        return min(self.r);

    def cell_volume(self,i=None):
        """
        Returns the volume of each cell
        """
        if i==None:
            return 4*np.pi/3 * ( self.r2**3 - self.r1**3 )
        else:
            return 4*np.pi/3 * ( self.r2[i]**3 - self.r1[i]**3 )


    def volume_btwn(self, i0=0, i1=0):
        """
        Returns the volume enclosed between indeces i0 and i1
        """
        if i1==0:
            i1 = self.size()-1

        return 4*np.pi/3 * (self.r2[i1]**3 - self.r1[i0]**3)


    def set_n_e(self, a_n_e=[]):
        if len(a_n_e) == 0:
            self.n_e = self.rho / C.M_P # fully ionized hydrogen
        else:
            assert len(a_n_e) == len(self.r1)
            self.n_e = a_n_e


    def set_n_I(self, a_n_I=[]):
        if len(a_n_I) == 0:
            self.n_I = self.rho / C.M_P  # fully ionized hydrogen
        else:
            assert len(a_n_I) == len(self.r1)
            self.n_I = a_n_I

    def get_n(self):
        assert self.n_I != None 
        assert self.n_e != None

        return self.n_e + self.n_I


    def write_as_initmod(self, outname='WOOOOOO.mod'):
        if self.time<0: 
            print('Can\'t write the ray until it has a legit positive time in seconds.')
            return self.time

        header = "{:d} {:d} {:d} {:f} {:e} {:d}\n".format(1, self.size(), 1, 0.1, self.time, 1)
        warr = np.vstack((self.r1, self.r2, self.rho, self.p_gas, self.v, self.v, np.zeros(self.size()))).T # What is it good for? 
        with open(outname, 'w') as wf:
            np.savetxt(wf, warr, header=header, comments='')
        

    def degrade(self, N_lowres, tol):
        # Initialize low resolution array
        low_props = np.zeros((8,N_lowres))
        low_props[1] = np.arange(N_lowres) + 1
        low_r = np.zeros(N_lowres)
        low_ray = Ray(low_props, a_r = low_r)
    
        # Populate low-res ray with merged hi-res cells
        i_last = merge_cells(self, 0, tol, low_ray, 0)

        low_ray = low_ray[:i_last]
        # Let me know if I gave too many cells
        if i_last+1 < N_lowres:
            print('Only needed {} cells of the {} alotted'.format(i_last+1, N_lowres))
        
        # Let me know if I gave too few cells
        dr = abs(low_ray.r2[-1] - self.r2[-1])/self.r2[-1]
        if dr>1e-16: 
            print('could not capture full ray -- increase tol or N_lowres, or trim the ray -- dr={:e}'.format(dr))
    
        # Calculate the average radius of low-res cells
        low_ray.r = CKtools.calc_zone_r(low_ray.r1, low_ray.r2)
    
        return low_ray

    def blind_degrade(self, N_lowres):
        # Initialize low resolution array
        low_props = np.zeros((8,N_lowres))
        low_props[1] = np.arange(N_lowres) + 1
        low_r = np.zeros(N_lowres)
        lo_ray = Ray(low_props, a_r = low_r)
    
        # Populate low-res ray with merged hi-res cells
        GAM = 5./3
    
        k_bin = np.linspace(0,self.size()-1,lo_ray.size()+1,dtype=int)

        for i in range(lo_ray.size()):
            k = slice(k_bin[i], k_bin[i+1])
        
            lo_ray.r1[i] = self.r1[k_bin[i  ]]
            lo_ray.r2[i] = self.r2[k_bin[i+1]]
        
            lo_ray.rho  [i] = calc_av(self.rho  [k])
            lo_ray.T_gas[i] = calc_av(self.T_gas[k])
            lo_ray.v    [i] = calc_av(self.v    [k])
        
            p_gas = C.K_B/C.M_P*lo_ray.rho[i]*lo_ray.T_gas[i]
            lo_ray.p_gas[i] = p_gas
            # u_gas = p_gas/(gam_fac - 1)
            lo_ray.u_gas[i] = p_gas/(GAM - 1)
            # sqrt(gamfac_ * grid->z[i].p_gas/grid->z[i].rho)
            lo_ray.cs   [i] = np.sqrt( GAM * p_gas/lo_ray.rho[i] )
    
        # Calculate the average radius of low-res cells
        lo_ray.r = CKtools.calc_zone_r(lo_ray.r1, lo_ray.r2)
    
        return lo_ray
    

class Cell(object):
    """
    Cell (of a Ray)
    Has all the same variables as the Ray class, plus 
    """
    def __init__(self, a_r1=0.0, a_r2=0.0, a_rho=0.0, a_T_gas=0.0, 
                 a_p_gas=0.0, a_v=0.0, a_u_gas=0.0, a_c_s=0.0, a_n_e=-1, a_i=-1   ):
        """
        INPUTS
        a_r1  : (float) lower radial bound of cell
        a_r2  : (float) upper ...
        a_rho : (float) den
        """
        if a_r1==a_r2 and a_r1!=0.0: 
            print( "Warning: zero-volume cell")

        self.r1 = a_r1
        self.r2 = a_r2
        self.r  = CKtools.calc_zone_r(r1,r2)
        self.T_gas = a_T_gas
        self.p_gas = a_p_gas
        self.v     = a_v
        self.u_gas = a_u_gas
        self.cs    = a_c_s
        self.n_e   = a_n_e
        self.i     = a_i


    def volume():
        return 4*np.pi/3 * (self.r2**3 - self.r1**3)

    def mass():
        return self.volume() * self.rho

