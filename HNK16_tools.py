###
# Functions for generating light-curves for the HNK16 fiducial family
###

import numpy             as np
from EvolvedModelClass import EvolvedModel as EvMod
import Const as C
import ChevKasen_tools as ckt

class ModLC(object):
    """
    The basic storage unit for a light-curve
    """
    
    def __init__(self, times, lums, nu=[], i_lo=[], i_hi=[] ):
        """
        times  : (float[]) times sampled
        lums   : (float[]) luminosity at each time
        i_lo   : (int[]  ) index of calculation lower bound
        i_hi   : (int[]  ) index of calculation upper bound
        """

        i_sorted = np.argsort(times)
        #i_sorted = np.arange(len(times))
        self.t    = np.array(times[i_sorted]) if len(i_sorted)>1 else times
        self.lum  = np.array(lums [i_sorted]) if len(lums    )>1 else lums

        if len(nu)>0:
            self.set_nu(nu)
        if len(i_lo)>0:
            self.set_ilo(i_lo)
        if len(i_hi)>0:
            self.set_ihi(i_hi)

        self._time_sort()


    def set_nu(self, nu):
        self.nu = np.array(set_nu)

    def set_ilo(self, i_lo):
        self.i_lo = np.array(i_lo)

    def set_ihi(self, i_hi):
        self.i_hi = np.array(i_hi)

    def _time_sort(self):
        i_sorted = np.argsort(self.t)

        self.t = self.t[i_sorted]
        self.lum = self.lum[i_sorted]
        
        try:
            self.nu = self.nu[i_sorted]
        except AttributeError:
            pass
        try:
            self.i_lo = self.i_lo[i_sorted]
        except AttributeError:
            pass
        try:
            self.i_hi = self.i_hi[i_sorted]
        except AttributeError:
            pass


    @classmethod 
    def from_file(cls, mod, rfn='synch_LCs.fwd.txt'):
        mat = np.loadtxt(rfn,usecols=[0,2,3,4,5,6])
        if mod not in mat[:,0]: 
            print('Did not find {0} in file {1}'.format(mod,rfn))
            return ModLC( [], [] )
        else:
            its_mat = mat[ mat[:,0] == mod ] 
            return ModLC( its_mat[:,1], its_mat[:,3], its_mat[:,2], its_mat[:,4], its_mat[:,5] )


    def size(self):
        return len(self.t)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        if type(i) == int:   return ModLC( [self.t[i]], [self.lum[i]] )
        else: return ModLC( self.t[i], self.lum[i] )


    def interp(self, new_times, interp_log=False, left=np.nan, right=np.nan):
        """
        Interpolate to get better time information. Interpolation will be linear -- use interp_log to interpolate 
        linearly in log-log space
        Does not overwrite existing LC object
        """
        if interp_log: 
            interp_t = np.log10(new_times)
            orig_t   = np.log10(self.t   )
            orig_lum = np.log10(self.lum )
        else:
            interp_t = new_times
            orig_t   = self.t
            orig_lum = self.lum

        new_lums = np.interp( interp_t, orig_t, orig_lum, left=left, right=right ) # this is log-L if interp_log
        if interp_log:
            new_lums = 10**new_lums 

        return  ModLC( new_times, new_lums )


    def scale( self, fact=-1 ):
        # default is normalize to peak
        if fact==-1:
            self.lum /= max(self.lum)
        else:
            self.lum /= fact


    # Peak time and luminosity
    def i_peak( self ):     return np.argmax(self.lum)
    def t_peak( self ):     return self.t[ np.argmax(self.lum) ]
    def L_peak( self ):     return max(self.lum)


    def i_oft( self, t ): return np.argmin( abs(self.t - t) )

    # Time at which LC hits given mag
    def t_ofL( self, des_L, terr=1, premax=False ):
        """
        INPUTS
        des_L   : (float) desired luminosity
        terr    : (float) error on time
        premax  : (bool)  find time pre-max light or post-max light
        """
        i_min = 0             if premax else self.i_peak()
        i_max = self.i_peak() if premax else None

        mean_step = np.mean( np.diff(self.t) )
        # If not dense enough times, interpolate
        if mean_step > terr:
            N_interp = int( np.ceil(mean_step/terr) * self.size() )
            new_t = np.arange( self.t[0], self.t[-1], N_interp, endpoint=True )
            use_lc = self.interp(new_t)[i_min:i_max]
        else:
            use_lc = self[i_min:i_max]

        if use_lc.size()==0: return -1, -1

        i_des = np.argmin( abs(use_lc.lum - des_L) )
        return use_lc.t[ i_des ], use_lc.lum[ i_des ], i_des+i_min


    def L_oft( self, des_t, terr=1 ):
        """
        INPUTS
        des_t   : (float) desired time
        terr    : (float) error on time
        """
        # If not dense enough times, interpolate
        mean_step = np.mean( np.diff(self.t) )
        if mean_step > terr:
            N_interp = int( np.ceil(mean_step/terr) * self.size() )
            new_t = np.linspace( self.t[0], self.t[-1], N_interp, endpoint=True )
            use_lc = self.interp(new_t)
            i_des = np.argmin( abs(use_lc.t - des_t) ) 
            return use_lc.lum[ i_des ], use_lc.t[ i_des ]
        else:
            i_des = np.argmin( abs( self.t - des_t ) )
            return self.lum[ i_des ], self.t[ i_des ], i_des


    def slope( self, log=False ):
        dLdt = np.empty( len(self) )

        def first_deriv(L, dt):
            """
            Calculates a second-order accurate dL/dt,
            assuming evenly-spaced time sampling
            """
            dLdt = np.empty( len(L) )
            divfactor = 0.5/dt

            dLdt[1:-1] =  (L[2:] - L[:-2])*divfactor
            dLdt[0]    = -(L[ 0] - 4*L[ 1] + L[ 2])*divfactor
            dLdt[-1]   = -(L[-1] - 4*L[-2] + L[-3])*divfactor

            return dLdt

        if not log:
            dt = np.mean( np.diff(self.t) )
            return first_deriv(self.lum, dt)
        else:
            logt = np.linspace(np.log10(self.t[0]), np.log10(self.t[-1]), len(self))
            loglc = self.interp( 10**logt, interp_log=True, left=None, right=None)

            dlogt = np.mean( np.diff(logt) )
            logL = np.log10(loglc.lum)

            return first_deriv(logL, dlogt)


#
# Optically thin shells
#
class Shell(object):
    def __init__(self, f_R=0, rho=0, mass=0, R_in=0, R_out=0, t_imp=0, t_cross=0, do_fill=True):
        """
        Make sure t_imp and t_cross are in seconds
        """
        self._have_props = {}
        # Set everything according to user inputs
        if f_R:
            self.f_R = f_R
            self._have_props['f'] = 1
        else:
            self._have_props['f'] = 0
        if rho:
            self.rho = rho
            self._have_props['rho'] = 1            
        else:
            self._have_props['rho'] = 0
        if mass:
            self.mass = mass
            self._have_props['M'] = 1            
        else:
            self._have_props['M'] = 0
        if R_in:
            self.R_in = R_in
            self._have_props['Ri'] = 1
        else:
            self._have_props['Ri'] = 0
        if R_out:
            self.R_out = R_out
            self._have_props['Ro'] = 1
        else:
            self._have_props['Ro'] = 0
        if t_imp:
            self.t_imp = t_imp
            self._have_props['ti'] = 1
        else:
            self._have_props['ti'] = 0
        if t_cross:
            self.t_x = t_cross
            self._have_props['tx'] = 1
        else:
            self._have_props['tx'] = 0

        # We now have all the inputs - try to fill in missing information
        if do_fill:
            if not R_in:
                self._calc_Rin() # has the most ways to be calculated
            if not f_R:
                self._calc_f_R()
            if not R_out:
                self._calc_Rout()
            if not t_imp:
                self._calc_timp()
            if not t_cross:
                self._calc_tx()
            if not rho:
                self._calc_rho()
            if not mass:
                self._calc_mass()

        # Check model hydrodynamic assumptions
        is_good = self._check_in_outer_ejecta()
        is_good = is_good & self._check_likely_adiabatic()
        self.valid = is_good
        

    def _apply_thickness_relation(self):
        """
        HNK16 Eqn 3
        """
        needed_count = self._have_props['f'] + self._have_props['Ri'] + self._have_props['Ro']
        if needed_count < 2: # not enough info to apply relation
            return -1
        elif needed_count == 3: # already have all the information
            return 1  
        # get missing information
        if self._have_props['f'] and self._have_props['Ri']:
            self.R_out = (1+self.f_R)*self.R_in
            self._have_props['Ro'] = 1
        elif self._have_props['f'] and self._have_props['Ro']:
            self.R_in = self.R_out/(1 + self.f_R)
            self._have_props['Ri'] = 1
        elif self._have_props['Ri'] and self._have_props['Ro']:
            self.f_R = self.R_out/self.R_in - 1
            self._have_props['f'] = 1
        else:
            return -1
            
    def _apply_shock_cross_relation(self):
        """
        HNK16 Eqn 7
        """
        needed_count = self._have_props['ti'] + self._have_props['tx'] + self._have_props['f']
        if needed_count < 2: # not enough info to apply relation
            return -1
        elif needed_count == 3: # already have all the information
            return 1  
        # get missing information
        if self._have_props['ti'] and self._have_props['f']:
            self.t_x = self.calc_t_cross()
            self._have_props['tx'] = 1                        
        elif self._have_props['ti'] and self._have_props['tx']:
            self.f_R = ((self.t_x/self.t_imp)/0.97744)**(1/1.28540) - 1
            self._have_props['f'] = 1            
        else:
            self.t_imp = self.t_x/self.calc_x_cross()
            self._have_props['ti'] = 1
        return 0

            
    def _apply_contact_relation(self):
        """
        HNK16 Eqns 4&5
        """
        needed_count = self._have_props['ti'] + self._have_props['rho'] + self._have_props['Ri']
        if needed_count < 2: # not enough info to apply relation
            return -1
        elif needed_count == 3: # already have all the information
            return 1  
        # get missing information        
        if self._have_props['ti'] and self._have_props['rho']:
            self.R_in = ckt.calc_R_c(self.t_imp, 0, 10, self.rho, M_ej=1.38*C.M_SUN)
            self._have_props['Ri'] = 1
        elif self._have_props['Ri']:
            R_norm = 5.850e14
            rho_norm = 1e-18
            t_norm = 8.64e4
            if self._have_props['rho']:
                self.t_imp = t_norm * (self.R_in/R_norm * (self.rho/rho_norm)**0.1)**(10/7)
                self._have_props['ti'] = 1
            else:
                self.rho = rho_norm * (self.R_in/R_norm * (self.t_imp/t_norm)**(-0.7))**(-10)
                self._have_props['rho'] = 1
        return 0

    
    def _apply_mass_relation(self):
        """
        Mass of a constant-density shell
        """
        needed_count = self._have_props['M'] + self._have_props['rho']
        needed_count+= self._have_props['Ri'] + self._have_props['Ro']
        if needed_count < 3: # not enough info to apply relation
            return -1
        elif needed_count == 4: # already have all the information
            return 1  
        # get missing information
        VOLFAC = 4*C.PI/3
        if self._have_props['rho'] and self._have_props['Ri'] and self._have_props['Ro']:
            self.mass = VOLFAC * self.rho * (self.R_out**3 - self.R_in**3)
            self._have_props['M'] = 1
        elif self._have_props['M'] and self._have_props['Ri'] and self._have_props['Ro']:
            self.rho = self.mass/(VOLFAC*(self.R_out**3 - self.R_in**3))
            self._have_props['rho'] = 1            
        elif self._have_props['M'] and self._have_props['rho'] and self._have_props['Ri']:
            self.R_out = (self.mass/(VOLFAC*self.rho) + self.R_in**3)**(1/3)
            self._have_props['Ro'] = 1
            
        else:
            self.R_in = (self.R_out**3 - self.mass/(VOLFAC*self.rho))**(1/3)
            self._have_props['Ri'] = 1
        return 0

            
    def _calc_f_R(self):
        res1 = self._apply_thickness_relation()
        if res1 == -1:
            res2 = self._apply_shock_cross_relation()
            if res2 == -1:
                print('Not enough information to determine f_R')
            return res2
        else:
            return res1

    def _calc_rho(self):
        res1 = self._apply_mass_relation()
        if res1==-1:
            res2 = self._apply_contact_relation()
            if res2 == -1:
                print('Not enough information to determine CSM density')
            return res2
        else:
            return res1

    def _calc_Rin(self):
        res1 = self._apply_thickness_relation()
        if res1==-1:
            res2 = self._apply_contact_relation()
            if res2==-1:
                res3 = self._apply_mass_relation()
                if res3==-1:
                    print('Not enough information to determine R_in')
                return res3
            else:
                return res2
        else:
            return res1
        
    def _calc_Rout(self):
        res1 = self._apply_thickness_relation()
        if res1==-1:
            res2 = self._apply_mass_relation()
            if res2==-1:
                print('Not enough information to determine R_out')
            else:
                return res2
        else:
            return res1

    def _calc_timp(self):
        res1 = self._apply_contact_relation()
        if res1==-1:
            res2 = self._apply_shock_cross_relation()
            if res2==-1:
                print('Not enough information to determine impact time')
            else:
                return res2
        else:
            return res1
        
    def _calc_tx(self):
        res1 = self._apply_shock_cross_relation()
        if res1==-1:
            print('Not enough information to determine shock crossing time')
        else:
            return res1
        
    def _calc_mass(self):
        res1 = self._apply_mass_relation()
        if res1==-1:
            print('Not enough information to determine CSM mass)')
        else:
            return res1

    #
    # Applicability checks
    #
    def _check_in_outer_ejecta(self):
        r_t = ckt.calc_r_t(self.t_imp, 10, delta=1, M_ej=1.38*C.M_SUN)
        if r_t > self.R_in:
            return False # initial point of contact is in inner ejecta (bad)
        else:
            return True # initial point of contact is in outer ejecta (good)

    def _check_likely_adiabatic(self):
        return (self.rho < 1e-14)

        
        
    # Time for forward shock to cross CSM outer edge
    def calc_x_cross(self):
        """
        HNK16 Eqn 7
        """
        return  0.97744*(1 + self.f_R)**1.28540 if self._have_props['f'] else -1
    def calc_t_cross(self):
        x = self.calc_x_cross()
        return self.t_imp*x if (x!=-1 and self._have_props['ti']) else -1

    #
    # Optically thin light-curve properties
    #
    def estimate_Lp(self, nu_GHz, eps_B=0.1, f_NT=1.):
        """
        HNK16 Eqn 11
        """
        if not self.valid: return 0
        
        const_term =  3.2e28 * f_NT**-1 * (eps_B/0.1)
        nu_term = nu_GHz**-1
        rho_term = (self.rho/1e-18)**(8./7)
        R_term = (self.R_in/1e16)**(3./7)
        f_term = ( 1 - (1+self.f_R)**-1.28 )

        return 3.2e28 * const_term*nu_term*rho_term*R_term*f_term
    
    # rise
    def rise_to_peak(self, x, L_p ):
        """
        HNK16 Eqns 9 & 10
        """
        if not self.valid: return 0
        
        # All x = t/t_imp assumed to be pre-peak
        F_R = self.f_R + 1
        if self.f_R!=1: 
            L_inf = L_p/(1 - F_R**-1.28) # 1.705*L_p2
        else:
            L_inf = 1.705*L_p

            L = L_inf * ( 1 - 0.985/x )

        return L

    # Time to reach characteristic points on the decline
    # HNK16 Eqn 12 and Table 1
    # time to wane to 10^-1/4 peak
    def x_dec_pt0(self):
        return  1.01376*(1 + self.f_R)**1.39019
    # time to wane to 10^-1 peak
    def x_dec_pt1(self):
        return  1.05677*(1 + self.f_R)**1.51952
    # time to wane to 10^-2 peak
    def x_dec_pt2(self):
        return  1.13408*(1 + self.f_R)**1.62007 
    # time to wane to 10^-3 peak
    def x_dec_pt3(self):
        return  1.26357*(1 + self.f_R)**1.69524

    def make_thin_LC(self, L_p, at_times=[], ntimes=10, include_rev=False ):
        # This function does nothing if the shell is not valid
        # within the model physical assumptions
        if not self.valid: return ModLC([],[])
        
        x_p = self.calc_x_cross() # normalized time of peak

        # fall
        if not include_rev:
            falling_x = [self.x_dec_pt0(), self.x_dec_pt1(), self.x_dec_pt2(), self.x_dec_pt3()]
            falling_lum = L_p * np.array([10**-0.25, 1e-1, 1e-2, 1e-3])            
        else:
            falling_x = [self.x_dec_pt0(), self.x_dec_pt3()]
            L_qt = L_p * 10**-0.25
            falling_lum = np.array([L_qt, L_qt*(falling_x[1]/falling_x[0])**-11.5])
        falling_x = np.array(falling_x) 


        if len(at_times)>0:
            x_targ = at_times/self.t_imp
            x_targ.sort()

            # calculate rise at target times directly from function
            rise_x_targ = x_targ[ x_targ <= x_p ]
            rise_lum = self.rise_to_peak(rise_x_targ, L_p)

            # interpolate fall
            # for decline interpolation, need to fill in fall after peak
            falling_x = np.concatenate(([x_p], falling_x))
            falling_lum = np.concatenate(([L_p], falling_lum))

            fall_x_targ = x_targ[ x_targ > x_p ]
            fall_lum = np.interp( np.log10(fall_x_targ), np.log10(falling_x), np.log10(falling_lum),
                                  left=np.nan, right=np.nan )
            fall_lum = 10**fall_lum
            # fill in the latest time stuff with adiabatic losses L ~ t^-9
            super_late = fall_x_targ > self.x_dec_pt3()
            fall_lum[super_late] = falling_lum[-1] * (fall_x_targ[super_late]/falling_x[-1])**-9
            
            all_x = np.concatenate((rise_x_targ, fall_x_targ))
            lum = np.concatenate((rise_lum, fall_lum))
        else:
            n_rise = ntimes - falling_x.size
            x_min = 1+1e-3
            rise_x = np.logspace(np.log10(x_min), np.log10(x_p), n_rise)
            rise_x = np.sort(np.concatenate( (rise_x, np.array([1.09])) ))

            rise_lum = self.rise_to_peak(rise_x, L_p)

            all_x = np.concatenate((rise_x, falling_x))
            lum = np.concatenate((rise_lum, falling_lum))

        t = self.t_imp*all_x
            
        return ModLC(t, lum)


    def get_tau_norm(self):
        return 3.4 * (self.rho/1e-18)**1.5 *(self.t_imp/(100*C.DAY2SEC))**-1.25

    def calc_tau_inshell(self, times):
        x = times/self.t_imp
        return self.get_tau_norm() * x**-1.34 * (1-x**-1.66)
        
    def tau_at_tx(self):
        return self.calc_tau_inshell(self.t_x)        
    
    def calc_tau_SSA(self, times):
        if not self.valid: return np.array([])

        x = times/self.t_imp

        # time shock crosses CSM edge
        x_p = self.calc_x_cross()
        # characteristic times along decline 
        x0 = 1.015*(self.f_R + 1)**1.38
        x1 = 1.046*(self.f_R + 1)**1.49
        x2 = 1.118*(self.f_R + 1)**1.54
        x3 = 1.206*(self.f_R + 1)**1.60

        pre_cross = x <= x_p
        near_cross = (x > x_p) & (x <= x3)
        super_late = x > x3

        # this is the function to describe tau evolution while
        # the shock is in the shell
        tau = self.calc_tau_inshell(times)

        # early part of the decline - interpolate between characteristic points as power-law
        tau_cross = self.tau_at_tx() # not guaranteed that times containts x_p
        dec_tau  = tau_cross*np.array([  1,0.5,1e-1,1e-2,1e-3]) # tau along initial decline
        dec_x = np.array([x_p, x0,  x1,  x2,  x3]) # characteristic times
        tau_interp = np.interp(np.log10(x[near_cross]), np.log10(dec_x), np.log10(dec_tau))
        tau_interp = 10**tau_interp
        tau[near_cross] = tau_interp

        # late part of the decline - adiabatic approximation
        # from alpha ~ rho * u^13/4
        #   rho ~ t^-3
        #   u ~ t^-5
        # vol/area ~ R ~ t
        # tau ~ alpha * vol/area 
        tau0 = dec_tau[-1]
        x0 = dec_x[-1]
        tau[super_late] = tau0 * (x[super_late]/x0)**-18.25

        return tau


    def make_LC(self, times, L_p, include_rev=False, include_SSA=True):
        if include_SSA & include_rev:
            print('Treatment of SSA does not include reverse shock absorption.')

        # Optically thin light-curve
        lc = self.make_thin_LC(L_p, times)

        # Extinction
        tau = self.calc_tau_SSA(lc.t)
        abs_fac = (1-np.exp(-4*tau))/(4*tau)

        lc.lum *= abs_fac

        return lc
        


# Random helpful function
def estimate_rho(L_p, R, f_R, nu_GHz, eps_B=0.1, f_NT=1.):
    """
    Lets you estimate the density from an observation
    """
    norm_rho = 1e-18
    pwr_in_L = 8./7

    F_R = f_R + 1
    rho = norm_rho * (L_p/3.2e28 * nu_GHz * f_NT * (R/1e16)**(-3./7) * (eps_B/0.1)**-1 / (1 - F_R**-1.28))**(1/pwr_in_L)
    return rho


