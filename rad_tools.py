# A set of functions useful for simple radiation calculations

import numpy             as np
import Const             as C
#import scipy.optimize    as optim

def get_ioniz_en(A, i):
# Get the ionization energy (in eV) for ion
# from http://dept.astro.lsa.umich.edu/~cowley/ionen.htm
#
# INPUTS
# A   : int   atomic number
# i   : int   ionization state
    assert type(A)==int;
    assert type(i)==int;

    # binding energies in eV
    chis = { 1: {0:13.5984} ,                # H
             2: {0: 24.5874, 1: 54.417760}, # He
             6: {0: 11.2603, 1: 24.3833, 2: 47.8878, 3: 64.4939, 4: 392.087}, # C
             8: {0: 13.6181, 1: 35.1211, 2: 54.9355, 3: 77.41353, 4: 113.8990} # O
           };

    assert A in chis.keys();
    assert i in chis[A].keys();

    return chis[A][i];


def calc_ioniz_T(A, i, n_e):
# Get the temperature at which ion loses e-
#
# INPUTS
# A   : int    atomic number
# i   : int    (current) ionization state
# n_e : float  electron density

    chi = get_ioniz_en(A,i); # binding energy in eV

    k = C.K_B ;

    h_eVs = C.H * C.ERG2EV ;
    mecsq_eV = C.M_E_ERG * C.ERG2EV ;
    csq_cgs = C.C_LIGHT ** 2;

    C_lm_therm = h_eVs/np.sqrt( 2*np.pi*mecsq_eV*k/csq_cgs ); # units of cm*sqrt(K)

    def closeness_to_ionized(T):
        lm_therm = C_lm_therm * T**-0.5; 
        return chi/(k*T) + np.log( 0.5 * n_e * lm_therm**3 ) ; # there should be a U^0/U^+ in the log as well

    T_0 = 0.03*chi/k;
    print(T_0, closeness_to_ionized(T_0));
    print('*** ABORTING: calc_ioniz_T() relied on scipy.optimize and I had issues importing that before so I commented out that part')
    return T_0;
    #T  = optim.newton(closeness_to_ionized, T_0);
    #
    #return T;


class SynchrotronCalculator(object):
# A tool for calculating synchrotron signal

    def __init__( self, a_p = 3, a_eps_B = 0.1, a_eps_e = 0.1 ):
        """ 
        INPUTS 
        a_p      : (int, float) dn_e ~ gamma^-p dgamma
        a_eps_B  : (float, float[]) u_B = eps_B * u_gas
        a_eps_e  : (float, float[]) u_e = eps_e * u_gas
        """    
        self.eps_B = a_eps_B
        self.eps_e = a_eps_e
        self.p     = a_p


    def calc_gamma_min( self, a_ray, a_fNT=1. ):
        """
        Calculate the minimum electron gamma given that
          f_NT*n_e = integral_gammin^infinity ( C*gam^-p dgam)
        and
          eps_e * u_gas = integral_gammin^infinity ( gam*m_e*c^2  * C*gam^-p dgam

        you get
          gammin = (p-2)/(p-1) * eps_e * u_gas / (f_NT * n_e * m_e * c^2 )
        for each cell in a_ray
        """
       
        return float(self.p-2)/(self.p-1) * (self.eps_e * a_ray.u_gas) / (a_fNT * a_ray.n_e * C.M_E_ERG )


    def calc_C( self, a_ray, a_fNT=1. ):
        """
        Calculates C as in 
          dn_e = C * gam^-p * dgam
        for each cell in a_ray
        """
        gammin = self.calc_gamma_min( a_ray, a_fNT )
        return a_fNT * a_ray.n_e * (self.p - 1) * gammin**(self.p-1)


    def calc_nu_syn( self, a_ray ):
        """
        Calculate synchrotron frequency
          nu_syn = eB/(2*pi*m_e*c) 
        since B = sqrt( 8*pi*u_B ),
          nu_syn = sqrt(2/pi) * e/(m_e*c) * sqrt(eps_B*u_gas)
        for each cell in a_ray
        """
        return  np.sqrt( 2 * self.eps_B * a_ray.u_gas / np.pi ) * (C.E_ESU / (C.M_E * C.C_LIGHT))


    def calc_nu_crit( self, a_ray, a_fNT=1. ):
        """
        Calculate the critical frequency 
          nu_crit = gamma_min**2 * nu_syn
        for each cell in a_ray
        """
        numer_fact = np.sqrt(2/np.pi) * C.E_ESU * C.C_LIGHT * C.M_E_ERG**-3 

        return numer_fact * self.eps_e**2 * np.sqrt(self.eps_B) * (float(self.p-2)/(self.p-1))**2 * (a_fNT * a_ray.n_e)**-2 * a_ray.u_gas**2.5


    def calc_j_nu( self, a_nu_Hz, a_ray, a_fNT=1. ):
        """
        Calculates synchrotron j_nu produced in a_ray  at each nu in a_nu_Hz, 
        assuming a power-law electron fraction a_fNT such that 
        n_e,NT = a_fNT * n_e
    
        assumes optically thin, and that only power-law electrons contribute
    
        for frequencies higher than nu_syn,
    
            j_nu = (sigam_Th * c / (6*pi)) * C * u_B * gamma**(1-p) / nu_syn
    
            if p = 3,
    
            j_nu = (sigam_Th * c / (6*pi)) * C * u_B / nu
    
        lower than nu_crit is just the electrons moving at gamma_min:
            j_nu = ?
    
    
        INPUTS
        a_nu_Hz   : (float[])      frequencies at which to calculate j_nu, in Hz
        a_ray     : (RayClass.Ray) cells to calculate emissivity in
        a_fNT     : (float)        nonthermal electron fraction

        OUTPUT
        j_nu      : (float[])  rows are frequency, columns are cells
        """
        assert a_fNT <= 1

        if self.p!=3:
            print('Sorry, I\'m not set up for p=/=3 yet!')
            return None

        numer_const = C.SIGMA_TH * C.C_LIGHT / (6 * np.pi)
     
        csm_vect = self.calc_C(a_ray, a_fNT) * self.eps_B * a_ray.u_gas

        # only valid for p=3:
        j_nu =  numer_const *  np.outer( 1/a_nu_Hz, csm_vect ) 

        nu_syn = self.calc_nu_syn( a_ray )
        
        below_crit = np.outer(a_nu_Hz, 1/nu_syn) < 1
        if np.any(below_crit):
            # p=3 only:
            jnu_if_below = np.outer( a_nu_Hz**(1./3) , numer_const*csm_vect*nu_syn**(-4./3) )
            j_nu[below_crit] = jnu_if_below[below_crit]

        return j_nu



    def calc_L_nu( self, a_nu_Hz, a_ray, a_fNT=1. ):
        j_nu = self.calc_j_nu( a_nu_Hz, a_ray, a_fNT )
        vols = a_ray.cell_volume()

        return np.sum( j_nu * vols, axis=1 )  # sum up contribution from each cell



def solve_ion_Saha():
    print("this function is not written yet.");
    return 0;
    # sum_Z ( sum_i ( n[Z][i] ) ) = n_e        neutrality
    # n[Z] = sum_i ( n[Z][i] )
    # n[Z] = frac[Z]*n_tot                     definition
    #
    # give: atom_nums[]     array of atomic numbers
    #       fracs[]         array of species mass fraction
    #       rho             the net density
    #       T               temperature
    #
    # kT = (C.k_B.to(units_of_chi)).value*T
    # 
    # M_E = C
    # two_over_lm_T_cubed = 2. * (2*np.pi*M_E*kT / H**2)**1.5
    #
    # mu = ( sum(fracs/atom_nums) )**-1
    # n_tot = rho / (mu*M_P)
    #
    # num_spec = len(atom_nums)   # number of species present
    #
    # 
    # def init_ion_fractions(atoms)
    #     X_neutral = np.ones(num_spec);
    #     X_i       = [];
    #     for A in atom_nums:
    #        its_start = np.zeros(A); # starting guess: all in neutral
    #        X_i.append( its_start );
    #     return X_neutral, np.array(X_i);
    #
    # X_neutral, X_i = init_ion_fractions(atoms)  # number densities of ions
    # Where the total number density of a particular species
    # ionization state is n_i = n_tot * fracs[Z] * X_neutral[Z] * X_i[Z][i]
    # and the total number density of neutral is n_0 = n_tot * fracs[Z] * X_neutral[Z]
    #
    #  
    # def get_partition_function(A, i)
    # def get_partition_functions(A, as_ratio=False)
    #     calls get_partition_function
    #
    # U_neutral, U_ratios = get_partition_functions(atom_nums, as_ratio=True)
    #
    #
    # def get_binding_energies(atoms)
    #
    # chi = get_binding_energies(atom_nums)  # energy *to get to* i
    #
    # 
    # def Saha_equations(X_i, X_neutral, U_ratios, chi)
    #     n_e = n * sum_A( fracs[A]*X_neutral[A]*(1+sum(X_i[A])) ) # charge conservation
    #
    #     equations = [ ];
    # 
    #     for A in atom_nums: # each species
    #         equations.append( X_neutral[A] * (1+sum(X_i[A])) - 1 ); # number conservation for this species
    #         for i in range(A): # each ion
    #             equations.append(  (X_i[Z][i] * n_e) - (U_ratios[Z][i] * two_over_lm_T_cubed * np.exp(-chi[Z][i]/kT))  ); # Saha for this ionization state
    #
    #     return tuple(equations)



    


