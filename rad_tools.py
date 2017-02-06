# A set of functions useful for simple radiation calculations
import numpy             as np
import Const             as C
#import scipy.optimize    as optim
from scipy.special import gamma as gamfunc

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


class RelE(object):
# A class for managing relativistic electron properties
    def __init__(self, a_p = 3, a_eps_e = 0.1 ):
        """ 
        INPUTS 
        a_p      : (int, float) dn_e ~ gamma^-p dgamma
        a_eps_B  : (float, float[]) u_B = eps_B * u_gas
        a_eps_e  : (float, float[]) u_e = eps_e * u_gas
        """    
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


    def sample_gamma(self, a_ray, a_N_p=1e6, a_gam_max=1e2, a_fNT=1.):
        # Calculate how many photons scatter in each cell
        cell_tau = a_ray.n_e * C.SIGMA_TH * (a_ray.r2 - a_ray.r1)
        frac_tau = cell_tau/sum(cell_tau)
        N_per = (frac_tau * a_N_p).astype(int)
        # check that we have the right number of total packets
        N_per[-1] = a_N_p - sum(N_per[:-1])

        gam_samp = np.zeros(a_N_p)

        # Sample gamma in each cell to populate whole gamma array
        x_mins = self.calc_gamma_min(a_ray, a_fNT)
        x_maxs = a_gam_max*np.ones(len(a_ray))

        # (check for non-relativistic values)
        nonrel = x_mins < 1
        x_mins[nonrel] = 1.0
        nonrel = x_maxs < 1
        if np.any(nonrel): 
            print('** Warning; {}% of cells are non-relativistic **'.format((100.0*sum(nonrel))/len(nonrel)))
        x_maxs[nonrel] = 1.0

        maxterm = x_maxs**(1-self.p)
        minterms = x_mins**(1-self.p)
        n = 0
        for i in range(len(a_ray)):
            term1 = (maxterm[i] - minterms[i]) * np.random.rand(N_per[i])
            term2 = minterms[i]
            gam_samp[n : n+N_per[i]] = (term1 + term2)**(1./(1-self.p))
            n += N_per[i]
        #term1 = ( x_max**(1-self.p) - x_min**(1-self.p) ) * np.random.rand(a_N_p) ;
        #term2 = x_min**(1-self.p) ;
        #return (term1 + term2)**(1./(1-self.p)) ;

        return gam_samp


    def calc_tau_es(self, a_ray):
        return sum( a_ray.n_e * C.SIGMA_TH * (a_ray.r2 - a_ray.r1))



class SynchrotronCalculator(object):
# A tool for calculating synchrotron signal
    def __init__( self, a_p = 3, a_eps_B = 0.1, a_eps_e = 0.1 ):
        """ 
        INPUTS 
        a_p      : (int, float) dn_e ~ gamma^-p dgamma
        a_eps_B  : (float, float[]) u_B = eps_B * u_gas
        a_eps_e  : (float, float[]) u_e = eps_e * u_gas
        """    
        self.elec  = RelE(a_p, a_eps_e)
        self.eps_B = a_eps_B

    def calc_gamma_min( self, a_ray, a_fNT=1. ):
        return self.elec.calc_gamma_min(a_ray, a_fNT)

    def calc_C( self, a_ray, a_fNT=1. ):
        return self.elec.calc_C(a_ray, a_fNT )


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


    def calc_alpha( self, a_nu_Hz, a_ray, a_fNT=1. ):
        """
        Calculate the extinction coefficient for synchrotron self-absorption
        """
        SINA = 2.0/C.PI
        GG = gamfunc((3*self.p + 2)/12.) * gamfunc((3*self.p + 22)/12.)

        C_E = C.M_E_ERG**(self.p-1) *self.calc_C(a_ray, a_fNT)

        Bmag = np.sqrt(8*C.PI*self.eps_e*a_ray.u_gas)

        # R&L Eqn 6.53
        al = np.sqrt(3)*C.E_ESU**3 / (8*C.PI*C.M_E)
        al*= (3*C.E_ESU/(2*C.PI *C.M_E**3 *C.C_LIGHT**5))**(0.5*self.p)
        al*= C_E *(Bmag*SINA)**(0.5*(self.p + 2))
        al*= GG

        fin_al = np.outer( a_nu_Hz**(-0.5*(self.p + 4)), al )

        return fin_al

    
    def calc_tau( self, a_nu_Hz, a_ray, a_fNT=1. ):
        al = self.calc_alpha( a_nu_Hz, a_ray, a_fNT )
        dr = a_ray.r2 - a_ray.r1

        tau = al.dot(dr)

        return tau


    def calc_L_nu( self, a_nu_Hz, a_ray, a_fNT=1. ):
        j_nu = self.calc_j_nu( a_nu_Hz, a_ray, a_fNT )
        vols = a_ray.cell_volume()

        return np.sum( j_nu * vols, axis=1 )  # sum up contribution from each cell


class ComptonCalculator(object):
# A tool for calculating compton scattering signal
    def __init__(self, a_p=3, a_eps_e=0.1, a_Npts=1e5):
        # a_Npts : number of photon packets to propagate
        self.elecs = RelE(a_p, a_eps_e)
        self.Npts = a_Npts

    def N_scat(self, a_ray):
        tau = self.elecs.calc_tau_es(a_ray)
        return max( tau, tau**2 )

    def scatter(self, gam, beta, hnu_in, mu_in, mu_out):
    # Calculate e_out from a single scatter 
    # INPUTS
    #   gam    : Lorentz factor of electron
    #   beta   : velocity (c=1 units) of electron
    #   hnu_in : incoming photon energy
    #   mu_in  : incoming photon angle relative to electron velocity
    #   mu_out : outgoing """
    # 
        return hnu_in * gam**2 * (1+beta*mu_in) * (1-beta*mu_out);


    def sample_spectrum(self, a_spec):
        summed = np.cumsum( a_spec[:,1] )
        summed /= summed[-1]

        rand = np.random.rand(self.Npts)
        i_closest = map( lambda r: np.argmin(abs(summed-r)), rand )

        hnu = a_spec[:,0]
        return hnu[i_closest]


    def calc_IC_spec(self, a_ray, a_L_back, a_spec):
        # a_L_back  : background luminosity
        # a_spec    : background spectrum, arbitrary normalization, as array
        #             col 0: energy (h*nu, in erg), col 1: flux (per erg)

        # Electron/scatterer properties
        tau = self.elecs.calc_tau_es(a_ray)
        N_scat = max(tau, tau**2)

        if N_scat > 3: 
            print('Not single scatter!')
            return (0,0)

        gam = self.elecs.sample_gamma(a_ray,self.Npts)
        beta = np.sqrt(1 - 1/gam**2)
        
        # Photon properties
        L_p       = a_L_back/self.Npts
        scat_frac = 1 - np.exp(-tau)
        mu_in     = 1 - 2*np.random.rand(self.Npts)
        mu_out    = 1 - 2*np.random.rand(self.Npts)

        hnu_in = self.sample_spectrum(a_spec) # erg
        hnu_out = self.scatter( gam, beta, hnu_in, mu_in, mu_out ) # erg

        L_out = L_p * (hnu_out/hnu_in) * scat_frac # erg/s per packet
        

        # Produce spectrum
        # bin photons by energy (demand 5000 packets per bin)
        hnu_bins = np.logspace( np.log10(min(hnu_out)), np.log10(max(hnu_out)), self.Npts/1000); 
        binsize = np.diff(hnu_bins);
        # figure out which bin each out-packet goes in:
        i_assign = np.digitize( hnu_out, hnu_bins ); 

        L_nu = np.zeros( len(binsize) ); # erg s^-1 Hz^-1
        for i in range( 1, len(hnu_bins) ):
            L_bin = sum( L_out[i_assign==i] );   
            L_nu[i-1] = L_bin;
        L_nu /= binsize ;


        return L_nu, (hnu_bins[:-1] + hnu_bins[1:])*0.5, sum(L_out) + a_L_back*np.exp(-tau)


    def calc_tot_spec(self, a_ray, a_L_back, a_spec):
        # a_L_back  : background luminosity
        # a_spec    : background spectrum, arbitrary normalization, as array
        #             col 0: energy (h*nu, in erg), col 1: flux (per erg)

        # Electron/scatterer properties
        tau = self.elecs.calc_tau_es(a_ray)
        N_scat = max(tau, tau**2)

        if N_scat > 3: 
            print('Not single scatter!')
            return (0,0)

        gam = self.elecs.sample_gamma(a_ray,self.Npts)
        beta = np.sqrt(1 - 1/gam**2)
        
        # Photon properties
        L_p       = a_L_back/self.Npts
        scat_frac = 1 - np.exp(-tau)
        mu_in     = 1 - 2*np.random.rand(self.Npts)
        mu_out    = 1 - 2*np.random.rand(self.Npts)

        hnu_in = self.sample_spectrum(a_spec) # erg
        hnu_out = self.scatter( gam, beta, hnu_in, mu_in, mu_out ) # erg

        L_out = L_p * (hnu_out/hnu_in) * scat_frac # erg/s per packet scattered 
        L_orig = L_p * np.exp(-tau) * np.ones(self.Npts)  # erg/s per packet unscattered 

        all_hnu = np.concatenate((hnu_in, hnu_out))
        all_L   = np.concatenate((L_orig, L_out))
        N_tot = len(all_hnu)

        # Produce spectrum
        # bin photons by energy (demand 500 packets per bin)
        hnu_bins = np.logspace( np.log10(min(all_hnu)), np.log10(max(all_hnu)), N_tot/200); 
        binsize = np.diff(hnu_bins);
        # figure out which bin each out-packet goes in:
        i_assign = np.digitize( all_hnu, hnu_bins ); 

        L_nu = np.zeros( len(binsize) ); # erg s^-1 Hz^-1
        for i in range( 1, len(hnu_bins) ):
            L_bin = sum( all_L[i_assign==i] );   
            L_nu[i-1] = L_bin;
        L_nu /= binsize ;

        return L_nu, (hnu_bins[:-1] + hnu_bins[1:])*0.5, sum(all_L)


class BremCalculator(object):
    def __init__(self):
        self.a_var = 0


    def calc_j_nu_therm(self, a_ray, a_Z, a_nu, a_gaunt=1.2):
        """
        Calculates the specific free-free emissivity from a Maxwellian electron distribution
        Units of returned value: erg s^-1 cm^-3 Hz^-1 str^-1
        """
        # R&L Eqn 5.14b
        # assuming n_e = n_I
        csm_vect = 6.8e-38/(4*C.PI) * a_gaunt * a_Z**2 * (a_ray.n_e)**2 * (a_ray.T_gas)**-0.5 

        j_nu = np.empty((len(a_nu), len(a_ray)))
        for i, this_nu in enumerate(a_nu):
            j_nu[i] = csm_vect * np.exp(-C.H*this_nu/(C.K_B*a_ray.T_gas))
            
        return j_nu


    def calc_j_tot(self, a_ray, a_Z, a_gaunt=1.2):
        """
        Calculates the total free-free emissivity from a Maxwellian electron distribution
        Units of returned value: erg s^-1 cm^-3 str^-1
        """
        # assuming n_e = n_I
        # and allowing for relativistic correction
        # assuming Maxwellian distribution for electrons
        # Rybicki & Lightman Eqn 5.25
        return a_gaunt*1.4e-27/(4*C.PI) * a_Z**2 * (a_ray.n_e)**2 * (a_ray.T_gas)**0.5 * (1 + 4.4e-10*a_ray.T_gas)


    def calc_al_BB(self, a_ray, a_Z, a_nu, a_gaunt=1.2):
        """
        Calculates the free-free absorption at all nu, assuming blackbody source function
        Units of returned value : cm^-1
        """
        # R&L Eqn 5.19b
        # assuming n_e = n_I 
        const = 0.018 *a_Z**2 *a_gaunt
        csm_vect = a_ray.T_gas**-1.5 * a_ray.n_e**2
        al = const * np.outer(a_nu**-2, csm_vect)
        return al
        

    def calc_al_Ross(self, a_ray, a_Z, a_gaunt=1.2):
        """
        Calculates the Rosseland mean absorption to free-free
        Units of returned value : cm^-1
        """
        # R&L Eqn 5.20
        # assuming n_e = n_I
        return 1.7e-25 * (a_ray.T_gas)**-3.5 * a_Z**2 * (a_ray.n_e)**2 * a_gaunt


    def calc_al_from_S(self, a_ray, a_Z, a_nu, a_S_nu, a_gaunt=1.2):
        """
        Calculates absorption at all a_nu, assuming source function a_S_nu
        (corresponding to frequencies in a_nu) and thermal electrons.
        """
        # assuming electrons are thermal but source is not
        # Kirchov's law
        j_nu = self.calc_j_nu(a_ray, a_Z, a_nu, a_gaunt)
        return j_nu/a_S_nu
        


def approx_ion_radius(n_base, R_base, Qdot_optical, A_H=1.0, s=0):
    """
        n_base is n_e (in cm^-3) at the base of the unshocked CSM
        R_base is radius (in cm) of forward shock (base of unshocked CSM)
        Qdot_optical is the rate of production of optical photons (s^-1)
        A_H is the mass fraction of hydrogen
        s describes the power law of the CSM (assumed 0 or 2)
    """
    fact = 0.1 if s==2 else 0.09
    tau_IC = fact*C.SIGMA_TH*n_base*R_base
    Qdot_ion = Qdot_optical * (1 - np.exp(-tau_IC))

    n_7 = n_base/1e7
    R_15 = R_base/1e15
    pow = 3 - 2*s

    R_ion = R_base *( 1 + pow *(Qdot_ion/1.25e47)/(A_H *n_7**2 *R_15**3) )**(1./pow)
    return R_ion


def approx_ion_volume(n_base, R_base, R_out, Qdot_optical, A_H=1.0, s=0):
    """
        n_base : n_e (in cm^-3) at the base of the unshocked CSM
        R_base : radius (in cm) of forward shock (base of unshocked CSM)
        R_out  : outer CSM radius (in cm)
        Qdot_optical : rate of production of optical photons (s^-1)
        A_H    : mass fraction of hydrogen
        s      : describes the power law of the CSM (assumed 0 or 2)
    """
    R_strom = approx_ion_radius(n_base, R_base, Qdot_optical, A_H, s)

    if type(R_out) in [int, float]: 
        R_rec = min(R_strom, R_out)
    else:
        R_rec = R_out.copy()
        R_rec[ R_strom < R_out ] = R_strom

    V_rec = 4*C.PI/3 * (R_rec**3 - R_base**3)
    return V_rec


def calc_Halpha_emiss(n_e):
    return 6.6e-25 *n_e**2 # 4*pi*j_Halpha


def approx_Halpha_luminosity(n_base, R_base, R_out, Qdot_optical, A_H=1.0, s=0):
    eps_Ha = calc_Halpha_emiss(n_base)
    V_rec = approx_ion_volume(n_base, R_base, R_out, Qdot_optical, A_H, s)

    L_Ha = eps_Ha * V_rec
    return L_Ha


def approx_R_fwd(R_in, n, t):
# for constant-density CSM and a normal SN Ia (1.38 M_sun, 1e51 erg),
# approximate the evolution of the shock radius for t > t_imp with
# t in seconds
    R11 = R_in/1e11
    n7 = n/1e7
    return 1.85e11 *(R11 * n7)**-0.11 *t**-0.78


def calc_inverse_mu_e(spec_Z, spec_A, mass_fracs, ioniz_fracs):
    if type(spec_Z     ) != np.ndarray: spec_Z      = np.array(spec_Z     )
    if type(spec_A     ) != np.ndarray: spec_A      = np.array(spec_A     )
    if type(mass_fracs ) != np.ndarray: mass_fracs  = np.array(mass_fracs )
    if type(ioniz_fracs) != np.ndarray: ioniz_fracs = np.array(ioniz_fracs)

    assert sum(mass_fracs)==1.0

    return sum( spec_Z*mass_fracs*ioniz_fracs / spec_A )


def calc_inverse_mu_I(spec_A, mass_fracs):
    assert sum(mass_fracs)==1.0

    return sum( mass_fracs / spec_A )


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



    


