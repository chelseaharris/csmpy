# Calculate inverse Compton luminosity, assuming single scatter

import numpy                 as np
import scipy                 as spy
import matplotlib.pyplot     as plt
import Const                 as C


def pwrlaw_variate(num_pts, n, x_min, x_max):
# Form a power law (P(x) propto x^n) variate from a uniform distribution 
    term1 = ( x_max**(n+1) - x_min**(n+1) ) * np.random.rand(num_pts) ;
    term2 = x_min**(n+1) ;
    return (term1 + term2)**(1./(n+1)) ;


def planck( hnu, kT ):
# Calculate the value of the Planck function (normalized to 1) 
# INPUTS
#   hnu : (erg) array of energies to sample
#   kT  : (erg) temperature
#
    x0 = 2.821439 ;                   # blackbody peak x (i.e. in frequency)
    norm = (np.exp(x0) - 1)/x0**3  ;  # so that planck func will peak at 1

    x  = hnu / kT  ; 
    return norm * x**3 / ( np.exp(x)-1 ) ;


def thermal_variate(num_pts, hnu_min, hnu_max, kT):
# Randomly sample the Planck distribution
# INPUTS
#   num_pts : number of samples to take; good results for >=1.e6
#   hnu_min : (erg) lower bound of sample 
#   hnu_max : (erg) upper bound of sample 
#   kT      : (erg) temperature of distribution
#
    energies = np.zeros(num_pts);
    i = 0 ;
    while i<num_pts:
        hnu_check = (hnu_max-hnu_min) * np.random.rand() + hnu_min ; # uniformly sample hnu
        bb_val = planck( hnu_check, kT ) ;                           # check value: BB at hnu
        comp_val = np.random.rand() ;                                # comparison value: uniform dist
        if bb_val > comp_val:
            energies[i] = hnu_check ;
            i += 1 ; 
    return energies;


def arbitrary_variate(num_pts, spec_filename):
# Randomly sample a spectrum -- no smoothing of spectrum [yet]
# INPUTS
#   num_pts       : number of samples to take
#   spec_filename : location of spectrum from which to sample;
#                   expects two-column format, space separated
#                   col 1: wavelength (Ang); col 2: flux per unit wavelength

    # Load spectrum
    lm_Ang, f_lm = np.loadtxt(spec_filename, usecols=[0,1], unpack=True)

    # This is where we'd smooth if we wanted to
    
    # Convert to energy (and per energy) units
    lm = lm_Ang * 1e-8  # convert wavelength to cm
    hnu_spec = C.H * C.C_LIGHT / lm # convert wavelength to energy
    f_e = lm_Ang * f_lm / hnu_spec # convert flux to per energy

    # use the CDF to sample values
    hnu_spec, f_e = hnu_spec[::-1], f_e[::-1] # reverse to have mon inc
    summed = np.cumsum(f_e) # convert to normed cumsum
    summed /= summed[-1]
    #cdf = spy.interpolate.interp1d(hnu_spec, summed) # for higher accuracy
    
    # Convenience variables
    rand = np.random.rand(num_pts)
    i_closest = map( lambda r: np.argmin(abs(summed-r)), rand )
    hnu_samp = hnu_spec[i_closest]

    # Return sampled energies
    return hnu_samp
        


def scatter(gam, beta, hnu_in, mu_in, mu_out):
# Calculate e_out from a single scatter 
# INPUTS
#   gam    : Lorentz factor of electron
#   beta   : velocity (c=1 units) of electron
#   hnu_in : incoming photon energy
#   mu_in  : incoming photon angle relative to electron velocity
#   mu_out : outgoing """
# 
    return hnu_in * gam**2 * (1+beta*mu_in) * (1-beta*mu_out);


def compton(L_background, back_info, show=False):
# Calculate the optically thin inverse Compton spectrum from a thermal background
# of photons.
# INPUTS
#   L_background : background luminosity (erg/s)
#   back_info    : if doing blackbody, this is temperature in K;
#                  if doing spectrum, this is the file name
#   show         : whether or not to show the spectrum (bool)  
#
    assert type(back_info) in [float, str]

    N_p       = int(1e6);                          # number of photon packets

    ## ELECTRON (SCATTERER) PROPERTIES ##
    n         = -3;                                # e- gamma power law index
    gam       = pwrlaw_variate(N_p, n, 10., 100.); # e- Lorentz factor
    beta      = np.sqrt(1 - 1/gam**2);             # electron velocity (units of c)
    
    ## INCOMING PHOTON PROPERTIES ##
    L_SN      = L_background;                      # erg/s lum of (blackbody) photon source
    L_p       = L_SN/N_p;                          # erg/s lum of each photon packet
    tau       = 0.01;                              # optical depth; single-scattering approximation
    scat_frac = 1 - np.exp(-tau);                  # scattering fraction (for L_out)
    mu_in     = 1 - 2*np.random.rand(N_p);         # incoming photon directions
    mu_out    = 1 - 2*np.random.rand(N_p);         # outgoing photon directions

    if type(back_info)==float:
        hnu_min   = 0.1/C.ERG2EV;                      # lower energy bound for BB variate, in erg
        hnu_max   = 15 /C.ERG2EV;                      # upper energy bound for BB variate, in erg
        kT        = C.K_B * back_info;              # sauce temperature
        hnu_in    = thermal_variate( N_p, hnu_min, hnu_max, kT ); # BB dist, in erg
        hnu_in   *= C.ERG2EV;                          # convert from erg to eV
    elif type(back_info)==str:
        hnu_in = arbitrary_variate(N_p, back_info)
        hnu_in *= C.ERG2EV

    ### PRODUCE SPECTRUM ##

    # calculate outgoing photon energies for each packet:
    hnu_out = scatter( gam, beta, hnu_in, mu_in, mu_out ); # eV

    # convert to packet luminosities:
    L_out = L_p * (hnu_out/hnu_in) * scat_frac; # erg/s


    ### PRODUCE SPECTRUM ##

    # bin photons by energy (demand 5000 packets per bin)
    hnu_bins = np.logspace( np.log10(min(hnu_out)), np.log10(max(hnu_out)), N_p/5000); 
    binsize = np.diff(hnu_bins);
    # figure out which bin each out-packet goes in:
    i_assign = np.digitize( hnu_out, hnu_bins ); 

    # bin L_out to make L_nu 
    L_nu = np.zeros( len(binsize) ); # erg s^-1 Hz^-1
    for i in range( 1, len(hnu_bins) ):
        L_bin = sum( L_out[i_assign==i] );   
        L_nu[i-1] = L_bin;
    L_nu /= binsize ;

    ### PLOT SPECTRUM ##
    if show:
        fm, lblsz = 'serif', 24.;
    
        fig = plt.figure(figsize=(5,4));
        ax = fig.add_axes([0.2,0.2,0.75,0.75]);
    
        ax.set_xlabel(r"$\epsilon_{\mathrm{out}} \; \mathrm{(eV)}$", 
                      family=fm, size=lblsz                          );
        ax.set_ylabel(r"$\mathcal{L}_\nu \; \mathrm{(erg/s)}$", 
                      family=fm, size=lblsz                          );
    
        ax.loglog(hnu_bins[1:],L_nu,'.k'); # plot spectrum
    
        # Make things pretty:
        if type(gam) in [int,float]: xlims = [2.e-4,5.e3];
        else                       : xlims = [2.e-4,8.e5];
        ylims = [3.e37, 6.e41];
        ax.set_xlim(xlims);
        ax.set_ylim(ylims);
    
        plt.show();
   
    return L_nu, hnu_bins;


if __name__=='__main__': 
    spec_name = '/Users/ceharris1/PTF/synapps_spectra/ss_10px_20100114_Keck1_v1.ascii'
    #compton(1.e49,1.e4,True);
    compton(1.e49,spec_name,True);
    #do_this_now();
