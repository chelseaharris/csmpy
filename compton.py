# Calculate inverse Compton luminosity, assuming single scatter

import numpy                 as np
import scipy                 as spy
#import matplotlib.pyplot     as plt
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


def compton(L_background, T_background, show=False):
# Calculate the optically thin inverse Compton spectrum from a thermal background
# of photons.
# INPUTS
#   L_background : (erg/s) background luminosity
#   T_background : (K)     background temperature
#   show         : (bool)  whether or not to show the spectrum
#
    N_p       = int(1e6);                          # number of photon packets

    ## ELECTRON (SCATTERER) PROPERTIES ##
    n         = -2.5;                              # e- gamma power law index
    gam       = pwrlaw_variate(N_p, n, 10., 100.); # e- Lorentz factor
    beta      = np.sqrt(1 - 1/gam**2);             # electron velocity (units of c)
    
    ## INCOMING PHOTON PROPERTIES ##
    kT        = C.K_B * T_background;                # sauce temperature
    L_SN      = L_background;                      # erg/s lum of (blackbody) photon source
    L_p       = L_SN/N_p;                          # erg/s lum of each photon packet
    tau       = 0.01;                              # optical depth; single-scattering approximation
    scat_frac = 1-np.exp(-tau);                    # scattering fraction (for L_out)
    mu_in     = 1 - 2*np.random.rand(N_p);         # incoming photon directions
    mu_out    = 1 - 2*np.random.rand(N_p);         # outgoing photon directions
    hnu_min   = 0.1*C.EV2ERG;                        # lower energy bound for BB variate
    hnu_max   = 15 *C.EV2ERG;                        # upper energy bound for BB variate
    hnu_in    = thermal_variate( N_p, hnu_min, hnu_max, kT ); # BB dist
    hnu_in   /= C.EV2ERG;                            # convert from erg to eV

    ### PRODUCE SPECTRUM ##

    # calculate outgoing photon energies:
    hnu_out = scatter( gam, beta, hnu_in, mu_in, mu_out ); # eV

    # convert to luminosities:
    L_out = L_p * (hnu_out/hnu_in) * scat_frac; # erg/s


    ### PRODUCE SPECTRUM ##

    # bin photons by energy (demand 5000 packets per bin)
    hnu_bins = np.logspace( np.log10(min(hnu_out)), np.log10(max(hnu_out)), N_p/5000); 
    binsize = np.diff(hnu_bins);
    # figure out which bin each hnu_out goes in:
    i_assign = np.digitize( hnu_out, hnu_bins ); 

    # bin L_out to make L_nu 
    L_nu = np.zeros( len(binsize) ); # erg s^-1 Hz^-1
    for i in range( 1, len(hnu_bins) ):
        L_bin = sum( L_out[i_assign==i] );   
        L_nu[i-1] = L_bin;
    L_nu /= binsize ;


    ### PLOT SPECTRUM ##
    #if show:
    #    fm, lblsz = 'serif', 24.;
    #
    #    fig = plt.figure(figsize=(5,4));
    #    ax = fig.add_axes([0.2,0.2,0.75,0.75]);
    #
    #    ax.set_xlabel(r"$\epsilon_{\mathrm{out}} \; \mathrm{(eV)}$", 
    #                  family=fm, size=lblsz                          );
    #    ax.set_ylabel(r"$\mathcal{L}_\nu \; \mathrm{(erg/s)}$", 
    #                  family=fm, size=lblsz                          );
    #
    #    ax.loglog(hnu_bins[1:],L_nu,'.k'); # plot spectrum
    #
    #    # Make things pretty:
    #    if type(gam) in [int,float]: xlims = [2.e-4,5.e3];
    #    else                       : xlims = [2.e-4,8.e5];
    #    ylims = [3.e37, 6.e41];
    #    ax.set_xlim(xlims);
    #    ax.set_ylim(ylims);
    #
    #    plt.show();
   
    return L_nu, hnu_bins;


if __name__=='__main__': 
    compton(1.e49,1.e4,True);
    #do_this_now();
