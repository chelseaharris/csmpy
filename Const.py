# Useful constants

PI = 3.141592653589793 # np.pi

# Physical constants in cgs
C_LIGHT   = 29979245800.0          # speed of light     (C.c.to(u.cm/u.s)).value;
H         = 6.6260695700e-27       # Planck's constant  (C.h.to(u.erg*u.s)).value;
K_B       = 1.3806488e-16          # Boltzman contstant (C.k_B.to(u.erg/u.K)).value;
M_P       = 1.672621777e-24        # proton mass        (C.m_p.to(u.g)).value;
M_E       = 9.10938291e-28         # electron mass      (C.m_e.to(u.g)).value;
SIGMA_SB  = 5.670373e-05           # Stefan-Boltzman    (C.sigma_sb.to(u.erg/u.s * u.K**-4 * u.cm**-2)).value;
M_SUN     = 1.9891e+33             # solar mass         (C.M_sun.to(u.g)).value;
E_ESU     = 4.803204505713468e-10  # electron charge    (C.cgs.e_esu).value;
A_RAD     = 4* SIGMA_SB / C_LIGHT;                       # radiation constant 
SIGMA_TH  = (8*PI/3) * E_ESU**4 / (M_E**2 * C_LIGHT**4); # Thomson cross-section
M_CH      = 1.4*M_SUN;                                   # non-rotating Chandrasekhar mass
GAMMA_AD  = 5./3;                                        # ideal gas adiabatic coefficient
SIGMA_TOT = PI*E_ESU**2 / (M_E*C_LIGHT)                  # atomic line profile nomalization [cm^2 Hz]
RYD_EN    = 2.17992e-11            # Rydberg energy/H ionization energy (erg) from Allen AQ
# Physical constants in other units
M_P_GEV   = 0.9382720462814463     # proton mass in GeV ((C.m_p * C.c**2).to(u.GeV)).value;
M_E_ERG   = M_E * C_LIGHT**2 ;     # electron mass in erg
# Unit conversions
# ... time
YR2SEC    = 31557600.0 # ((1*u.yr).to(u.s)).value;
WEEK2SEC  = 604800.0   # ((1*u.week).to(u.s)).value;
DAY2SEC   = 86400.0    # ((1*u.day).to(u.s)).value;
HR2SEC    = 3600.
MIN2SEC   = 60.
# ... energy
ERG2GEV   = 624.150934326018 # ( (1*u.erg).to(u.GeV) ).value;
ERG2KEV   = 624150934.326018 # ( (1*u.erg).to(u.keV )).value;
ERG2EV    = 624150934326.018 # ( (1*u.erg).to(u.eV ) ).value;
EV2ERG    = 1./ERG2EV
HZ2ERG    = 6.62606957e-27   # ( (C.h*(1*u.Hz)).to(u.erg) ).value;
HZ2EV     = HZ2ERG * ERG2EV;
EV2HZ     = 1/HZ2EV;
MET2ERG   = HZ2ERG * C_LIGHT/(1e2) # meters to erg
# ... length
ANG2CM    = 1e-8
KM2CM     = 1e5
KKM2CM    = 1e3*KM2CM
PC2CM     = 3.08567758149137e18

# More Complicated Conversions
# ... wavelength/frequency
Ang2Hz      = lambda lm_Ang: C_LIGHT/(lm_Ang*ANG2CM) [::-1]
Ang2erg     = lambda lm_Ang: H*Ang2Hz(lm_Ang)
# ... specific fluxes or luminosities
F_Ang2F_Hz  = lambda lm_Ang, f_Ang: lm_Ang**2/C_LIGHT * f_Ang * ANG2CM # (f_Ang/ANG2CM) * (lm_Ang*ANG2CM)**2
F_Ang2F_erg = lambda lm_Ang, f_Ang: F_Ang2F_Hz(lm_Ang, f_Ang) / H
MJY2CGS = 1e-26
# ... times
def TimeConvert( a_time, a_u_now, a_u_des ):
    Time_to_Sec = {'yr'  : YR2SEC, 
                   'week': WEEK2SEC, 
                   'day' : DAY2SEC, 
                   'hr'  : HR2SEC, 
                   'min' : MIN2SEC, 
                   's'   : 1. };    

    assert a_u_now in Time_to_Sec.keys();
    assert a_u_des in Time_to_Sec.keys();

    conv_fact = Time_to_Sec[a_u_now] / Time_to_Sec[a_u_des] 

    return a_time * conv_fact;
