# Useful constants

PI = 3.141592653589793 # np.pi

# Physical constants in cgs
C_LIGHT   = 29979245800.0         # (C.c.to(u.cm/u.s)).value;
H         = 6.6260695700e-27      # (C.h.to(u.erg*u.s)).value;
K_B       = 1.3806488e-16         # (C.k_B.to(u.erg/u.K)).value;
M_P       = 1.672621777e-24       # (C.m_p.to(u.g)).value;
M_E       = 9.10938291e-28        # (C.m_e.to(u.g)).value;
SIGMA_SB  = 5.670373e-05          # (C.sigma_sb.to(u.erg/u.s * u.K**-4 * u.cm**-2)).value;
M_SUN     = 1.9891e+33            # (C.M_sun.to(u.g)).value;
E_ESU     = 4.803204505713468e-10 # (C.cgs.e_esu).value;
A_RAD     = 4* SIGMA_SB / C_LIGHT;
SIGMA_TH  = (8*PI/3) * E_ESU**4 / (M_E**2 * C_LIGHT**4);
M_CH      = 1.4*M_SUN;
GAMMA_AD  = 5./3;
# Physical constants in other units
M_P_GEV   = 0.9382720462814463    # ((C.m_p * C.c**2).to(u.GeV)).value;
M_E_ERG   = M_E * C_LIGHT**2 ;
# Unit conversions
# ... time
YR2SEC    = 31557600.0 # ((1*u.yr).to(u.s)).value;
WEEK2SEC  = 604800.0   # ((1*u.week).to(u.s)).value;
DAY2SEC   = 86400.0    # ((1*u.day).to(u.s)).value;
HR2SEC    = 3600.
MIN2SEC   = 60.
# ... energy
ERG2GEV   = 624.150934326018 # ( (1*u.erg).to(u.GeV) ).value;
ERG2EV    = 624150934326.018 # ( (1*u.erg).to(u.eV ) ).value;
EV2ERG    = 1./ERG2EV
HZ2ERG    = 6.62606957e-27   # ( (C.h*(1*u.Hz)).to(u.erg) ).value;
HZ2EV     = HZ2ERG * ERG2EV;
# ... length
ANG2CM    = 1e-8
KM2CM     = 1e5
KKM2CM    = 1e3*KM2CM

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
