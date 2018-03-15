# These are tools related to the Chevalier 1982a (C82) and 
# Kasen 2010 (K10) studies/models.
# They include:
# - computing a zone's average radius (as half-mass radius)
# - retrieving Chevalier results, computing Chevalier values
# - creating power law CSM profiles as in Chevalier 1982
# - creating power law ejecta profiles as in Kasen 2010
#

import numpy as np
import Const as C

FOE = 1e51

def calc_zone_r( r_in, r_out ):
    """
    Define characteristic radius of zone (bounds r_in and r_out) 
    by the zone's half-mass (half-volume) radius 
      r_zone^3 - r_in^3 = r_out^3 - r_zone^3
      r_zone^3 = 0.5 * (r_out^3 + r_in^3)

    INPUTS
    r_in (r_out) : numpy.ndarray of zone lower (upper) boundaries
    """
    return ( 0.5*(r_out**3 + r_in**3) )**(1./3);


def calc_zeta_v( n, delta=1 ):
    """
    Calculate the K10 zeta_v 
   
    INPUTS
    n     : (int) outer ejecta power law slope (-n)
    delta : (int) inner ...
    
    """
    return np.sqrt( 2 * (5.-delta)/(3-delta) * (n-5.)/(n-3) );


def calc_zeta_rho( n, delta=1 ):
    """
    Calculate the K10 zeta_rho

    INPUTS
    n     : (int) outer ejecta power law slope (-n)
    delta : (int) inner ...
    """
    return 0.25/np.pi * (n-3)*(3-delta)/(n-delta);


def calc_v_t(n, delta=1, M_ej=C.M_CH, E_ej=FOE):
    """
    Calculate the K10 transition velocity of the ejecta

    INPUTS
    n     : (int)   outer ejecta power law slope (-n)
    delta : (int)   inner ""
    M_ej  : (float) total ejecta mass
    """
    zeta_v = calc_zeta_v(n, delta);
    return 6.e8 * zeta_v * np.sqrt(C.M_CH/M_ej) * np.sqrt(E_ej/1e51);


def calc_r_t(t, n, delta=1, M_ej=C.M_CH, E_ej=FOE):
    """
    Calculate the transition radius according to K10

    INPUTS
    t       : (float) time since explosion (seconds)
    n       : (int)   power law slope of outer ejecta
    delta   : (int)   ...                inner ejecta
    M_ej    : (float) total ejecta mass (grams)
    """
    return t * calc_v_t(n, delta, M_ej, E_ej);


def calc_g_tothe_n(s, n, M_ej=C.M_CH, delta=1, E_ej=FOE):
    """
    Calculate g^n as in C82
      C82:  rho_ej = g^n * t^(n-3) * r^(-n)
      K10:         = zeta_rho * M_ej * v_t^(n-3) * t^(n-3) * r^(-n)
    compare these:
      g^n = zeta_rho * M_ej * v_t^(n-3)

    INPUTS
    s       : (int)   power law slope of CSM
    n       : (int)   ...                outer ejecta
    M_ej    : (float) total ejecta mass (grams)
    delta   : (int)   ...                inner ejecta
    """

    zeta_rho  = calc_zeta_rho(n, delta);
    v_t       = calc_v_t( n, delta, M_ej, E_ej );
    g_tothe_n = zeta_rho * M_ej * v_t**(n-3);

    return g_tothe_n;


def calc_ejecta_q( n, delta=1, r_t=1, which='outer', M_ej=C.M_CH ):
    """
    Calculate q as in 
      rho = q * r^-alpha 
    for the ejecta according to K10

    INPUTS
    n       : (int)     power law slope of outer ejecta
    delta   : (int)     ...                inner ejecta
    r_t     : (float)   transition radius
    which   : (string)  desired region (inner/outer)
    M_ej    : (float)   total ejecta mass
    """
    assert which in ['i','in','inner','o','out','outer'];

    is_outer = (which in ['o','out','outer']);

    zeta_rho = calc_zeta_rho(n, delta);

    pwr_diff = (n if is_outer else delta) - 3;

    return zeta_rho * M_ej * r_t**pwr_diff;


def make_ejecta_rho(r, t, n, delta=1, which='outer', M_ej=C.M_CH, E_ej=FOE):
    """
    Make a K10 model of ejecta density

    INPUTS
    r       : (np.ndarray) radii to calcuate at (cm)
    t       : (float)      time ... (seconds)
    n       : (int)        power law slope of outer ejecta rho
    delta   : (int)        ...                inner ...
    which   : (string)     which region you want
    M_ej    : (float)      total ejecta mass (grams)
    """
    assert which in ['in','inner','out','outer'];
    is_outer = (which in ['out','outer']);

    r_t = calc_r_t(t, n, delta, M_ej, E_ej);
    zeta_rho = calc_zeta_rho(n, delta);

    return calc_ejecta_q( n, delta, r_t, which, M_ej ) * r**(-n if is_outer else -delta);


def calc_ejecta_mass_between( r1, r2, t, n, delta=1, M_ej=C.M_CH, E_ej=FOE ):
    assert r1 >= 0

    # avoid log in the integrand
    assert delta != 3 
    assert n != 3

    # ensure r1 and r2 are proper order
    if r1 > r2: 
        r_hi = r1
        r1 = r2
        r2 = r_hi

    # Calculate transition radius to compare with r1 and r2
    r_t = calc_r_t( t, n, delta, M_ej, E_ej )

    has_inner = ( r1 < r_t ) # any inner ejecta here?
    has_outer = ( r2 > r_t ) # any outer ejecta here?

    zeta_rho = calc_zeta_rho(n, delta)
    C = 4 * np.pi * zeta_rho * M_ej
    M_enc = 0
    if has_inner:
        pwr = 3.-delta
        M_enc += C/pwr * r_t**-pwr * ( min(r2,r_t)**pwr - r1**pwr)
    if has_outer:
        pwr = 3.-n
        M_enc += C/pwr * r_t**-pwr * ( r2**pwr - max(r1,r_t)**pwr )

    return M_enc



def calc_csm_q( s, M_csm, R_lo, R_hi ):
    """
    Calculate q as in 
      rho = q * r^-alpha
    for CSM 

    INPUTS
    s       : (int)   power law slope of shell
    M_csm   : (float) CSM mass (grams)
    R_lo    : (float) lower CSM radial boundary (cm)
    R_hi    : (float) upper CSM radial boundary (cm)
    """
    if s!=3:
        pwr = 3-s;
        return (0.25*pwr/np.pi) * M_csm / (R_hi**pwr - R_lo**pwr);
    elif s==3:
        return (0.25/np.pi) * M_csm / np.log( R_hi/R_lo );


def make_csm_rho( r, q, s ):
    """
    Make model of CSM density: rho = q r^-s

    INPUTS
    r       : (np.ndarray) radii to calcuate at (cm)
    q       : (float)      proportionality constant
    s       : (int)        power law index (-s)
    """
    return q * r**(-s);



def calc_R_c( t, s, n, q, M_ej=C.M_CH, delta=1, E_ej=FOE):
    """
    Calculate the self-similar contact radius as given in C82:
      R_c = ( A g^n / q )^[1/(n-s)] * t^[(n-3)/(n-s)]

    INPUTS
    t       : (float)      model time since explosion (sec)
    s, q    : (int, float) in rho_CSM = q * r^(-s)
    n, M_ej : (int, float) in rho_ej
    """

    A = get_A(s, n);
    g_tothe_n = calc_g_tothe_n(s, n, M_ej, delta, E_ej)

    return (A*g_tothe_n/q)**(1./(n-s)) * t**((n-3.)/(n-s)) ;    


def calc_shell_R_c( t, s, n, f_R, M_csm, M_ej=C.M_CH, delta=1, E_ej=FOE):
    """
    Calculate the self-similar contact radius given a shell of CSM
    with mass M_CSM, inner radius R_c, outer radius f_R*R_c. 

    INPUTS
    t      : (float) time since explosion (seconds)
    s      : (int  ) power law slope of CSM
    n      : (int  ) power law slope of outer ejecta
    f_R    : (float) outer radius of CSM is given by (1+f_R)*R_c
    M_csm  : (float) shell mass (grams)
    M_ej   : (float) total ejecta mass (grams)
    delta  : (int  ) power law slope of inner ejecta
    """
    A        = get_A(s, n);
    g_tothe_n = calc_g_tothe_n( s, n, M_ej, delta, E_ej );

    return  t * ( A * g_tothe_n * (4*np.pi/(3-s)) * ((1+f_R)**(3-s)-1) * (1/M_csm) )**(1./(n-3));


def calc_t_imp( R_c0, s, n, q, M_ej=C.M_CH, delta=1, E_ej=FOE ):
    """                                                                                                                                                                                                   
    Calculates the self-similar impact time given contact discontinuity                                                                                                                                   
    parameters and other parameters as in calc_R_c()                                                                                                                                                      
    """

    A = get_A(s, n)
    g_tothe_n = calc_g_tothe_n(s, n, M_ej, delta, E_ej)

    return (A*g_tothe_n/q)**(-1./(n-3)) * R_c0**((n-s)/(n-3.))


def get_A(s, n):
    """
    Returns C82 value of A
    """
    # From C82 (level 0 key is 's', level 1 key is 'n')
    A_dict = { 0: {6: 2.4 , 7: 1.2  , 8: 0.71,  9: 0.47 , 10:0.33 , 12:0.19 , 14: 0.12 },
               2: {6: 0.62, 7: 0.270, 8: 0.15,  9: 0.096, 10:0.067, 12:0.038, 14: 0.025}
             }
    if s not in A_dict.keys():
        print("No recorded A values for s={:d}".format(s));
        return 0;
    elif n not in A_dict[s].keys():
        print("No recorded A values for s={:d} and n={:d}".format(s,n));
        return 0;
    else:
        return A_dict[s][n];



def get_R1factor(s, n):
    """
    Return the Chevalier result for R_1/R_c

    INPUTS
    s : CSM power law index (-s)
    n : ejecta power law index (-n)
    """
    R1 = { 0: { 6:1.256,  7:1.181, 8:1.154, 9:1.140, 10:1.131,
               12:1.121, 14:1.116 },
           2: { 6:1.377,  7:1.299, 8:1.267, 9:1.250, 10:1.239,
               12:1.226, 14:1.218 }
         };
    return R1[s][n];



def get_R2factor(s, n):
    """
    Return the Chevalier result for R_2/R_c

    INPUTS
    s : CSM power law index (-s)
    n : ejecta power law index (-n)
    """
    R2 = { 0: { 6:0.906,  7:0.935, 8:0.950, 9:0.960, 10:0.966,
               12:0.974, 14:0.979 },
           2: { 6:0.958,  7:0.970, 8:0.976, 9:0.981, 10:0.984,
               12:0.987, 14:0.990 },
         };
    return R2[s][n];



def get_selfsim_vals(s, n):
    """
    Return the Chevalier result for ... everything.

    INPUTS
    s : CSM power law index (-s)
    n : ejecta power law index (-n)
    """

    if s!=0: 
        print("Haven't entered s=2 tables yet.");
        return {};
    if n not in [7,12]: 
        print("Haven't entered in n!=7,12 tables yet.");
        return {};

    x = [];    # r/R_c
    rho = [];  # density
    p = [];    # pressure
    u = [];    # velocity
    if s==0 and n==7:
        x   = [0.935, 0.94 , 0.95 , 0.96 , 0.97 , 0.98 , 0.99 ,
               0.995, 1.00 , 1.01 , 1.02 , 1.04 , 1.06 , 1.08 ,
               1.10 , 1.12 , 1.14 , 1.16 , 1.18 , 1.181]
        rho = [1.336, 1.335, 1.357, 1.315, 1.225, 1.075, 0.826,
               0.621, 0.00 , 0.219, 0.297, 0.411, 0.503, 0.586,
               0.667, 0.746, 0.826, 0.909, 0.995, 1.000]
        p   = [0.471, 0.504, 0.564, 0.615, 0.660, 0.697, 0.727,
               0.738, 0.744, 0.748, 0.755, 0.774, 0.796, 0.821,
               0.849, 0.880, 0.915, 0.954, 0.997, 1.000]
        u   = [1.253, 1.237, 1.211, 1.189, 1.171, 1.155, 1.141,
               1.135, 1.129, 1.117, 1.105, 1.084, 1.065, 1.049,
               1.035, 1.023, 1.013, 1.006, 1.000, 1.000]
    if s==0 and n==12:
        x   = [0.974 , 0.980 , 0.985 , 0.990 , 0.995 , 0.9975,
               1.00  , 1.005 , 1.01  , 1.02  , 1.03  , 1.04  ,
               1.05  , 1.06  , 1.07  , 1.08  , 1.09  , 1.10  ,
               1.11  , 1.121]
        rho = [7.18  , 7.61  , 7.77  , 7.71  , 7.24  , 6.63  ,
               0.00  , 0.583 , 0.651 , 0.729 , 0.781 , 0.821 ,
               0.855 , 0.883 , 0.908 , 0.930 , 0.950 , 0.968 ,
               0.984 , 1.000]
        p   = [0.602 , 0.701 , 0.777 , 0.844 , 0.904 , 0.930 ,
               0.951 , 0.954 , 0.958 , 0.966 , 0.974 , 0.980 ,
               0.986 , 0.991 , 0.995 , 0.998 , 1.000 , 1.001 ,
               1.001 , 1.000]
        u   = [1.255 , 1.235 , 1.222 , 1.210 , 1.199 , 1.194 ,
               1.189 , 1.180 , 1.170 , 1.152 , 1.135 , 1.119 ,
               1.103 , 1.087 , 1.072 , 1.057 , 1.043 , 1.029 ,
               1.015 , 1.000]

    return {'x':x, 'rho':rho, 'P':p, 'v':u};

    
    
def get_plotlims(s, n):
    """
    Return Chevalier's (approximate) plot limits
    """
    avail_s = np.array([0,2]);
    avail_n = np.array([7,12]);
    use_s = avail_s[ np.argmin( abs(avail_s - s) ) ];
    use_n = avail_n[ np.argmin( abs(avail_n - n) ) ];
  
    if use_s==0:
        if use_n==7:
            xlim = [ 0.925, 1.195 ];  ylim = [ 0, 1.5 ];
        elif use_n==12:
            xlim = [ 0.96 , 1.125 ];  ylim = [ 0.2, 10.5 ];
    elif use_s==2:
        if use_n==7:
            xlim = [ 0.91, 1.32 ];  ylim = [ 0.2, 53 ];
        elif use_n==12:
            xlim = [ 0.96, 1.28 ];  ylim = [ 0.2, 350 ];

    return xlim, ylim


def calc_t0_max( s, n, q, M_ej = C.M_CH, v_max = 5e9, delta=1, E_ej=FOE):
    """
    Assuming supernova ejecta does not travel faster than 
    v_max due to physical constraints, calculate the latest
    time a model can interact, assuming the SN density parameter
    g_tothe_n and the CSM density parameter q are fixed givens.

    INPUTS
    s      : (int)    power law slope of CSM
    n      : (int)    ...                outer ejecta
    q      : (float)  CSM density parameter, rho = q * r^-s
    M_ej   : (float)  total ejecta mass (grams)
    v_max  : (float)  maximum ejecta velocity (cm/s)
    delta  : (int)    power law slope of inner ejecta
    """

    assert v_max<=C.C_LIGHT;

    A = get_A(s,n);

    g_tothe_n = calc_g_tothe_n( s, n, M_ej, delta, E_ej );

    return ( v_max**(n-s) * (A * g_tothe_n/q)**-1 )**1/(3.-s) ;


def calc_q_min( s, n, t, M_ej = C.M_CH, v_max = 5e9, delta=1, E_ej=FOE ):
    """
    Assuming supernova ejecta does not travel faster than 
    v_max due to physical constraints, calculate the minimum
    density parameter q that the CSM can have, assuming the SN 
    density parameter g_tothe_n and the time since explosion t
    are fixed

    INPUTS
    s      : (int)    power law slope of CSM
    n      : (int)    ...                outer ejecta
    t      : (float)  time since explosion (sec)
    M_ej   : (float)  total ejecta mass (grams)
    v_max  : (float)  maximum ejecta velocity (cm/s)
    delta  : (int)    power law slope of inner ejecta
    """

    A = get_A(s,n);
    g_tothe_n = calc_g_tothe_n( s, n, M_ej, delta, E_ej );

    return A * g_tothe_n * t**(3-s) * v_max**1./(n-s)


def calc_g_tothe_n_max( s, n, t, q, v_max = 5e9, delta=1 ):
    """
    Assuming supernova ejecta does not travel faster than 
    v_max due to physical constraints, calculate the maximum
    density parameter g^n that the ejecta can have, assuming the 
    CSM density parameter q and the time since explosion t
    are fixed

    INPUTS
    s      : (int)    power law slope of CSM
    n      : (int)    ...                outer ejecta
    t      : (float)  time since explosion (sec)
    q      : (float)  CSM density parameter, rho = q * r^-s
    v_max  : (float)  maximum ejecta velocity (cm/s)
    delta  : (int)    power law slope of inner ejecta
    """

    return q * v_max**(n-s) / (A * t**(3-s))


def calc_E_max( s, n, t, q, M_ej = C.M_CH, v_max = 5e9, delta=1, E_ej=FOE ):
    """
    Assuming supernova ejecta does not travel faster than 
    v_max due to physical constraints, calculate the maximum
    energy the SN ejecta can have, assuming the 
    CSM density parameter q, the time since explosion t,
    and the SN ejecta mass are fixed

    INPUTS
    s      : (int)    power law slope of CSM
    n      : (int)    ...                outer ejecta
    t      : (float)  time since explosion (sec)
    q      : (float)  CSM density parameter, rho = q * r^-s
    M_ej   : (float)  total ejecta mass (grams)
    v_max  : (float)  maximum ejecta velocity (cm/s)
    delta  : (int)    power law slope of inner ejecta
    """

    g_tothe_n = calc_g_tothe_n_max( s, n, t, q, v_max, delta, E_ej );
    zeta_rho  = calc_zeta_rho(n, delta);
    zeta_v    = calc_zeta_v(n,delta);

    return ( g_tothe_n / (zeta_rho * zeta_v**(n-3) * M_ej**(0.5*(5-n)) ) )**(2./(n-3))


def calc_Mej_max( s, n, t, q, v_max = 5e9, delta=1, E_ej = FOE ):
    """
    Assuming supernova ejecta does not travel faster than 
    v_max due to physical constraints, calculate the maximum
    SN ejecta mass can have, assuming the 
    CSM density parameter q, the time since explosion t,
    and the SN ejecta energy are fixed

    INPUTS
    s      : (int)    power law slope of CSM
    n      : (int)    ...                outer ejecta
    t      : (float)  time since explosion (sec)
    q      : (float)  CSM density parameter, rho = q * r^-s
    E_ej   : (float)  total ejecta energy (erg)
    v_max  : (float)  maximum ejecta velocity (cm/s)
    delta  : (int)    power law slope of inner ejecta
    """

    g_tothe_n = calc_g_tothe_n_max( s, n, t, q, v_max, delta, E_ej );
    zeta_rho  = calc_zeta_rho(n, delta);
    zeta_v    = calc_zeta_v(n,delta);

    return ( g_tothe_n / (zeta_rho * zeta_v**(n-3)) )**(2./(5-n))


def approx_runoff_limits(s, n, t_0, t_f, R_c_0):
    """
    # Get the worst-case-scenario shock front positions
    # at t_f
    # INPUTS
    # s       :  (int)   CSM pwr law; rho ~ r^-s
    # n       :  (int)   outer ejecta pwr law; rho ~ r^-n
    # t_0     :  (float) simulation start time
    # t_f     :  (float) simulation end time
    # R_c_0   :  (float) contact discontinuity distance at t_0
    #
    # OUTPUTS
    # R_1_max :  (float) maximum location of forward shock
    # R_2_min :  (float) minimum location of reverse shock
    """
    pwr = float(n-3)/(n-s);
    # self-similar
    R_c_f_min = R_c_0 * (t_f/t_0)**pwr;
    # free expansion
    R_c_f_max = R_c_0 * (t_f/t_0);

    # Forward shock
    R_1_max = get_R1factor(s,n) * R_c_f_max;
    # Reverse shock
    R_2_min = get_R2factor(s,n) * R_c_f_min;

    return R_1_max, R_2_min;


def approx_time_fall(s, n, t_0, R_c_0, R_out_0, do_fwd=True, moving_medium=False):
    """
    Approximate time for forward shock front to 
    hit simulation edge, assuming constant velocity motion
    for outer edge if moving_medium=True
    
    INPUTS
    s             :  (int)   CSM pwr law; rho ~ r^-s
    n             :  (int)   outer ejecta pwr law; rho ~ r^-n
    t_0           :  (float) simulation start time
    R_c_0         :  (float) contact discontinuity distance at t_0
    R_out_0       :  (float) outer radius/final position of shock front
    do_fwd        :  (bool)  whether to solve for the forward shock (does reverse if this is False)
    moving_medium :  (bool)  whether the material that the shock is moving through is moving
    
       If moving at constant speed R_out_0/t_0 then 
       R_out_f / t_f = R_out_0 / t_0 ==> R_out_f = R_out_0 * (t_f/t_0)
    
       R_1_f = R1factor * R_c_f = R1factor * R_c_0 * (t_f/t_0)**(n-3/n-s)
    
       Condition:
       R_out_f = R_1_f
       If moving:
         R_out_0 * (t_f/t_0) = R1factor * R_c_0 * (t_f/t_0)**(n-3/(n-s))
         (t_f/t_0)**(3-s/(n-s)) = R1factor * R_c_0/R_out_0
         t_f = t_0 * ( R1factor * R_c_0/R_out_0 )**(n-s/(3-s))
       Else:
         R_out_0 = R1factor * R_c_0 * (t_f/t_0)**(n-3/(n-s))
         (t_f/t_0)**(n-3/(n-s)) = R1factor * R_c_0/R_out_0
         t_f = t_0 * (R1factor * R_c_0/R_out_0)**(n-s/(n-3))
    """

    # reverse shock should be going through freely explanding medium     
    if not do_fwd: moving_medium = True;  

    alpha = float(n-3)/(n-s);
    pwr = ( 1/(-alpha) )    if not moving_medium else ( 1/(1-alpha) );
    f   = get_R1factor(s,n) if do_fwd            else get_R2factor(s,n)   ;

    return t_0 * ( f * R_c_0/R_out_0 )**pwr;


def evolve_R_c(t_f, s, n, t_0, R_c_0):
    """
    Given R_c at some time t_0, calculate R_c at a later time t_f
    """
    pwr = float(n-3)/(n-s);
    return R_c_0 * (t_f/t_0)**pwr;
    
