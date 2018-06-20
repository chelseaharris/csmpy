import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import rad_tools as rt
import plotLC_tools as pltLC
import Const as C
from EvolvedModelClass import EvolvedModel as EvMod
from RayClass import Ray

model_dir = '/Users/ceharris1/Science/sedona_hydro/trunk/run/IaCSMhydro/models/'

def main():
    #test_F_nu_calc()
    #test_F_nu_calc_on_ray()
    #construct_inflated_ray(3)
    test_longest_path()

###
# Bremsstrahlung Calculator Tests
###

def test_gaunt_table():
#def main():
    """
    Test that the table of gaunt factors is being loaded in correctly
    """
    FFCalc = rt.BremCalculator()

    gff = np.loadtxt('gauntff.dat') # rows: constant u; columns: constant gamma^2

    N_u, N_g = 146, 81
    gff = gff[:N_u]

    log_u_grid    = np.linspace(-16, -16 + (N_u*0.2), N_u)
    log_gam2_grid = np.linspace( -6,  -6 + (N_g*0.2), N_g)

    xx, yy = np.meshgrid( log_u_grid, log_gam2_grid )

    plt.contourf(-yy.T, xx.T, np.log10(gff), levels=np.linspace(-2,2,30) )
    plt.colorbar()

    plt.contour(-yy.T, xx.T, np.log10(gff), levels=[-1,0,1], colors='k', linewidths=2)

    plt.plot([0,0],[-3,3],'k')
    plt.plot([-3,0],[3,0],'k')
    plt.plot([-4,2],[0,0],'k')
    plt.plot([-4,0],[-2,0],'k')

    plt.ylabel('log $u$')
    plt.xlabel('log $\gamma^2$')

    plt.xlim(-4,2)
    plt.ylim(-3,3)

    #plt.show()

def test_gaunt_func():
#def main():
    """
    Test that the interpolation function is working correctly
    """
    FFCalc = rt.BremCalculator()

    embiggen = 1
    N = 5*embiggen + 1
    x = np.linspace(-3, 3, 5*embiggen + 1)
    y = np.linspace(-2, 4, 5*embiggen + 1)
    #N_u, N_g = 149, 81 
    #x = np.linspace(-16, -16 + (N_u*0.2), N_u) # I want u to correspond to x
    #y = np.linspace( -6,  -6 + (N_g*0.2), N_g) # and gamma^2 to y

    xx = np.repeat(x, N)
    yy = np.tile(y, N)

    gff = FFCalc.gaunt_func.ev(xx, yy).reshape(N, N)

    xx, yy = xx.reshape(N, N), yy.reshape(N, N)

    plt.contourf(-yy, xx, np.log10(gff), levels=np.linspace(-2,2,30))
    plt.colorbar()

    plt.plot([0,0],[-3,3],'k')
    plt.plot([-3,0],[3,0],'k')
    plt.plot([-4,2],[0,0],'k')
    plt.plot([-4,0],[-2,0],'k')

    plt.ylabel('log $u$')
    plt.xlabel('log $1/\gamma^2$')

    plt.xlim(-4,2)
    plt.ylim(-3,3)

    #plt.show()
    

def construct_test_ray(N_cells=2):
    r0 = 1e15
    dr = 1e14
    r1 = np.linspace(r0, r0+N_cells*dr, N_cells)
    r2 = np.linspace(r0+dr, r0+(N_cells+1)*dr, N_cells)
    rho = 1e-19*np.ones(N_cells)
    T = 1e4*np.ones(N_cells)
    p = (rho/C.M_P) * C.K_B*T
    u = 1.5*p
    v = 1e8 * np.ones(N_cells)

    test_ray = Ray( (r1, r2, rho, T, p, v, u), u_per_cc=True)
    test_ray.set_n_e()
    test_ray.set_n_I()

    return test_ray


def construct_inflated_ray(N_cells=2):
    """
    A test ray with an extra empty cell on the outside for
    testing the flux at a larger radius (should go like r^-2)
    """
    r0 = 1e15
    dr = 1e14
    r1 = np.linspace(r0, r0+N_cells*dr, N_cells)
    r2 = np.linspace(r0+dr, r0+(N_cells+1)*dr, N_cells)

    r1 = np.concatenate((r1, [r2[-1]]))
    r2 = np.concatenate((r2, [5*r2[-1]]))

    rho = 1e-19*np.ones(N_cells+1)
    T = 1e4*np.ones(N_cells+1)
    p = (rho/C.M_P) * C.K_B*T
    u = 1.5*p
    v = 1e8 * np.ones(N_cells+1)
    #rho[-1] = 0
    #T[-1] = 0
    #p[-1] = 0
    #u[-1] = 0
    #v[-1] = 0

    test_ray = Ray( (r1, r2, rho, T, p, v, u), u_per_cc=True)
    test_ray.set_n_e()
    test_ray.set_n_I()

    return test_ray


def test_ff_funcs():
    """
    Test that calc_j_nu_therm(), 
    calc_L_nu_therm(),
    calc_al_BB(), and
    calc_al_from_S()
    are all working as expected
    """

    test_ray = construct_test_ray()

    nu = np.linspace(5, 30, 3)*1e9
    
    FFCalc = rt.BremCalculator()

    # The new functions that use the gaunt factor:
    j_nu = FFCalc.calc_j_nu_therm(test_ray, 1, nu)
    al = FFCalc.calc_al_BB(test_ray, 1, nu)
    therm = rt.B_nu(test_ray, nu)

    # related quantities for comparison:
    raltay = 2*C.C_LIGHT**-2 * C.K_B * np.outer(nu**2, T)
    source = j_nu/al

    # Qualitative and quantitative comparisons for expectation:
    diff = np.max(abs(therm - raltay)/therm)
    print('Max fractional difference between blackbody and Raleigh-Taylor: {:.2e}'.format(diff))

    diff = np.max(abs(therm - source)/therm)
    print('Max fractional difference between blackbody and j_nu/alpha: {:.2e}'.format(diff))

    plt.loglog( nu, therm[:,0])
    plt.loglog( nu, raltay[:,0], ls=':' )
    plt.loglog( nu, (j_nu/al)[:,0], ls='--')

    
    # Absorption should lead to alpha propto nu^-2.1 and  flux propto nu^-0.1
    delta_al = al[:2,0]/al[1:,0]
    delta_nu = nu[:2]/nu[1:]
    al_slp = np.log10(delta_al)/np.log10(delta_nu)
    print('Power law slope for alpha propto nu^slope (expect -2.1): '+str(al_slp))

    tau = np.sum(al * (r2 - r1), axis=1)
    flux_abs = (therm.T*tau).T
    delta_f = flux_abs[:2,0]/flux_abs[1:,0]
    f_slp = np.log10(delta_f)/np.log10(delta_nu)
    print( 'Power law slope for flux propto nu^slope (expect -0.1): '+str(f_slp) )

    plt.show()


def test_F_nu_calc():
    # number of cells to make in test ray
    N_ray_cells = 3

    # radiation calculation parameters
    syncalc = rt.SynchrotronCalculator() # a synchrotron calculator; note it assumes p=3 and eps_B = 0.1
    f_NT = 0.01

    nu = np.array([4.9e9, 8.5e9, 15e9, 22.5e9])

    # the test ray 
    ray = construct_test_ray(N_cells=N_ray_cells)

    # resulting quantities
    al = syncalc.calc_alpha(nu, ray, f_NT)
    j_nu = syncalc.calc_j_nu(nu, ray, f_NT)
    S_nu = j_nu/al

    # flux calculation we're testing
    F_nu = rt.calc_F_nu(ray, nu, S_nu, al, N_mu=10)
    print(max(F_nu))

    # show it
    plt.loglog(nu, F_nu, lw=4, color='C0', label = 'flux at outer emitting boundary')

    # put extra vacuum outside explicitly to make sure
    # flux scales like r^-2
    ray = construct_inflated_ray(N_cells = N_ray_cells)

    # resulting quantities
    al = syncalc.calc_alpha(nu, ray, f_NT)
    j_nu = syncalc.calc_j_nu(nu, ray, f_NT)
    S_nu = j_nu/al
    S_nu[:,-1] = 0
    al[:,-1] = 1e-40

    # flux calculation we're testing
    F_nu = rt.calc_F_nu(ray, nu, S_nu, al, N_mu=100)
    print(max(F_nu))
    dilution = 2*( 1 - np.sqrt(1 - (ray.r2[-2]/ray.r2[-1])**2) )
    print(max(F_nu)/dilution)

    # show it
    plt.loglog(nu, F_nu, color='C1', label='flux at 5x distance')
    plt.loglog(nu, F_nu/dilution, ls = '--', color='C2', label='dilution-factor corrected')

    plt.legend()

    plt.show()


def test_longest_path():
    # Load model information
    mod_num = 173
    n_cells = [20000,  681, 12468, 3000] 

    mod_path = model_dir + '/mod_n9s2_{:05d}'.format(mod_num)
    prop_path = '/Users/ceharris1/Science/iSNe/sn2014c/mod{}_dict.pkl'.format(mod_num)

    mod = EvMod(mod_path)
    with open(prop_path,'rb') as rf:
        md = pkl.load(rf)

    # Initialize radiation calculators
    syncalc = rt.SynchrotronCalculator() # a synchrotron calculator; note it assumes p=3 and eps_B = 0.1
    f_NT = 0.01
    ffcalc = rt.BremCalculator()

    nu = np.array([4.9e9, 7e9, 16e9, 22.5e9])

    def calc_I_nu(dtau, S_nu):
        tau_prof = np.cumsum(dtau, axis=1)
        tau_btwn = (tau_prof[:,-1] - tau_prof.T).T

        return np.sum( dtau * S_nu * np.exp(-tau_btwn), axis=1 )
    

    def do_one_bounds(bound_ray):
        # I want to play with changing i_r and i_f
        j_nu = syncalc.calc_j_nu(nu, bound_ray, f_NT) + ffcalc.calc_j_nu_therm(bound_ray, 1, nu)
        al = syncalc.calc_alpha(nu, bound_ray, f_NT) + ffcalc.calc_al_BB(bound_ray, 1, nu)
                    
        S_nu = j_nu/al

        th_longest = np.arcsin( bound_ray.r1[0]/bound_ray.r2[-1] )
        mu_longest = np.cos(th_longest)

        x_out = np.sqrt(bound_ray.r2**2 - bound_ray.r1[0]**2)
        x_in  = np.sqrt(bound_ray.r1**2 - bound_ray.r1[0]**2)
        ds = x_out - x_in

        dtau = al*ds

        # "reflect" the ray to get the back side of the chord
        full_dtau = np.concatenate( (dtau[:,::-1], dtau), axis=1 )
        full_S_nu = np.concatenate( (S_nu[:,::-1], S_nu), axis=1 )
                    
        return calc_I_nu(full_dtau, full_S_nu), mu_longest

    L_nu = np.zeros((nu.size, len(mod)))
    L_nu_long = np.zeros((nu.size, len(mod)))
    L_nu_short = np.zeros((nu.size, len(mod)))

    di = 3 # how much I think the shock indexes could be off by
    # Go through rays calculating F_nu for only the longest path
    for n_ray in range(len(mod)):
        i_r, i_f = md['i_rev'][n_ray], md['i_fwd'][n_ray]

        # Quantitatively check the error with different N_mu
        ray = mod.get_ray(n_ray)
        ray.set_n_e()
        ray.set_n_I()

        # Cut out vacuum, which does not contribute to flux
        if N_vac > 0: 
            # check if cutting out the vacuum will cut out cells we need
            if (len(ray) - N_vac) >= (i_f+di):
                ray = ray[:-N_vac] # if not, go ahead
            else:
                ray = ray[:i_f+di] # if so, use maximum we need as the limit

        # How much are the choices of i_r, i_f affecting the outcome?
        i_r_long, i_f_long = max(0,i_r-di), min(len(ray)-1,i_f+di)
        i_r_short, i_f_short = i_r+di, i_f-di

        I_nu_long , mu_long  = do_one_bounds(ray[ i_r_long : i_f_long ])
        I_nu_short, mu_short = do_one_bounds(ray[i_r_short : i_f_short])
        I_nu      , mu       = do_one_bounds(ray[      i_r : i_f      ])

        L_nu      [:,n_ray] = 2*C.PI*I_nu      *mu       * (4*C.PI*ray.r2[i_f      ]**2)
        L_nu_long [:,n_ray] = 2*C.PI*I_nu_long *mu_long  * (4*C.PI*ray.r2[i_f_long ]**2)
        L_nu_short[:,n_ray] = 2*C.PI*I_nu_short*mu_short * (4*C.PI*ray.r2[i_f_short]**2)


    plt.fill_between(md['t'][1:], L_nu_short[1,1:], L_nu_long[1,1:], alpha=0.5, color='C0', label='longest path '+r'$\pm 6$'+' cells')
    plt.loglog(md['t'][1:], L_nu_short[1,1:], lw=1, color='C0', label='longest path -6 cells')
    plt.loglog(md['t'][1:], L_nu[1,1:], lw=3, color='C4', label='longest path')

    plt.legend()
    plt.xlabel('days since explosion')
    plt.ylabel(r'$\mathcal{L}_\nu$ (7 GHz)')
    plt.show()

def test_F_nu_calc_on_ray():

    ray = Ray.from_file(model_dir+'/mod_n9s2_00170', 10)
    ray.set_n_e()
    ray.set_n_I()

    syncalc = rt.SynchrotronCalculator() # a synchrotron calculator; note it assumes p=3 and eps_B = 0.1
    f_NT = 0.01

    nu = np.array([4.9e9, 8.5e9, 15e9, 22.5e9])

    al = syncalc.calc_alpha(nu, ray, f_NT)
    j_nu = syncalc.calc_j_nu(nu, ray, f_NT)
    S_nu = j_nu/al

    F_nu = rt.calc_F_nu(ray, nu, S_nu, al, N_mu=10)

    plt.loglog(nu, F_nu)
    plt.show()


def main1():
    rhocsm = 1e-16

    rhoref = 7*rhocsm #1e-16
    Rref = 1e16
    delR = 0.1*Rref
    uref = 1e3
    print("T_9 = {}".format(1e-9*(2*C.M_P*uref)/(3*C.K_B*rhoref)))


    props = np.array([[Rref],[Rref+delR],[rhoref],[1.e9],[1.e3],[1.e1],[uref]])
    ray = Ray(props, u_per_cc=True)

    ray.set_n_e()


    nu = np.array([1.e9, 4.9e9, 8.5e9, 15e9, 22.5e9])

    SynCal = rt.SynchrotronCalculator()
    al = SynCal.calc_alpha(nu, ray)
    tau = SynCal.calc_tau(nu, ray)

    print(al)
    print(tau)


def main0():
    SynCal = rt.SynchrotronCalculator()

    mods = [40, 49, 54, 55, 47, 42]
    

    modpath = '/Users/caxen/IaCSM/sedona_hydro/trunk/run/IaCSMhydro/models/'
    
    lcgrp = pltLC.LCGroup.readin(mods, '../postproc/synch_LCs.fwd.txt')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$t/t_\mathrm{imp}$', size=18)

    
    #nu = np.array([4.9e9])
    nu = np.array([22.5e9])
    ax.set_ylabel(r'$\tau_{SSA}(22.5\ \mathrm{GHz})$', size=18)
    
    max_rays_per = 40
    
    colors = sns.color_palette(None, len(mods))
    
    min_tau, max_tau = np.inf, -np.inf
    for i, mod in enumerate(mods):
        moddir = 'mod_n10s0_{:05d}'.format(mod)
        evmod = EvMod(modpath+moddir)
        lc = lcgrp.lc_of(mod)
    
        t2plot = []
        tau2plot = []
    
        raynums = np.logspace(1e-16, np.log10(evmod.size()-1), max_rays_per, dtype=int)
        raynums[0] = 0
        raynums = np.unique(raynums)
            
        for raynum in raynums:
            i_rad0, i_rad1 = lc.i_lo[raynum], lc.i_hi[raynum]
    
            try:
                ray_syn = evmod.get_ray(raynum)[i_rad0:i_rad1]
            except:
                print(mod, raynum)
                continue
            ray_syn.set_n_e()

            tau = SynCal.calc_tau( nu, ray_syn )[0]

    
            t2plot.append(evmod.times[raynum]/evmod.times[0])
            tau2plot.append(tau)
            
            if tau>max_tau: max_tau = tau
            if tau<min_tau: min_tau = tau
                
        ax.loglog(t2plot, tau2plot, lw=2, color=colors[i])
        print(max(tau2plot))
        
        tpeak = lc.t_peak()
        ax.loglog([tpeak/evmod.times[0], tpeak/evmod.times[0]], [1e-16,1e16], color=colors[i], ls='--')
    
    ax.set_xlim(7e-1,8)    
    ax.set_ylim(1e-2*min_tau, 1e1*max_tau)


    plt.show()



if __name__=='__main__': main()
