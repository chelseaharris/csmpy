# This code demonstrates how to use the EvolvedModelClass, 
# RayClass, front_finder, and rad_tools to analyze a simulation.

# Import modules for basic array manipulation
import numpy as np

# Import constants
import Const as C

# Import data-loading tools
from EvolvedModelClass import EvolvedModel as EvMod
from RayClass import Ray

# Import radiation calculation tools
from find_fronts import find_fronts
import rad_tools as rt

def main():
    # Define location of the simulation
    sim_path = './small_sim'

    # Define an EvolvedModel for this simulation.
    # This object is basically a "wrapper" to tie 
    # rays to simulation times.
    evmod = EvMod(sim_path)
    print('This model has {} snapshots.'.format(len(evmod)))
    print('Currently the snapshot time unit is: {}'.format(evmod.tuni))

    # Determine which radiation calculations to perform.
    # Here we will do synchrotron.
    syn_cal = rt.SynchrotronCalculator()

    # We will calculate in the radio at 4.9 and 15.7 GHz:
    freqs = np.array([4.9e9, 15.7e9])

    # Go through the rays 
    for ray_num in evmod.ray_nums:
        # Form our first Ray object by calling the get_ray() function
        ray = evmod.get_ray(ray_num)

        # Set the number density of this gas; we'll assume it's all pure
        # hydrogen for simplicity, so we can use the default functions
        ray.set_n_e()
        ray.set_n_I()

        # Find shock fronts in this ray using the internal energy
        # density, in which there should be two discontinuities
        i_r, i_f = find_fronts(ray.u_gas, N_fronts=2)
        # i_r is the index of the reverse shock, i_f is of the forward shock
        print('There are {} cells in the shock region.'.format(i_f-i_r))

        # Rays can be indexed! Here we will isolate the shocked
        # gas.
        shocked_gas = ray[i_r:i_f]

        # Calculate the emission coefficient of the gas:
        j_nu = syn_cal.calc_j_nu(freqs, shocked_gas)

        # This array is shape (freqs.size, len(shocked_gas))

        # Calculate the optically thin luminosity:
        vols = shocked_gas.cell_volume()
        # We need to multiply the emission coefficient by the cell volumes
        # and then sum up over all cells (columns)
        Lnu_thin = np.sum(j_nu*vols, axis=1)

        # Calculate the optically thick luminosity using ray tracing.
        # This requires the extinction coefficient:
        al = syn_cal.calc_alpha(freqs, shocked_gas)
        # Which then gives a source function:
        S_nu = j_nu/al
        # Which is all we need for the flux calculation:
        F_nu = rt.calc_F_nu(shocked_gas, freqs, S_nu, al)
        # Turn this into a luminosity by multiplying by the emitting area:
        area = 4*C.PI * shocked_gas.r2[-1]**2 # outer radius of the ray 
        Lnu_thick = F_nu * area



if __name__=='__main__': 
    main()
