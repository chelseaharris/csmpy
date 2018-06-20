# csmpy
CSMpy provides Python tools for the postprocessing of one-dimensional 
simulations of supernovae interacting with circumstellar material (CSM). 
These tools are intended to interface with the hydrodynamic 
extension to the sedona code (which I call "hydrona"; Roth & Kasen, 2014), 
i.e., assume files are as output by that code.
The radiation postprocessing is separate from sedona because the energy source 
(shocks, relativistic electrons) is currently not accommodated by sedona. 

These tools require Python 3, numpy, and scipy.

## Scripts Included

* `Const.py` is basically a header file of quantities -- in cgs units -- that
  are often used in calculations, as well as conversion constants and functions.
  These numbers came from the `astropy` package.
* `ChevKasen_tools.py` combines formulae from Chevalier (1982) and Kasen (2010) that 
  describe supernova ejecta and CSM density profiles, and the interaction between the two. 
* `EvolvedModelClass.py` contains the `EvolvedModel` class. Typically this is 
  used with \s
  `from EvolvedModelClass import EvolvedModel as EvMod`
* `RayClass.py` contains the `Ray` and `Cell` classes as well as function for 
  degrading the resolution of a ray (`merge_cells()`). 
* `rad_tools.py` contains calculators for different types of radiation. 
* `find_fronts.py` provides tools for finding shock fronts. Its main function is
  `find_fronts()` so typically it is imported like: \s
  `from find_fronts import find_fronts` \s
  I encourage caution with aliasing this function as `ff` since that can be
  confused with free-free radiation.
* `find_powlaw.py` simply finds the power law slope between adjacent points
  in _(x,y)_ data.


## Typical Workflow
An example of this workflow is provided in `example/basic_use`.

1. In the simulation directory (which I will refer to as `my_sim`), 
   perform a simulation with hydrona, storing the IO stream to a file 
   called `times.txt`: \s
   `./hydrona > times.txt`
2. Use `EvolvedModelClass` to read in the model information. An EvolvedModel
   is simply a way to link the simulation time with a specific snapshot (`ray_<num>`)
   file.
3. Determine which radiation signatures you want to calculate, load those calculators;
   e.g., \s
   `import rad_tools as rt`\s
   `syn_cal = rt.SynchrotronCalculator()`
4. Use the `EvolvedModel.get_ray()` function to load a snapshot into a `Ray` object,
   which stores the ray properties.
5. Use your calculators to perform necessary radiation calculations on the ray or 
   piece of the ray. 
6. Save calculations to file.
