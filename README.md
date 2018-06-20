# csmpy
CSMpy provides Python tools for the postprocessing of one-dimensional 
simulations of supernovae interacting with circumstellar material (CSM). 
These tools are intended to interface with the hydrodynamic 
extension to the sedona code (which I call "hydrona"; Roth & Kasen, 2014), 
i.e., assume files are as output by that code.
The radiation postprocessing is separate from sedona because the energy source 
(shocks, relativistic electrons) is currently not accommodated by sedona. 

These tools require Python 3, numpy, scipy, as well as astropy (if you want to use the 
provided Const.py). 

## Assumed Format of a Simulation

This code assumes you have created a model in a directory and evolved it with
hydrona, resulting in a directory that contains snapshots of the simulation 
named `ray_<number>`, e.g.,
`my_model/` </br>
  `ray_00000`
  `ray_00001`
  ...
  
