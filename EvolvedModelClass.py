# A class for accessing the output of a hydro code, particularly for
# linking output times to the output number

import numpy             as np
import Const             as C
import glob

import ChevKasen_tools   as CKtools

from RayClass import Ray

# If you want custom or new time units, add to this dictionary
TimeUnits = {'yr': C.YR2SEC, 'week': C.WEEK2SEC, 'day': C.DAY2SEC, 'hr': C.HR2SEC, 'min':C.MIN2SEC, 's':1. };

class EvolvedModel(object):
    # INITIALIZATION #

    def __init__( self, model_dir, time_fn='times.txt', tunit_des='day' ):
    # INPUTS
    # model_dir  : (string)  directory that houses the hydro output files
    #                        and the file of output times
    # time_fn    : (string)  name of the file of output times (the output
    #                        stream of the hydro code)
    #
        # Make sure everything is legit
        assert model_dir!='';
        assert len(glob.glob(model_dir))==1;

        self.dir = model_dir;

        if time_fn!='':
            times = [];
            with open(model_dir+'/'+time_fn) as rf:
                for line in rf:
                    if line.startswith('WRITING'):
                        times.append( float( line.split()[-1] ) );
            self.times = np.array(times); # times corresponding to ray files
        else:
            rays = glob.glob(model_dir+'/ray_*')
            self.times = np.linspace( 0, len(rays) )

        self.tunit = 's'; # ray files have times in seconds
        if tunit_des!='s': self.set_time_unit(tunit_des);
            
        

    def size(self):
    # Returns number of ray files this model has
        return len(self.times);


    def set_time_unit(self, new_tunit):
    # Express times in different units
    # new_tunit : (string)  unit to change to, 
    #                       either 's', 'hr', 'day', 'week', or 'yr'
        assert new_tunit in TimeUnits.keys();

        if new_tunit==self.tunit: return 0;

        self.times *= TimeUnits[self.tunit]/TimeUnits[new_tunit];
        self.tunit = new_tunit


    def get_times_in(self, unit):
    # Returns the array of times in a different unit 
    # (like set_time_unit() but doesn't change the object)
        assert unit in TimeUnits.keys();
        if unit==self.tunit: return self.tunit;
        else               : return self.times*TimeUnits[self.tunit] / TimeUnits[unit];
    

    def get_mean_step(self, unit):
    # Returns the mean time step in units of unit
        return np.mean( abs(np.diff( self.get_times_in(unit) )) );


    def times_between( self, t1, t2, unit, endpoints=True, index=False ):
    # Returns array of times that fall between 't1' and 't2' or the indeces
    # of these times, with t1 and t2 expressed in units 'unit'
    # INPUTS
    # t1        : (float)  time lower limit
    # t2        : (float)  time upper limit
    # unit      : (string) units of t1 and t2 
    # endpoints : (bool)   whether or not to include the endpoints
    # index     : (bool)   return an array of the *indeces* of the times
    #                      between t1 and t2, INSTEAD of the times themselves
    #
        assert unit in TimeUnits.keys();

        # Compare apples to apples
        if self.tunit != unit: 
            t1 *= TimeUnits[self.tunit]/TimeUnits[unit];
            t2 *= TimeUnits[self.tunit]/TimeUnits[unit];

        btwn = ((self.times>=t1) & (self.times<=t2)) if endpoints else ((self.times>t1) & (self.times<t2));

        return np.where(btwn)[0] if index else self.times[ btwn ] ;


    def index_of( self, t_desired, unit, tol=1. ):
    # Get index of the model time that is nearest to the desired time.
    # Returns -1 if no result is found within tolerance.
    # INPUTS
    # t_desired  : (float)  desired time
    # unit       : (string) units that t_desired is expressed in;
    #                       either 's', 'min', 'hr', 'day', 'week', or 'yr'
    # tol        : (float)  tolerance of result, in units of t_desired
        assert unit in TimeUnits.keys();
        if unit != self.tunit:
            t_desired *= TimeUnits[self.tunit] / TimeUnits[unit];
            tol       *= TimeUnits[self.tunit] / TimeUnits[unit];

        i_closest = np.argmin( abs( self.times - t_desired ) );

        diff = abs( self.times[i_closest] - t_desired );

        return i_closest if diff<=tol else -1; 


    def get_ray_list(self, skip=1):
    # Generate a list of the ray files corresponding to each time
        rays = [ 'ray_{0:05d}'.format(i)  for i in range(self.size())[::skip] ];
        return np.array(rays);


    def get_ray( self, ray_num ):
    # Get the ray and set its time (initialize Ray with proper time)
        assert ray_num < self.size();
        ray_time = self.times[ray_num] * TimeUnits[self.tunit]; # time in seconds
        return Ray.from_file(self.dir, ray_num, ray_time);

    
    def get_ray_at_time(self, t_desired, unit, tol=1.):
    # Get the ray closest to time t_desired within tolerance tol
        i_ray = self.index_of( t_desired, unit, tol );
        return get_ray(i_ray) if i_ray!=-1 else None;
