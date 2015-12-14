#
# Function(s) for finding shock fronts (or any sudden jump)
# in model. Uses two running boxcar averages separated by
# some distance (in pixels), and looks for divergence in
# their values
#


import numpy as np


def find_D2( x, y, start=None, end=None, N_find=3 ):
    assert type(data) in [np.ndarray,list];
    if type(data)==list: data = np.array(data);    # User can put in a list; we'll convert
    assert type(start)==int;
    assert type(end)==int;
    assert type(N_find)==int;

    #   |<- dx[i] ->|<-- dxR1[i] -->|
    #   o-----------o---------------o
    # yL1[i]      y[i]           yR1[i]
    #
    #
    # d2y      1   [ yR1[i]-y[i]     y[i] - yL1[i] ]
    # ___  = _____ [ ___________  -  _____________ ]
    # dx2    dx[i] [  dxR1[i]           dx[i]      ]


    dx = np.diff(x)
    dxR1 = np.roll(dx,-1) # dxR1[i] = dx[i+1] 
    yR1 = np.roll(y,-1) # yR1[i] = y[i+1] -- looks one right
    yL1 = np.roll(y,1)  # yL1[i] = y[i-1] -- looks one left

    if np.any(dx==0): 
        print('At least one cell has zero width. Finer resolution required.')
        return np.zeros(N_find)

    # Can't do endpoints with this method
    d2y = dx[1:-1]**-1 * ( (yR1[1:-1] - y[1:-1])/dxR1[1:-1] - (y[1:-1] - yL1[1:-1])/dx[1:-1] )

    if (start-1) < 0: 
        print('Desired start is too low.')
        start = 1
    elif start==None:
        start = 1
     
    if (end-1) >= len(d2y):
        print('Desired end is too high.')
        end = 0

    # select the section the user wants to use, adjusting the indeces because of the cut-off
    use_d2y = d2y[ start-1 : (end-1 if end!=None else None) ] 

    search_d2y = use_d2y.copy()
    breaks = np.nan*np.ones(N_find)
    for i in range(N_find):
        i_break       = np.argmax( abs(search_d2y) );  
        breaks[i]     = int(i_break)
        #break_amps[i] = np.min( search_diff ) / diff_sig;      
        search_diff[ i_break-3 : i_break+3 ] = 0.; # don't look here anymore

    breaks.sort();
    breaks += int(start);   # now breaks give index of data, not (shorter) average



def find_fronts( data, car_size=13, start=-1, end=-1, N_find=3, car2_size=-1 ):
# INPUTS
# data       : (np.ndarray) values to find break in
# car_size   : (int)        pixel size over which to take averages; must be ODD
# start      : (int)        pixel to start the search (middle of trailing car). 
#                           If -1, goes to function default (see below)
# end        : (int)        pixel to end the search (middle of leading car). 
#                           If -1, goes to function default (see below)
# N_find     : (int)        how many breaks to look for 
# car2_size  : (int)        like car_size but for the second smoothing (one will be
#                           the 'short' car, the other the 'long' car; for default 
#                           set to -1)
#
    # Basically enforce strong typing
    assert type(data) in [np.ndarray,list];
    if type(data)==list: data = np.array(data);    # User can put in a list; we'll convert
    assert type(car_size)==int;
    assert type(start)==int;
    assert type(end)==int;
    assert type(N_find)==int;
    assert type(car2_size)==int;

    # Because of the way we do our boxcar average (with convolve, 'same'),
    # things get screwy unless car_size is odd...
    if car_size%2==0:
        car_size -= 1;
        print( "ALERT: Even-valued car size. Decreasing car size by 1." );

    half_size = car_size/2;  # a useful number: pixels left/right of car center

    # Define short and long cars
    #  Defaults
    if car2_size==-1:
        if half_size==0: long_car_size = 2*car_size+1;  short_car_size = car_size;
        else           : long_car_size = car_size    ;  short_car_size = half_size+1;
    else:
        if car_size > car2_size:   long_car_size=car_size ;   short_car_size=car2_size;
        else                   :   long_car_size=car2_size;   short_car_size=car_size;

    #
    # We will look for fronts using the smoothed data. We might not want to use all
    # the data, for instance because we have a guess about where the front is.
    # We definitely don't want to use the very ends of the data, where the boundaries
    # make the convolution act up. We need to compare the "long" and "short" cars, 
    # so we need to chop off at least the half width of the long car (so that they are 
    # both good everywhere).
    # 'start' and 'end' specify the chunk of data we will use to look for fronts 
    #
    half_long_size = long_car_size/2;
    default_start = half_long_size;             # set default start index (to first possible)
    default_end   = len(data) - half_long_size; # set default end index (to last possible)
    if start==-1:  start = default_start;
    if end  ==-1:  end   = default_end;
    # check [user-defined] start and end before convolving
    if start<default_start:
        print( 'ALERT: Boundary condition issue. Setting \'start\' to default value.' );        
        start = default_start;
    if end > default_end:
        print( 'ALERT: Boundary condition issue. Setting \'end\' to default value.' );
        end = default_end;

    #
    # Take the running (boxcar) average with a wide and narrow window.
    # We do it using the np.convolve() function. We're just
    # convolving the data with a constant value (because we want the 
    # geometric mean). Anyway, with the 'same' keyword on, it will 
    # be the mean at the window's *central* value.
    # We do two convolutions: one "broad" and one "narrow". These will differ
    # only near a very sharp change in the data.
    # This part could probably be improved by making it so you only do the
    # convolution over the necessary data (the range you want w/ half_long_size 
    # wings).
    #
    long_car = np.ones( long_car_size ) * (1./long_car_size);    # this weight will give geometric mean
    long_avg = np.convolve( data, long_car, 'same' );
    short_car = np.ones( short_car_size ) * (1./short_car_size);
    short_avg = np.convolve( data, short_car, 'same' );
    #
    # Chop the ends off!
    #
    long_avg  = long_avg[start:end];
    short_avg = short_avg[start:end];
    # Look at the difference between the *log of the* smoothed curves
    #
    diff = np.log10(long_avg)-np.log10(short_avg);
    diff -= np.mean(diff);
    # Set points where the curves are basically equal, to be equal
    diff_sig = np.std(diff)
    diff[ abs(diff) < 3*diff_sig ] = 0;

    #
    # Look for breaks by the minimum in the difference (maximum negative difference)
    #
    breaks = np.nan*np.zeros(N_find,dtype=int);     # indeces of the breaks
    #break_amps = np.zeros(N_find); # break amplitudes (in units of the standard deviation)
    search_diff = diff.copy();              # a copy diff for the search
    search_diff[ search_diff==0 ] = np.inf
    for i in range(N_find):
        i_break       = np.argmin( search_diff );  
        breaks[i]     = int(i_break)
        #break_amps[i] = np.min( search_diff ) / diff_sig;      
        search_diff[ i_break-long_car_size : i_break+long_car_size ] = np.inf; # don't look here anymore

    breaks.sort();
    breaks += int(start);   # now breaks give index of data, not (shorter) average
    #break_amps.sort();

    # give the ideces back (add 'start' so that it is indeces with respect to the original data);


    return breaks;

