#
# Functions for determining the power law slope of a
# set of data 
#

import numpy as np

# Find without fitting, point-to-point
def find_powlaw_slope(x,y):
    """
    Assuming y = C * x^alpha
    find alpha via
    log(y) = log(C) + alpha * log(x)
    => d(logy)/d(logx) = alpha
    """
    if np.any(y<=0): 
        print('some y<=0; not from power law')
        return -1

    logx, logy = np.log10(x), np.log10(y)

    dX = np.diff(logx)
    dY = np.diff(logy)

    dYdX = np.zeros(x.size)

    dYdX[1:-1] = 0.5*( dY[1:]/dX[1:] + dY[:-1]/dX[:-1] ) 
    dYdX[0] = dY[0]/dX[0]
    dYdX[-1] = dY[-1]/dX[-1]

    return dYdX



def main():
    """
    Test the slope finder
    """
    x = np.linspace(1,10,10)
    y = x**2
    y[5:] = x[5:]**3

    al = find_powlaw_slope(x,y)
    print(al)


if __name__=='__main__': main()
