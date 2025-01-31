"""A suite of utility functions"""

import numpy as np

def cartesian_to_cylindrical(coords):
    """Return cylindrical phase coordinates given cartesian coordinates"""
    x, y, z, vx, vy, vz = coords.T
    rho = np.sqrt(x**2. + y**2.)
    phi = np.arctan2(y, x)
    vrho = (x*vx + y*vy)/rho
    vphi = rho*(x*vy - y*vx)/rho**2.

    return np.array([rho, phi, z, vrho, vphi, vz]).T

