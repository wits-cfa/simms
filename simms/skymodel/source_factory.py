import numpy as np
from simms.constants import FWHM_scale_fact

def singlegauss_1d(freqs, flux, width, nu0):
    """
    Function for a single gaussian line spectrum

    nu: spectral axis
    flux: peak flux
    width: width of peak in frequency units
    n0: frequency at which peak appears
    """
    sigma = width/ FWHM_scale_fact
    return flux*np.exp(-(freqs-nu0)**2/(2*sigma**2))

def singlegauss_2d(x, y, peakflux, x0, y0, a, b, theta):
    """
    Function that creates an ellipse from the cross-section of a 2D gaussian

    x: the x axis of the grid
    y: the y axis of thr grid
    peakflux: peak flux of extended source
    x0, y0: central coordinates of the source in degrees
    a: major axis of ellipse
    b: minor axis of ellipse
    theta: postion angle of the source (measured anticlockwise from the x-axis)
    """
    theta = -np.deg2rad(theta)#added this negative so we an have postive meaning anticlockwise
                                #can also be achieved by switching around a and b by the user 
    x0 = np.deg2rad(x0)
    y0 = np.deg2rad(y0)
    x_prime = x - x0
    y_prime = y - y0
    sigma_x = a / np.sqrt(2 * np.log(2))
    sigma_y = b / np.sqrt(2 * np.log(2))

    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)

    return peakflux * np.exp(- (a * (x_prime**2) + 2 * b * x_prime * y_prime + c * (y_prime**2)))


def poly(x, coeffs):
    return np.polyval(coeffs, x)

#def contspec(chan, flux, coeffs, nu0):
#    alpha = poly(nu, coeffs)
#    return flux*(nu/nu0)**alpha

def contspec(freqs,flux, coeff,nu_ref):
    return flux*(freqs/nu_ref)**(coeff)


