import numpy as np

def singlegauss_1d(nu,flux,sigma,nu0):
    return flux*np.exp(-(nu-nu0)**2/(2*sigma**2))

def singlegauss_2d():

def poly(x, coeffs):
    return np.polyval(coeffs, x)

def contspec(nu, flux, coeffs, nu0):
    alpha = 
    return flux*(nu/nu0)**alpha