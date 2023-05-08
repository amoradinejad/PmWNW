
"""
Developed by Azadeh Moradinezhad Dizgah, 2020

This script computes the EH fit (astro-ph/9709112) for wiggle and no-wiggle split of the 
matter power spectrum, assuming LCDM model

"""

#!/usr/bin/env python
# coding: utf-8

import sys, platform, os
import scipy.fftpack as fft
from scipy.integrate import quad
from scipy.special import spherical_jn as jn
from scipy.interpolate import interp1d,splev,splrep, InterpolatedUnivariateSpline
from scipy.interpolate import make_lsq_spline, BSpline
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.special import spherical_jn
import scipy.integrate as integrate


"""
1.1 EH setup

First, we set up the constants needed for the computation as global variables.
Parmaters (list of symbols given in Table of EH97) 

"""


# Set the values of cosmological parmaeters. Minerva power spectrum
H0    = 67.
h     = H0/100.
ombh2 = 0.049*h**2.
omch2 = (0.319- 0.049)*h**2.
ns    = 0.96

# Set other necassary constants
HH0   = 1.e3*H0/299792458.    # H0 value devided by the speed of light
om0h2 = omch2 + ombh2
om0   = om0h2/h**2.
# theta = 2.728/2.7           # COBBE-FIRAS value I was using before
theta = 2.7255/2.7            # The value used bby Martin Crocce.

# Set the length and time scales
zeq    = 2.5e4*om0h2*theta**(-4.) 
keq    = np.sqrt(2.*om0*HH0**2.*zeq)
kSilk  = 1.6*ombh2**0.52*om0h2**0.73*(1.+(10.4*om0h2)**(-0.95))  ###in 1/Mpc
bb1    = 0.313*om0h2**(-0.419)*(1.+0.607*(om0h2)**(0.674)) 
bb2    = 0.238*om0h2**0.223
zd     = 1291. * om0h2**0.251/(1.+0.659*om0h2**0.828)*(1.+bb1*(ombh2**bb2))

# Set a bunch of ther constants too.
a1      = (46.9*om0h2)**0.670*(1.+(32.1*om0h2)**(-0.532))
a2      = (12.0*om0h2)**0.424*(1.+(45.*om0h2)**(-0.582))
alphac  = a1 ** (-ombh2/om0h2) * a2**(-((ombh2/om0h2)**3.))
b1      = 0.944*(1.+(458.*om0h2)**(-0.708))**(-1.)
b2      = (0.395 * om0h2)**(-0.0266)
betac   = (1. + b1*((omch2/om0h2)**b2-1.))**(-1.)

# Parmaeters for baryon transfer function
beta_node = 8.41*om0h2**0.435
betab     = 0.5 + ombh2/om0h2 + (3.-2.*ombh2/om0h2) * np.sqrt((17.2*om0h2)**2+1.) 

def R(z):
    return 31.5 * ombh2*theta**(-4.)*(z/1.e3)**(-1.)

def cs(): 
    # s      = 2./(3.*keq)*np.sqrt(6./Req)*np.log((np.sqrt(1.+Rd)+np.sqrt(Rd+Req))/(1.+np.sqrt(Req)))   # The exact equation for the sound speed in Eq. 2 of EH that I used initially  
    # Note: If I use this compared to the next equation, I get some reisual at the k~0.01, so I am using the next one
    s = 44.5 * np.log(9.83/om0h2)/np.sqrt(1.+10.*ombh2**(3./4.))    ## approximate sound speed given in Eq. (26) of EH, used by Martin 
    return cs

# parameters for small-scale limit
def qq(k):
    out = k*theta**2.*om0h2**(-1.)
    return out
                                            
#  TODO: Check what was this function returning?
def G(y):
    out = y*(-6.*np.sqrt(1.+y)+(2.+3.*y)*np.log((np.sqrt(1.+y)+1.)/(np.sqrt(1.+y)-1.)))
    return out
                                            
def alphab_func():
    s       = cs()
    Req     = R(zeq)
    Rd      = R(zd)
    y       = (1.+zeq)/(1.+zd)
    alphab  = 2.07*keq*s*(1.+Rd)**(-3./4.)*G((1.+zeq)/(1.+zd))
    return alphab



"""
1.2. Compute the total, baryon and CDM and zero-baryon transfer functions

"""

# CDM transfer function
def f(k):
    s   = cs()
    out = 1./(1.+(k*s/5.4)**4.)
    return out

def C(k, x1):  ### x1=alphac
    out = 14.2/x1+386./(1.+69.9*qq(k)**(1.08))
    return out

def Tt0(k, x1, x2):  ##x1 = alphac, x2 = betac
    out = np.log(math.e+1.8*x2*qq(k))/(np.log(math.e+1.8*x2*qq(k))+C(k, x1)*qq(k)**2.)
    return out 

def Tc(k): 
    out = f(k)*Tt0(k,1.,betac) + (1.-f(k)) *Tt0(k,alphac,betac)                                            
    return out

def st(k):
    s   = cs()
    out = s/(1.+ (beta_node/(k*s))**3.)**(1./3.)                                           
    return out

def Tb(k):
    s       = cs()
    alphab  =  alphab_func()
    out     = (Tt0(k,1.,1.)/(1.+(k*s/5.2)**2.) + alphab/(1.+(betab/(k*s))**3.) * np.exp(-(k/kSilk)**1.4))*jn(0,k*st(k)) 
    return out

# Total baryon+CDM transfer function
def T(k):
    s   = cs()
    out = ombh2/om0h2 * Tb(k,s) + omch2/om0h2 * Tc(k)
    return out

def Gamma(k):
    s   = cs()
    AG  = 1. - 0.328*np.log(431.*om0h2)*ombh2/om0h2 + 0.38*np.log(22.3*om0h2)*(ombh2/om0h2)**2.                                            
    out = om0 * h * (AG + (1. -AG)/(1.+(0.43*k*s)**4.))
    return out
                     
def q(k):
    out = k/h * theta**2./Gamma(k)                                        
    return out

def T0(k):
    out = L0(k)/(L0(k)+C0(k)*q(k)**2.)
    return out

def L0(k):
    out = np.log(2.*math.e + 1.8 * q(k))
    return out

def C0(k):
    out = 14.2 + 731./(1.+62.5*q(k)) 
    return out      


"""
1.3. Compute the EH power spectrum

"""

def EH_PS_w(k,k0,p0):
    kh   = h*k
    kh0  = h*k0
    norm = p0/(k0**ns*T(kh0)**2.)
    out  = k**ns * T(kh)**2. * norm
    return eh_w

def EH_PS_nw(k,k0,p0):
    kh   = h*k
    kh0  = h*k0
    norm = p0/(k0**ns*T(kh0)**2.)
    out  = k**ns * T0(kh)**2. * norm
    return eh_nw

