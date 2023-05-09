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



class EHFits:
    """
    This class computes the all the necasssary functions for computing the Eisentein-Hu fit for 
    wiggle and no-wiggle components of the matter power spectrum.

    List of the symbols used below are given in Table of EH97. 

    """
    # 1.1 EH setup
    def __init__(self, h, ombh2, omch2, ns):
        self.h = h
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.ns = ns

        # Set other necassary constants
        self.om0h2 = omch2 + ombh2
        self.om0 = self.om0h2/self.h**2.
        self.HH0 = 1.e5*self.h/299792458.
        self.theta = 2.7255/2.7  # The value used by Martin Crocce.

        # Set the length and time scales
        self.zeq = 2.5e4*self.om0h2*self.theta**(-4.)
        self.keq = np.sqrt(2.*self.om0*self.HH0**2.*self.zeq)
        self.kSilk  = 1.6*self.ombh2**0.52*self.om0h2**0.73*(1.+(10.4*self.om0h2)**(-0.95))  ###in 1/Mpc
        self.bb1 = 0.313*self.om0h2**(-0.419)*(1.+0.607*(self.om0h2)**(0.674)) 
        self.bb2 = 0.238*self.om0h2**0.223
        self.zd = 1291. * self.om0h2**0.251/(1.+0.659*self.om0h2**0.828)*(1.+self.bb1*(self.ombh2**self.bb2))
        
        # Set a bunch of ther constants too.
        self.a1 = (46.9*self.om0h2)**0.670*(1.+(32.1*self.om0h2)**(-0.532))
        self.a2 = (12.0*self.om0h2)**0.424*(1.+(45.*self.om0h2)**(-0.582))
        self.alphac = self.a1 ** (-self.ombh2/self.om0h2) * self.a2**(-((self.ombh2/self.om0h2)**3.))
        self.b1 = 0.944*(1.+(458.*self.om0h2)**(-0.708))**(-1.)
        self.b2 = (0.395 * self.om0h2)**(-0.0266)
        self.betac = (1. + self.b1*((self.omch2/self.om0h2)**self.b2-1.))**(-1.)
        
        # Parmaeters for baryon transfer function
        self.beta_node = 8.41*self.om0h2**0.435
        self.betab     = 0.5 + self.ombh2/self.om0h2 + (3.-2.*self.ombh2/self.om0h2) * np.sqrt((17.2*self.om0h2)**2+1.) 


    def R(self,z):
        return 31.5 * self.ombh2*self.theta**(-4.)*(z/1.e3)**(-1.)

    def cs(self): 
        # s      = 2./(3.*keq)*np.sqrt(6./Req)*np.log((np.sqrt(1.+Rd)+np.sqrt(Rd+Req))/(1.+np.sqrt(Req)))   # The exact equation for the sound speed in Eq. 2 of EH that I used initially  
        # Note: If I use this compared to the next equation, I get some reisual at the k~0.01, so I am using the next one
        s = 44.5 * np.log(9.83/self.om0h2)/np.sqrt(1.+10.*self.ombh2**(3./4.))    ## approximate sound speed given in Eq. (26) of EH, used by Martin 
        return s

    # parameters for small-scale limit
    def qq(self,k):
        out = k*self.theta**2.*self.om0h2**(-1.)
        return out
                                                
    #  TODO: Check what was this function returning?
    def G(self,y):
        out = y*(-6.*np.sqrt(1.+y)+(2.+3.*y)*np.log((np.sqrt(1.+y)+1.)/(np.sqrt(1.+y)-1.)))
        return out
                                                
    def alphab_func(self):
        s       = self.cs()
        Rd      = self.R(self.zd)
        y       = (1.+self.zeq)/(1.+self.zd)
        out     = 2.07*self.keq*s*(1.+Rd)**(-3./4.)*self.G((1+self.zeq)/(1+self.zd))
        return  out

    """
    1.2. Compute the total, baryon and CDM and zero-baryon transfer functions

    """

    # CDM transfer function    
    def f(self,k):
        s   = self.cs()
        out = 1./(1.+(k*s/5.4)**4.)
        return out

    def C(self,k, x1):  ### x1=alphac
        out = 14.2/x1+386./(1.+69.9*self.qq(k)**(1.08))
        return out

    def Tt0(self,k, x1, x2):  ##x1 = alphac, x2 = betac
        out = np.log(math.e+1.8*x2*self.qq(k))/(np.log(math.e+1.8*x2*self.qq(k))+self.C(k, x1)*self.qq(k)**2.)
        return out 

    def Tc(self,k): 
        out = self.f(k)*self.Tt0(k,1.,self.betac) + (1.-self.f(k)) *self.Tt0(k,self.alphac,self.betac)                                            
        return out

    def st(self,k):
        s   = self.cs()
        out = s/(1.+ (self.beta_node/(k*s))**3.)**(1./3.)                                           
        return out

    def Tb(self,k):
        s       = self.cs()
        alphab  =  self.alphab_func()
        out     = (self.Tt0(k,1.,1.)/(1.+(k*s/5.2)**2.) + alphab/(1.+(self.betab/(k*s))**3.) * np.exp(-(k/self.kSilk)**1.4))*jn(0,k*self.st(k)) 
        return out

    # Total baryon+CDM transfer function
    def T(self,k):
        out = self.ombh2/self.om0h2 * self.Tb(k) + self.omch2/self.om0h2 * self.Tc(k)
        return out

    def Gamma(self,k):
        s   = self.cs()
        AG  = 1. - 0.328*np.log(431.*self.om0h2)*self.ombh2/self.om0h2 + 0.38*np.log(22.3*self.om0h2)*(self.ombh2/self.om0h2)**2.                                            
        out = self.om0 *self.h * (AG + (1. -AG)/(1.+(0.43*k*s)**4.))
        return out
                         
    def q(self,k):
        out = k/self.h * self.theta**2./self.Gamma(k)                                        
        return out

    def T0(self,k):
        out = self.L0(k)/(self.L0(k)+self.C0(k)*self.q(k)**2.)
        return out

    def L0(self,k):
        out = np.log(2.*math.e + 1.8 * self.q(k))
        return out

    def C0(self,k):
        out = 14.2 + 731./(1.+62.5*self.q(k)) 
        return out      


    """
    1.3. Compute the EH power spectrum

    """

    def EH_tot(self,k,k0,p0):
        kh    = self.h*k
        kh0   = self.h*k0
        norm  = p0/(k0**self.ns*self.T(kh0)**2.)
        eh_w  = k**self.ns * self.T(kh)**2. * norm
        return eh_w

    def EH_nw(self,k,k0,p0):
        kh    = self.h*k
        kh0   = self.h*k0
        norm  = p0/(k0**self.ns*self.T(kh0)**2.)
        eh_nw = k**self.ns * self.T0(kh)**2. * norm
        return eh_nw





