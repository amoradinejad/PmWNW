#!/usr/bin/env python
# coding: utf-8

import sys, platform, os
import EH_fit as EH
import sys, platform, os
import scipy.integrate as integrate
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import numpy as np
import math
from timeit import default_timer as timer


FILE = 1
CAMB = 2

sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))


# Do the wiggle no-wiggle split using the 1d Gaussian filtering in logaithmic space    
ka, pka       = np.loadtxt('/Volumes/Data/Documents/Git/gc-wp-nonlinear/linear_spectra_flagship/matter/high_res/flagship_linear_cb_hr_matterpower_z0p0.dat',unpack=True)
logpka        = np.log10(pka)
logka         = np.log10(ka)
logpka_interp = InterpolatedUnivariateSpline(logka,logpka, k=3)


kEH, EHnw = np.loadtxt('/Volumes/Data/Documents/Git/gc-wp-nonlinear/linear_spectra_flagship/matter/high_res/EH_linear_spectra_flagship_high_res_matterpower_z0p0.dat',unpack=True)
logkEH  = np.log10(kEH)
logpkEH = np.log10(EHnw)
logpkEH_interp = InterpolatedUnivariateSpline(logkEH,logpkEH, k=3)


def EH_nw(k):
    logk = np.log10(k)
    return 10.**logpkEH_interp(logk)

def pk_interp(k):
    logk = np.log10(k)
    return 10.**logpka_interp(logk)


def pk_nw_integrand(logq, k, kf0, pk_camb, pk_switch):
    q     = 10.**logq
    logk  = np.log10(k)
    a     = 0.25 
    if(pk_switch == FILE):
        pkf0  = pk_interp(kf0)
        # ratio = pk_interp(q)/EH.EH_PS_nw(q, kf0, pkf0)  
        ratio = pk_interp(q)/EH_nw(q)  
    elif(pk_switch == CAMB):    
        pkf0  = pk_camb.P(0,kf0)
        # ratio = pk_camb.P(0,q)/EH.EH_PS_nw(q, kf0, pkf0)  
        ratio = pk_interp(q)/EH_nw(q)  
    out   = ratio * np.exp(-1./(2.*a**2.)*(logk-logq)**2.)
    return out
        
def pk_Gfilter_nw(k,kf0,pk_camb, pk_switch):    
    a     = 0.25

    ###For minerva
    logqmin =  np.log10(k) - 4.* a
    logqmax =  np.log10(k) + 4.* a

    if(pk_switch == FILE):
        pkf0  = pk_interp(kf0)
    elif(pk_switch == CAMB):    
        pkf0  = pk_camb.P(0,kf0)
    ratio = 1./(np.sqrt(2.*np.pi)* a)* integrate.quad(pk_nw_integrand, logqmin, logqmax, epsabs=0.,epsrel=1e-6, limit = 10000, args=(k,kf0,pk_camb,pk_switch))[0]
    # out   = EH.EH_PS_nw(k, kf0, pkf0) * ratio
    out   = EH_nw(k) * ratio

    return out

