#!/usr/bin/env python
# coding: utf-8

import sys, platform, os
import sys, platform, os
import scipy.integrate as integrate
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import numpy as np
import math
from timeit import default_timer as timer


def pk_nw_integrand(logq, k, kf0, pk_camb, EH):
    q     = 10.**logq
    logk  = np.log10(k)
    a     = 0.25    
    pkf0  = pk_camb.P(0,kf0)
    ratio = pk_camb.P(0,q)/EH.EH_nw(q, kf0, pkf0)  
    out   = ratio * np.exp(-1./(2.*a**2.)*(logk-logq)**2.)
    return out
        
def pk_Gfilter_nw(k, kf0, pk_camb, EH):    
    a       = 0.25
    logqmin =  np.log10(k) - 4.* a
    logqmax =  np.log10(k) + 4.* a
   
    pkf0  = pk_camb.P(0,kf0)
    ratio = 1./(np.sqrt(2.*np.pi)* a)* integrate.quad(pk_nw_integrand, logqmin, logqmax, epsabs=0.,epsrel=1e-6, limit = 10000, args=(k,kf0,pk_camb,EH))[0]
    out   = EH.EH_nw(k, kf0, pkf0) * ratio

    return out

