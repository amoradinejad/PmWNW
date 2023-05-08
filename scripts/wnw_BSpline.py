#!/usr/bin/env python
# coding: utf-8


import sys, platform, os
import EH_fit as EH
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.special import spherical_jn
import scipy.integrate as integrate
import numpy as np
import math
from timeit import default_timer as timer

FILE = 1
CAMB = 2

sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))


# # Wiggle no-wiggle split using BSpline basis
# Note: This method is based on performing regression to fit the full power spectrum (with wiggles) to a smooth curve. The set of basis for this regression is chosen to be Bspline basis. Zvonomir was using Mathematica's LinearModelFit() to do the regression, while I use the python make_lsq_spline(). Both functions perform the least square fit. For the mathematica function, you can provide any set of basis to use in the fit, while this python function I am using, is specifically for Bspline basis. 

# 1) Fit the ratio of the power spectrum to EH to a smooth curve, and plot the results for different values of Bspline degree and number of knots.

##for Minerva
ka, pka       = np.loadtxt('/Volumes/Data/Documents/Git/gc-wp-nonlinear/linear_spectra_flagship/matter/high_res/flagship_linear_cb_hr_matterpower_z0p0.dat',unpack=True)
logpka        = np.log10(pka)
logka         = np.log10(ka)
logpka_interp = InterpolatedUnivariateSpline(logka,logpka, k=3)


kEH, EHnw = np.loadtxt('/Volumes/Data/Documents/Git/gc-wp-nonlinear/linear_spectra_flagship/matter/high_res/EH_linear_spectra_flagship_high_res_matterpower_z0p0.dat',unpack=True)
logkEH    = np.log10(kEH)
logpkEH   = np.log10(EHnw)
logpkEH_interp = InterpolatedUnivariateSpline(logkEH,logpkEH, k=3)


def EH_nw(k):
    logk = np.log10(k)
    return 10.**logpkEH_interp(logk)

def pk_interp(k):
    logk = np.log10(k)
    return 10.**logpka_interp(logk)



def pk_ratio(k, kf0, pk_camb, pk_switch):
    if(pk_switch == FILE):  
        pkf0   = pk_interp(kf0)
        # result = pk_interp(k)/EH.EH_PS_nw(k, kf0, pkf0)  
        result = pk_interp(k)/EH_nw(k)    
    elif(pk_switch == CAMB):    
        pkf0   = pk_camb.P(0,kf0)
        # result = pk_camb.P(0,k)/EH.EH_PS_nw(k, kf0, pkf0)    
        result = pk_camb.P(0,k)/EH_nw(k)     
    return result


def logpk_nw_Bspline(logk, logpk_ratio, interp):
    p = interp.deg
    m = interp.knot
    n = m-p-1
    knotsPrim = np.concatenate((np.zeros(p+1),np.arange(1., m-2.*p)/(m-2.*p),np.ones(p+1)), axis=None)
    minknots  = np.amin(logk)
    maxknots  = np.amax(logk)  
    knotsSec  = (maxknots - minknots) * knotsPrim + minknots
    wdata     = 1.1 + 1.e6* np.tanh(5.e-4 * (logk + 1.)**16.);
    bsfit     = make_lsq_spline(logk, logpk_ratio, knotsSec, interp.deg, wdata)

    return bsfit

# Note: if you try to fit the ratio of the power spectra over 
# whole range of k, the fit would triply go wrong. You should only choose a limited range of k
# for fitting and then append the rest of kvalues. In other words, you only fit teh 
# part of the power spectrum that has the wiggles with a smooth line, 
# and stich that fit to the original power spectrum for the rest of the k-values. 

def pk_ratio_smooth_interp(interp, pk_camb, pk_switch):
    kv     = interp.kk
    kf_min = interp.kfmin
    kf_max = interp.kfmax

    kL     = kv[(kv < kf_min)]
    kR     = kv[(kv > kf_max)] 
    kf     = kv[(kv >= kf_min) & (kv <= kf_max)]   
    logkf  = np.log10(kf)    
    logpkf = np.log10(pk_ratio(kf,kf0,pk_camb))
    kf0    = kf[0] 
    if(pk_switch == FILE): 
        pkf0   = pk_interp(kf0)
    elif(pk_switch == CAMB):    
        pkf0   = pk_camb.P(0, kf0)

    logpk_ratio_smoothF = logpk_nw_Bspline(logkf, logpkf, interp)
    pk_ratio_smoothF    = 10. ** logpk_ratio_smoothF(logkf)
    pk_ratio_smoothL    = pk_ratio(kL, kf0, pk_camb, pk_switch)
    pk_ratio_smoothR    = pk_ratio(kR, kf0, pk_camb, pk_switch) 
    pk_ratio_smooth     = np.concatenate((pk_ratio_smoothL, pk_ratio_smoothF, pk_ratio_smoothR), axis = None)        
    ratio_smooth_interp = InterpolatedUnivariateSpline(kv, pk_ratio_smooth, k=3)       
    
    return ratio_smooth_interp



def pk_smooth_interp(interp, pk_camb, pk_switch):
    kv     = interp.kk
    kf_min = interp.kfmin
    kf_max = interp.kfmax

    kL     = kv[(kv < kf_min)]
    kR     = kv[(kv > kf_max)] 
    kf     = kv[(kv >= kf_min) & (kv <= kf_max)]     
    logkf  = np.log10(kf)   
    logpkf = np.log10(pk_ratio(kf, kf_min, pk_camb, pk_switch))
    kf0    = kf[0] 
    if(pk_switch == FILE): 
        pkf0   = pk_interp(kf0)
    elif(pk_switch == CAMB):    
        pkf0   = pk_camb.P(0, kf0)

    logpk_ratio_smoothF = logpk_nw_Bspline(logkf, logpkf, interp)
    pk_ratio_smoothF    = 10. ** logpk_ratio_smoothF(logkf)
    pk_ratio_smoothL    = pk_ratio(kL, kf0, pk_camb, pk_switch)
    pk_ratio_smoothR    = pk_ratio(kR, kf0, pk_camb, pk_switch) 
    pk_ratio_smooth     = np.concatenate((pk_ratio_smoothL, pk_ratio_smoothF, pk_ratio_smoothR), axis = None)     
    # pk_smooth           = EH.EH_PS_nw(kv, kf0, pkf0) * pk_ratio_smooth
    pk_smooth           = EH_nw(kv) * pk_ratio_smooth
    pk_smooth_interp    = InterpolatedUnivariateSpline(kv, pk_smooth, k=2)       

    return pk_smooth_interp


## Impose two constraints that the smooth curve should have the same velocity and density dispersion as the original curve

def wG(k,R):
    kR = k * R
    # out = 3.0 * spherical_jn(1, kR)/kR
    out =  3.0 * (np.sin(kR)/kR**3. - np.cos(kR)/kR**2.)
    return out
           
def svlin_integrand(logx, pk_camb, pk_switch):
    x = np.exp(logx)
    if(pk_switch == FILE): 
        pk   = pk_interp(x)
    elif(pk_switch == CAMB):    
        pk   = pk_camb.P(0, x)
    return  x * pk

def sv_integrand(logx, interp, pk_camb, pk_switch):
    x = np.exp(logx)
    a = pk_smooth_interp(interp, pk_camb, pk_switch)(x)
    return x * a

def s8lin_integrand(logx, pk_camb, pk_switch):
    x = np.exp(logx)
    if(pk_switch == FILE): 
        pk   = pk_interp(x)
    elif(pk_switch == CAMB):    
        pk   = pk_camb.P(0, x)
    return  x**3. * wG(x, 8.)**2. * pk

def s8_integrand(logx, interp,  pk_camb, pk_switch):
    x = np.exp(logx)
    a = pk_smooth_interp(interp, pk_camb, pk_switch)(x)
    return x**3. * wG(x, 8)**2. * a

def integ_sv(logkmin, logkmax, interp, pk_camb, pk_switch):
    out = integrate.quad(sv_integrand,logkmin,logkmax, args=(interp, pk_camb, pk_switch),epsrel = 1.e-3, limit=1000)[0]
    sv = 1./(6.*np.pi**2.)*out
    return sv

def integ_s8(logkmin, logkmax, interp, pk_camb, pk_switch):
    out = integrate.quad(s8_integrand,logkmin,logkmax, args=(interp, pk_camb, pk_switch),epsrel = 1.e-3, limit=1000)[0]
    s8 = 1./(2.*np.pi**2.)*out
    return s8

def nknots(x):
    a = 2*x
    n = np.arange(a+1,a+4,1)
    return n


def pk_Bspline_nw(k, interp, pk_camb, pk_switch):
    logkmin = np.log(0.0001)
    logkmax = np.log(10.)

    integ_svlin = integrate.quad(svlin_integrand, logkmin, logkmax, args=(pk_camb, pk_switch), epsrel = 1.e-3,limit=1000)[0]
    integ_s8lin = integrate.quad(s8lin_integrand, logkmin, logkmax, args=(pk_camb, pk_switch), epsrel = 1.e-3,limit=1000)[0]

    svlin = 1./(6.*np.pi**2.)*integ_svlin
    s8lin = 1./(2.*np.pi**2.)*integ_s8lin

    print svlin, np.sqrt(s8lin)

    s8_vec = []
    sv_vec = [] 
    for x in nknots(interp.deg):
        interp.knot = x
        s8_vec = np.append(s8_vec, integ_s8(logkmin, logkmax,  interp, pk_camb, pk_switch))
        sv_vec = np.append(sv_vec, integ_sv(logkmin, logkmax,  interp, pk_camb, pk_switch))
 
        
    rhs = np.array([1., s8lin, svlin])   
    lhs = np.array([np.ones(3), s8_vec, sv_vec])
    res = np.linalg.solve(lhs, rhs)

    i = 0
    pk_smooth = 0.
    for x in nknots(interp.deg):
        interp.knot = x
        pk_smooth += res[i]*pk_smooth_interp(interp,pk_camb, pk_switch)(k)
        i += 1

    return pk_smooth     
