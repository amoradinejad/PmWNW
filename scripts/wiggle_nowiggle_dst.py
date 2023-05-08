#!/usr/bin/env python
# coding: utf-8

import broadband_extraction as broadband_ben
import wiggle_nowiggle_BSpline as wnw_BSpline
import wiggle_nowiggle_Gfilter as wnw_Gfilter
import EH_fit as EH

import sys, platform, os
import scipy.fftpack as fft
from scipy.interpolate import interp1d,splev,splrep
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline,  UnivariateSpline

### scipy functions to find local minima and maxima of data
from scipy.signal import argrelmin
from scipy.signal import argrelmax
from scipy.signal import find_peaks_cwt
from scipy.signal import gaussian
from scipy.signal import convolve

FILE = 1
CAMB = 2

sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))

ka, pka       = np.loadtxt('/Volumes/Data/Documents/Git/gc-wp-nonlinear/linear_spectra_flagship/matter/high_res/flagship_linear_cb_hr_matterpower_z0p0.dat',unpack=True)
logpka        = np.log10(pka)
logka         = np.log10(ka)
logpka_interp = InterpolatedUnivariateSpline(logka,logpka,k=3)


def pk_interp(k):
    logk = np.log10(k)
    return 10.**logpka_interp(logk)


def pk_dst_nw(k, pk_camb, pk_switch): 
##  k       : wavenumber over which we want to compute the nw spectrum 
### kf      : an array over which to interpolate,
##  pkf_lin : the linear matter power spectrum array for the kf

### The numer of points and sample spacing, as well as kmax, should be chosen 
### such that the sine transform of the PS does not show ringing bbehavior
      count = 0
      while count<1:
            N    = 2**16   #number of sample points  
            T    = 0.005
            kmin = 1.e-4
            kmax = 10.

            kf    = np.linspace(kmin, kmax, N)
            if(pk_switch == FILE):
                  grid  = np.log10(kf*pk_interp(kf))
            elif (pk_switch == CAMB):      
                  grid  = np.log10(kf*pk_camb.P(0,kf))


            # 3 - Perform a fast sine transform of the $\log(kP(k))$-array using the orthonomralized type-II sine transform. Denoting the index of the resulting array by $i$, split the even and off entries. i.e. those entries with even $i$ and odd $i$, into separate arrays.
            grid_sine  = fft.dst(grid, type=2, norm="ortho")
            grid_freqs = fft.fftfreq(grid_sine.shape[0],d=T)


            even_ind  = []
            even_freq = []
            odd_ind   = []
            odd_freq  = []

            even_ind  = grid_sine[0:len(grid_sine):2]
            even_freq = grid_freqs[0:len(grid_freqs):2]
            odd_ind   = grid_sine[1:len(grid_sine):2]
            odd_freq  = grid_freqs[1:len(grid_freqs):2]


            even_freq = np.fft.fftshift(even_freq)
            even_ind  = np.fft.fftshift(even_ind)
            odd_freq  = np.fft.fftshift(odd_freq)
            odd_ind   = np.fft.fftshift(odd_ind)


            # 4 - Linearly interpolate the two arrays separately using cubic splines.

            even = splrep(even_freq, even_ind,s=0)
            odd  = splrep(odd_freq, odd_ind,s=0)

            even_new = splev(even_freq, even, der=0)
            odd_new  = splev(odd_freq, odd, der=0)


            # 5 - Identify baryonic bumps by computing the second derivatie separately for the interpolated even and odd arrays. 

            second_der_even = splev(even_freq, even, der=2)
            second_der_odd  = splev(odd_freq, odd, der=2)


            ### Since the BAO signal is on the positive frequencies, I am limiting the range of frequencies to check for 
            ### the extrema of the second derivative. Otherwise it extrema finder will pick out numerical noise around 
            ### zero frequency. There is perhaps a better way of doing this. 

            even_freq_pos = np.ndarray.flatten(np.argwhere(even_freq>0.1))[0]
            odd_freq_pos  = np.ndarray.flatten(np.argwhere(odd_freq>0.1))[0]


            ### Build new temporary arrays of the frequencies and second derivatives which only span the frequencies > 0.1
            second_der_even_n = second_der_even[even_freq_pos:]
            second_der_odd_n  = second_der_odd[odd_freq_pos:]
            even_freq_n       = even_freq[even_freq_pos:]
            odd_freq_n        = odd_freq[odd_freq_pos:]

            min_even = argrelmin(second_der_even_n,order=10)
            max_even = argrelmax(second_der_even_n,order=10)

            first_min_ind_even  = [min_even[0][0]]
            second_min_ind_even = [min_even[0][1]]
            first_max_ind_even  = [max_even[0][0]]
            second_max_ind_even = [max_even[0][1]]


            min_odd = argrelmin(second_der_odd_n,order=10)
            max_odd = argrelmax(second_der_odd_n,order=10)

            first_min_ind_odd = [min_odd[0][0]]
            second_min_ind_odd = [min_odd[0][1]]
            first_max_ind_odd = [max_odd[0][0]]
            second_max_ind_odd = [max_odd[0][1]]
               


            # Sine we need the full array of the sine transform and not only the part of it we used to find the BAO signal, now 
            # we add back the indecies of the part of the array for frequencies of <0.1. Then remove the array elements to the 
            # right and left that would remove the BAO signal. This is thebest choice I could find to get the relatively clean broadband. The values in Baumann et al did not work. 

            i_min_even = even_freq_pos+first_min_ind_even[0]-2
            i_min_odd = odd_freq_pos+first_min_ind_odd[0]-2
            i_max_even = even_freq_pos+second_max_ind_even[0] + 20
            i_max_odd = even_freq_pos+ second_max_ind_odd[0] + 20


            # ###### 6 - Cut baryonic bumps: remove the elements within the range $[i_{min},i_{max}]$.

            x1e = even_freq[:i_min_even]
            x2e = even_freq[i_max_even:]

            part1_even = even_new[:i_min_even]
            part2_even = even_new[i_max_even:]

            x1o = odd_freq[:i_min_odd]
            x2o = odd_freq[i_max_odd:]

            part1_odd = odd_new[:i_min_odd]
            part2_odd = odd_new[i_max_odd:]


            freq_cute = np.append(x1e, x2e)
            even_cut  = np.append(part1_even, part2_even)

            freq_cuto = np.append(x1o, x2o)
            odd_cut   = np.append(part1_odd, part2_odd)

            ind_arr_e = np.arange(0, len(even_cut),1)
            ind_arr_o = np.arange(0, len(odd_cut),1)

            ### REscaling suggested to reduce the noise
            even_cut_resc = (freq_cute + 1.)**2. * even_cut
            odd_cut_resc  = (freq_cuto + 1.)**2. * odd_cut


            even_resc = splrep(freq_cute, even_cut_resc, s=0,k=3)
            odd_resc  = splrep(freq_cuto, odd_cut_resc, s=0,k=3)

            freqe = np.linspace(min(freq_cute), max(freq_cute), N/2)
            freqo = np.linspace(min(freq_cuto), max(freq_cuto), N/2)

            even_interp = splev(freqe, even_resc, der=0)
            odd_interp  = splev(freqo, odd_resc, der=0)


            ind_arr_e = np.arange(0, len(freqe), 1)
            ind_arr_o = np.arange(0, len(freqo), 1)

            even_interp_N = even_interp/(freqe+1.)**2.
            odd_interp_N  = odd_interp/(freqo+1.)**2.

            freq_tot  = np.append(freqe, freqo)
            array_tot = np.append(even_interp_N, odd_interp_N)

            ind            = np.argsort(freq_tot)
            freq_tot_sort  = freq_tot[ind]
            array_tot_sort = array_tot[ind]


            # 7 - Inverse fast sine transform, leading to a discretized version of $\log[k \; P^{\rm nw}(k)]$.

            logkpk_grid = fft.dst(np.fft.fftshift(array_tot_sort), type=3, norm= "ortho")
            kpk_grid = 10.**(logkpk_grid)
            pk_grid = kpk_grid/kf

            pk_nw_interp = InterpolatedUnivariateSpline(kf,pk_grid,k=3)

            count += 1

      pk_nw  = pk_nw_interp(k)
      
      return pk_nw


