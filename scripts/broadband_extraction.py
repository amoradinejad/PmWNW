#!/usr/bin/env python
# coding: utf-8

# # Broadband Extraction


"""Function to decompose the matter power spectrum in its broadband and oscillatory parts

The implementation is discussed and detailed in Appendix C.1 of arXiv:1712.08067.
Please cite both this paper (arXiv:1712.08067) and the paper proposing the underlying
method (arXiv:1003.3999) when using this code.
"""


# ## Preliminaries

# Make numerical tools available:
import numpy as np
from scipy import interpolate
from scipy import fftpack


__author__ = 'Benjamin Wallisch'
__copyright__ = 'Copyright 2020'
__license__ = 'GPL'
__version__ = '1.0.1'
__maintainer__ = 'Benjamin Wallisch'
__email__ = 'bwallisch@ias.edu'
__status__ = 'Production'


# ## Broadband Extraction Function

def broadband_extraction(k_range, pk_array, points=10, k_max_extr=1.0):
    """Decompose the matter power spectrum into its broadband/no-wiggle and oscillatory/wiggle parts
    using discrete spectral analysis following Appendix A.1 of arXiv:1003.3999.
    (Default values optimized for speed and output wavenumbers in the range [0.003,0.6] h/Mpc.)

    Parameters
    ----------
    k_range : ndarray
        Range of wavenumbers k in [k_{min}, k_{max}] with k_{min} leq 10^{-3} h/Mpc
        and k_{max} geq 1 h/Mpc.

    pk_array : ndarray
        Matter power spectrum P(k) corresponding to wavenumbers k.

    points : int
        Number of sample points: 2^points

    k_max_extr : float
        Maximum wavenumber used for extraction

    Returns
    -------
    ndarray
       Range of wavenumbers k in [k_{min}, k_{max}] with k_{min} leq 10^{-3} h/Mpc
       and k_{max} geq 1 h/Mpc. (Same as input.)

    ndarray
       Matter power spectrum P(k) corresponding to wavenumbers k. (Same as input.)

    ndarray
       Broadband/no-wiggle/smooth matter power spectrum P^{nw}(k) corresponding
       to input wavenumbers k.

    ndarray
       Oscillatory/wiggle matter power spectrum P^{w}(k) corresponding to input wavenumbers k.

    tuple
       Tuple (tilde{k}_{min}, tilde{k}_{max}) denoting range of wavenumbers
       [tilde{k}_{min}, tilde{k}_{max}] in which the broadband extraction is valid.
       Outside this range, we have P^{w}(k) = 0 and P^{nw}(k) = P(k).
    """

    # Hard-coded extraction parameters
    x_max_shift_even = 10
    x_max_shift_odd = 20
    x_min_shift = 3

    # if k_range[-1] < k_max_extr:
    #     raise ValueError(f"Supply a range of k up to at least k = {k_max_extr} h/Mpc.")

    # Needed functions
    def pk_func(logplogk, k):
        """Interpolated P(k)"""
        return np.exp(logplogk(np.log(k)))


    def sample_lnk_pk(logplogk, kmin, kmaxextr, n_samps):
        """Sample ln(k P(k)) in 2^n_samps points, equidistant, i.e. lin-spaced, in k"""
        karray = np.linspace(kmin, kmaxextr, num=2**n_samps)
        lnkpk_array = np.log(karray*pk_func(logplogk, karray))
        return karray, lnkpk_array


    def fast_sine_transform(lnkpk_array):
        """Fast sine transform of ln(k*P(k))"""
        array = fftpack.dst(lnkpk_array, type=2, norm='ortho')
        even_array = array[0::2]
        odd_array = array[1::2]
        return even_array, odd_array


    def cut_out_and_interpolate(array, even_or_odd):
        """Detect and cut the baryonic bump, then interpolate"""
        def second_deriv_averaged(sec_deriv, num, size):
            """Averaged second derivative to get rid of numerical noise/..."""
            if num in [0, size]:
                averaged = sec_deriv(num)
            else:
                averaged = (sec_deriv(num-1)+sec_deriv(num+1))/2.
            return averaged

        length = len(array)
        x_range = np.arange(length)
        assert(even_or_odd in ['even', 'odd'])
        if even_or_odd == 'even':
            x_max_shift = x_max_shift_even
        elif even_or_odd == 'odd':
            x_max_shift = x_max_shift_odd
        else:
            raise ValueError("even_or_odd has to be either even or odd.")

        second_derivative = interpolate.InterpolatedUnivariateSpline(np.arange(len(array)), array,
                                                                     k=3).derivative(2)

        # Find first minimum of second_derivative
        x_min = 0
        sec_deriv_previous = second_deriv_averaged(second_derivative, 0, length)
        for x_iter in x_range:
            sec_deriv_current = second_deriv_averaged(second_derivative, x_iter, length)
            if sec_deriv_previous >= sec_deriv_current:
                sec_deriv_previous = sec_deriv_current
            else:
                x_min = x_iter - 1
                break
        x_min -= x_min_shift

        # Find first maximum of second_derivative
        x_max_temp = x_min + x_min_shift
        sec_deriv_previous = second_deriv_averaged(second_derivative, x_max_temp, length)
        for x_iter in np.arange(x_max_temp, length, 1):
            sec_deriv_current = second_deriv_averaged(second_derivative, x_iter, length)
            if sec_deriv_previous <= sec_deriv_current:
                sec_deriv_previous = sec_deriv_current
            else:
                x_max_temp = x_iter - 1
                break

        # Find second minimum of second_derivative
        x_min_temp = x_max_temp
        sec_deriv_previous = second_deriv_averaged(second_derivative, x_min_temp, length)
        for x_iter in np.arange(x_min_temp, length, 1):
            sec_deriv_current = second_deriv_averaged(second_derivative, x_iter, length)
            if sec_deriv_previous >= sec_deriv_current:
                sec_deriv_previous = sec_deriv_current
            else:
                x_min_temp = x_iter - 1
                break

        # Find first maximum of second_derivative
        x_max_temp = x_min_temp
        sec_deriv_previous = second_deriv_averaged(second_derivative, x_max_temp, length)
        for x_iter in np.arange(x_max_temp, length, 1):
            sec_deriv_current = second_deriv_averaged(second_derivative, x_iter, length)
            if sec_deriv_previous <= sec_deriv_current:
                sec_deriv_previous = sec_deriv_current
            else:
                x_max_temp = x_iter - 1
                break
        x_max = x_max_temp + x_max_shift

        # Cut baryonic bump
        cut_nums = np.arange(x_min, x_max+1, 1)
        x_range_new = np.delete(x_range, cut_nums)
        array_new = np.delete(array, cut_nums)

        # Interpolate
        interpolated = interpolate.InterpolatedUnivariateSpline(x_range_new,
                                                                (x_range_new+1)**2 * array_new, k=3)
        return interpolated(x_range)/(x_range+1)**2, x_min, x_max


    def inverse_fast_sine_transform(even_array, odd_array):
        """Perform inverse sine transform after reassembling the even and odd arrays"""
        new_array = [None] * (len(even_array)+len(odd_array))
        new_array[0::2] = even_array
        new_array[1::2] = odd_array
        return fftpack.idst(new_array, type=2, norm='ortho')


    def smooth_pk(karray, lnkpk_array, krange, logplogk, n_samps, kmax_extr):
        """Compute smooth P(k) by using P(k) for k < 3 times 2^(-n_samps) and
        k > k(smallest peak height of P^w/P^{nw}) (before peaks grow again due to numerics/...)
        and P^w(k) in between"""
        # Compute and interpolate P^{nw}
        smooth_pk_array = np.exp(lnkpk_array)/karray
        smooth_interpolated = interpolate.InterpolatedUnivariateSpline(np.log(karray),
                                                                       np.log(smooth_pk_array), k=3)

        # Set first split in k
        k_split1 = 2**(-n_samps)*3

        # Compute second split in k:
        # Find peaks and troughs (i.e. maxima and approximate zeros) of |(P(k)-P^{nw}(k))/P^{nw}(k)|
        prev_value = -np.inf
        bool_list = []
        k_temp = krange[krange >= 0.01]
        k_values = k_temp[k_temp <= kmax_extr]
        exp_smooth = np.exp(smooth_interpolated(np.log(k_values)))
        y_values = np.abs(pk_func(logplogk, k_values) / exp_smooth - 1.)
        for k in k_values:
            value = np.abs(pk_func(logplogk, k) / np.exp(smooth_interpolated(np.log(k))) - 1.)
            bool_list.append(value > prev_value)
            prev_value = value

        # Distinguish peaks and troughs
        n_peaks = []
        n_troughs = []
        for n_all in np.arange(len(bool_list)-1):
            if bool_list[n_all] and not bool_list[n_all+1]:
                n_peaks.append(n_all)
            if not bool_list[n_all] and bool_list[n_all+1]:
                n_troughs.append(n_all)
        p_maxima = y_values[n_peaks].tolist()
        n_min_peak = p_maxima.index(min(p_maxima))

        # Set kSplit2 as the position of the trough/'zero' after the smallest peak to cut off
        # oscillations with increasing amplitude, i.e. numerical artifacts
        for n_count in np.arange(len(n_troughs)):
            if k_values[n_troughs[n_count]] > k_values[n_peaks[n_min_peak]]:
                n_min_trough = n_count
                break
        k_split2 = k_values[n_troughs[n_min_trough]]

        # Split up k-range
        split_num1 = len(krange[krange <= k_split1])
        split_num2 = len(krange[krange <= k_split2])
        k_ranges = np.split(krange, [split_num1, split_num2])

        # Set P^{nw} in three k-ranges
        smooth_pk1 = pk_func(logplogk, k_ranges[0])
        smooth_pk2 = np.exp(smooth_interpolated(np.log(k_ranges[1])))
        smooth_pk3 = pk_func(logplogk, k_ranges[2])
        smooth_pk_combined = np.append(np.append(smooth_pk1, smooth_pk2), smooth_pk3)
        return smooth_pk_combined, (k_split1, k_split2)

    # Now do the work:
    # Compute necessary arrays and transform
    logp_logk = interpolate.InterpolatedUnivariateSpline(np.log(k_range), np.log(pk_array), k=3)
    k_array, lnk_pk_array = sample_lnk_pk(logp_logk, k_range[0], k_max_extr, points)
    array_even, array_odd = fast_sine_transform(lnk_pk_array)

    # Cut bump and interpolate
    array_even_new = cut_out_and_interpolate(array_even, 'even')[0]
    array_odd_new = cut_out_and_interpolate(array_odd, 'odd')[0]

    # Transform back, and get P^{nw} and P^w
    lnk_lnp_array_new = inverse_fast_sine_transform(array_even_new, array_odd_new)
    pknw_array, k_splits = smooth_pk(k_array, lnk_lnp_array_new, k_range, logp_logk, points,
                                     k_max_extr)
    pkw_array = pk_array - pknw_array

    return k_range, pk_array, pknw_array, pkw_array, k_splits
