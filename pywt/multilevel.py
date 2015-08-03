# -*- coding: utf-8 -*-

# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# See COPYING for license details.

"""
Multilevel 1D and 2D Discrete Wavelet Transform
and Inverse Discrete Wavelet Transform.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from ._pywt import Wavelet
from ._pywt import dwt, idwt, dwt_max_level
from .multidim import dwt2, idwt2

__all__ = ['wavedec', 'waverec', 'wavedec2', 'waverec2', 'iswt', 'iswt2']


def wavedec(data, wavelet, mode='sym', level=None):
    """
    Multilevel 1D Discrete Wavelet Transform of data.

    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string
        Wavelet to use
    mode : str, optional
        Signal extension mode, see MODES (default: 'sym')
    level : int, optional
        Decomposition level. If level is None (default) then it will be
        calculated using `dwt_max_level` function.

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.

    Examples
    --------
    >>> from pywt import multilevel
    >>> coeffs = multilevel.wavedec([1,2,3,4,5,6,7,8], 'db1', level=2)
    >>> cA2, cD2, cD1 = coeffs
    >>> cD1
    array([-0.70710678, -0.70710678, -0.70710678, -0.70710678])
    >>> cD2
    array([-2., -2.])
    >>> cA2
    array([  5.,  13.])

    """

    if not isinstance(wavelet, Wavelet):
        wavelet = Wavelet(wavelet)

    if level is None:
        level = dwt_max_level(len(data), wavelet.dec_len)
    elif level < 0:
        raise ValueError(
            "Level value of %d is too low . Minimum level is 0." % level)

    coeffs_list = []

    a = data
    for i in range(level):
        a, d = dwt(a, wavelet, mode)
        coeffs_list.append(d)

    coeffs_list.append(a)
    coeffs_list.reverse()

    return coeffs_list


def waverec(coeffs, wavelet, mode='sym'):
    """
    Multilevel 1D Inverse Discrete Wavelet Transform.

    Parameters
    ----------
    coeffs : array_like
        Coefficients list [cAn, cDn, cDn-1, ..., cD2, cD1]
    wavelet : Wavelet object or name string
        Wavelet to use
    mode : str, optional
        Signal extension mode, see MODES (default: 'sym')

    Examples
    --------
    >>> from pywt import multilevel
    >>> coeffs = multilevel.wavedec([1,2,3,4,5,6,7,8], 'db2', level=2)
    >>> multilevel.waverec(coeffs, 'db2')
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    """

    if not isinstance(coeffs, (list, tuple)):
        raise ValueError("Expected sequence of coefficient arrays.")

    if len(coeffs) < 2:
        raise ValueError(
            "Coefficient list too short (minimum 2 arrays required).")

    a, ds = coeffs[0], coeffs[1:]

    for d in ds:
        a = idwt(a, d, wavelet, mode, 1)

    return a


def wavedec2(data, wavelet, mode='sym', level=None):
    """
    Multilevel 2D Discrete Wavelet Transform.

    Parameters
    ----------
    data : ndarray
        2D input data
    wavelet : Wavelet object or name string
        Wavelet to use
    mode : str, optional
        Signal extension mode, see MODES (default: 'sym')
    level : int, optional
        Decomposition level. If level is None (default) then it will be
        calculated using `dwt_max_level` function.

    Returns
    -------
    [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)] : list
        Coefficients list

    Examples
    --------
    >>> from pywt import multilevel
    >>> coeffs = multilevel.wavedec2(np.ones((4,4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> multilevel.waverec2(coeffs, 'db1')
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])
    """

    data = np.asarray(data, np.float64)

    if data.ndim != 2:
        raise ValueError("Expected 2D input data.")

    if not isinstance(wavelet, Wavelet):
        wavelet = Wavelet(wavelet)

    if level is None:
        size = min(data.shape)
        level = dwt_max_level(size, wavelet.dec_len)
    elif level < 0:
        raise ValueError(
            "Level value of %d is too low . Minimum level is 0." % level)

    coeffs_list = []

    a = data
    for i in range(level):
        a, ds = dwt2(a, wavelet, mode)
        coeffs_list.append(ds)

    coeffs_list.append(a)
    coeffs_list.reverse()

    return coeffs_list


def waverec2(coeffs, wavelet, mode='sym'):
    """
    Multilevel 2D Inverse Discrete Wavelet Transform.

    coeffs : list
        Coefficients list [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)]
    wavelet : Wavelet object or name string
        Wavelet to use
    mode : str, optional
        Signal extension mode, see MODES (default: 'sym')

    Returns
    -------
    2D array of reconstructed data.

    Examples
    --------
    >>> from pywt import multilevel
    >>> coeffs = multilevel.wavedec2(np.ones((4,4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> multilevel.waverec2(coeffs, 'db1')
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])
    """

    if not isinstance(coeffs, (list, tuple)):
        raise ValueError("Expected sequence of coefficient arrays.")

    if len(coeffs) < 2:
        raise ValueError(
            "Coefficient list too short (minimum 2 arrays required).")

    a, ds = coeffs[0], coeffs[1:]

    for d in ds:
        a = idwt2((a, d), wavelet, mode)

    return a


def iswt(coeffs, wavelet):
    """
    Multilevel 1D Inverse Discrete Stationary Wavelet Transform.

    Parameters
    ----------
    coeffs : array_like
        Coefficients list of tuples::

            [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

        where cA is approximation, cD is details, and n is start_level.
    wavelet : Wavelet object or name string
        Wavelet to use

    Examples
    --------
    >>> import pywt
    >>> coeffs = pywt.swt([1,2,3,4,5,6,7,8], 'db2', level=2)
    >>> pywt.iswt(coeffs, 'db2')
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    """

    output = coeffs[0][0].copy()  # Avoid modification of input data

    # num_levels, equivalent to the decomposition level, n
    num_levels = len(coeffs)
    for j in range(num_levels,0,-1):
        step_size = int(pow(2, j-1))
        last_index = step_size
        _, cD = coeffs[num_levels - j]
        for first in range(last_index):  # 0 to last_index - 1

            # Getting the indices that we will transform
            indices = np.arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2]
            # select the odd indices
            odd_indices = indices[1::2]

            # perform the inverse dwt on the selected indices,
            # making sure to use periodic boundary conditions
            x1 = idwt(output[even_indices], cD[even_indices], wavelet, 'per')
            x2 = idwt(output[odd_indices], cD[odd_indices], wavelet, 'per')

            # perform a circular shift right
            x2 = np.roll(x2, 1)

            # average and insert into the correct indices
            output[indices] = (x1 + x2)/2.

    return output


def iswt2(coeffs, wavelet):
    """
    Multilevel 2D Inverse Discrete Stationary Wavelet Transform.

    Parameters
    ----------
    coeffs : list
        Approximation and details coefficients::

            [
                (cA_1,
                    (cH_1, cV_1, cD_1)
                ),
                (cA_2,
                    (cH_2, cV_2, cD_2)
                ),
                ...,
                (cA_n
                    (cH_n, cV_n, cD_n)
                )
            ]

        where cA is approximation, cH is horizontal details, cV is
        vertical details, cD is diagonal details and n is number of
        levels.
    wavelet : Wavelet object or name string
        Wavelet to use

    Examples
    --------
    >>> import pywt
    >>> coeffs = coeffs = pywt.swt2([[1,2,3,4],[5,6,7,8],
                                     [9,10,11,12],[13,14,15,16]],
                                    'db1', level=2)
    >>> pywt.iswt2(coeffs, 'db1')
    array([[  1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.],
           [  9.,  10.,  11.,  12.],
           [ 13.,  14.,  15.,  16.]])

    """

    output = coeffs[-1][0].copy()  # Avoid modification of input data

    # num_levels, equivalent to the decomposition level, n
    num_levels = len(coeffs)
    for j in range(num_levels,0,-1):
        step_size = int(pow(2, j-1))
        last_index = step_size
        _, (cH, cV, cD) = coeffs[j-1]
        # We are going to assume cH, cV, and cD are square and of equal size
        assert(cH.shape == cV.shape)
        assert(cH.shape == cD.shape)
        assert(cH.shape[0] == cH.shape[0])
        for first_h in range(last_index):  # 0 to last_index - 1
            for first_w in range(last_index):  # 0 to last_index - 1
                # Getting the indices that we will transform
                indices_h = slice(first_h, cH.shape[0], step_size)
                indices_w = slice(first_w, cH.shape[1], step_size)

                even_idx_h = slice(first_h, cH.shape[0], 2*step_size)
                even_idx_w = slice(first_w, cH.shape[1], 2*step_size)
                odd_idx_h = slice(first_h + step_size, cH.shape[0], 2*step_size)
                odd_idx_w = slice(first_w + step_size, cH.shape[1], 2*step_size)

                # perform the inverse dwt on the selected indices,
                # making sure to use periodic boundary conditions
                x1 = idwt2((output[even_idx_h, even_idx_w],
                                 (cH[even_idx_h, even_idx_w],
                                  cV[even_idx_h, even_idx_w],
                                  cD[even_idx_h, even_idx_w])),
                                wavelet, 'per')
                x2 = idwt2((output[even_idx_h, odd_idx_w],
                                 (cH[even_idx_h, odd_idx_w],
                                  cV[even_idx_h, odd_idx_w],
                                  cD[even_idx_h, odd_idx_w])),
                                wavelet, 'per')
                x3 = idwt2((output[odd_idx_h, even_idx_w],
                                 (cH[odd_idx_h, even_idx_w],
                                  cV[odd_idx_h, even_idx_w],
                                  cD[odd_idx_h, even_idx_w])),
                                wavelet, 'per')
                x4 = idwt2((output[odd_idx_h, odd_idx_w],
                                 (cH[odd_idx_h, odd_idx_w],
                                  cV[odd_idx_h, odd_idx_w],
                                  cD[odd_idx_h, odd_idx_w])),
                                wavelet, 'per')

                # perform a circular shifts
                x2 = np.roll(x2, 1, axis=1)
                x3 = np.roll(x3, 1, axis=0)
                x4 = np.roll(x4, 1, axis=0)
                x4 = np.roll(x4, 1, axis=1)
                output[indices_h, indices_w] = (x1 + x2 + x3 + x4) / 4

    return output
