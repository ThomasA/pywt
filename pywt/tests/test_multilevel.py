#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (run_module_suite, assert_almost_equal,
                           assert_allclose, assert_)

import pywt


def test_wavedec():
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    db1 = pywt.Wavelet('db1')
    cA3, cD3, cD2, cD1 = pywt.wavedec(x, db1)
    assert_almost_equal(cA3, [8.83883476])
    assert_almost_equal(cD3, [-0.35355339])
    assert_allclose(cD2, [4., -3.5])
    assert_allclose(cD1, [-2.82842712, 0, -4.94974747, -1.41421356])
    assert_(pywt.dwt_max_level(len(x), db1) == 3)


def test_waverec():
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    coeffs = pywt.wavedec(x, 'db1')
    assert_allclose(pywt.waverec(coeffs, 'db1'), x, rtol=1e-12)


def test_swt_decomposition():
    x = [3, 7, 1, 3, -2, 6, 4, 6]
    db1 = pywt.Wavelet('db1')
    (cA2, cD2), (cA1, cD1) = pywt.swt(x, db1, level=2)
    expected_cA1 = [7.07106781, 5.65685425, 2.82842712, 0.70710678,
                    2.82842712, 7.07106781, 7.07106781, 6.36396103]
    assert_allclose(cA1, expected_cA1)
    expected_cD1 = [-2.82842712, 4.24264069, -1.41421356, 3.53553391,
                    -5.65685425, 1.41421356, -1.41421356, 2.12132034]
    assert_allclose(cD1, expected_cD1)
    expected_cA2 = [7, 4.5, 4, 5.5, 7, 9.5, 10, 8.5]
    assert_allclose(cA2, expected_cA2, rtol=1e-12)
    expected_cD2 = [3, 3.5, 0, -4.5, -3, 0.5, 0, 0.5]
    assert_allclose(cD2, expected_cD2, rtol=1e-12, atol=1e-14)

    # level=1, start_level=1 decomposition should match level=2
    res = pywt.swt(cA1, db1, level=1, start_level=1)
    cA2, cD2 = res[0]
    assert_allclose(cA2, expected_cA2, rtol=1e-12)
    assert_allclose(cD2, expected_cD2, rtol=1e-12, atol=1e-14)

    coeffs = pywt.swt(x, db1)
    assert_(len(coeffs) == 3)
    assert_(pywt.swt_max_level(len(x)) == 3)


def test_swt_iswt_integration():
    """
    This function performs a round-trip swt/iswt transform test on
    all available types of wavelets in PyWavelets - except the 'dmey'
    wavelet. The latter has been excluded because it does not produce
    very precise results. This is likely due to the fact that the
    'dmey' wavelet is a discrete approximation of a continuous
    wavelet. All wavelets are tested up to 3 levels. The test
    validates neither swt or iswt as such, but it does ensure
    that they are each other's inverse.
    """
    max_level = 3
    wavelets = pywt.wavelist()
    if 'dmey' in wavelets:
        wavelets.remove('dmey') # The 'dmey' wavelet seems to be a bit special - disregard it for now
    for current_wavelet_str in wavelets:
        current_wavelet = pywt.Wavelet(current_wavelet_str)
        input_length_power = np.ceil(np.log2(max(current_wavelet.dec_len, current_wavelet.rec_len)))
        input_length = 2**int(input_length_power + max_level - 1)
        X = np.arange(input_length)
        coeffs = pywt.swt(X, current_wavelet, max_level)
        Y = pywt.iswt(coeffs, current_wavelet)
        assert_allclose(Y, X, rtol=1e-5, atol=1e-7)


def test_swt2_iswt2_integration():
    """
    This function performs a round-trip swt2/iswt2 transform test on
    all available types of wavelets in PyWavelets - except the 'dmey'
    wavelet. The latter has been excluded because it does not produce
    very precise results. This is likely due to the fact that the
    'dmey' wavelet is a discrete approximation of a continuous
    wavelet. All wavelets are tested up to 3 levels. The test
    validates neither swt2 or iswt2 as such, but it does ensure
    that they are each other's inverse.
    """
    max_level = 3
    wavelets = pywt.wavelist()
    if 'dmey' in wavelets:
        wavelets.remove('dmey') # The 'dmey' wavelet seems to be a bit special - disregard it for now
    for current_wavelet_str in wavelets:
        current_wavelet = pywt.Wavelet(current_wavelet_str)
        input_length_power = np.ceil(np.log2(max(current_wavelet.dec_len, current_wavelet.rec_len)))
        input_length = 2**int(input_length_power + max_level - 1)
        X = np.arange(input_length**2).reshape(input_length, input_length)
        coeffs = pywt.swt2(X, current_wavelet, max_level)
        Y = pywt.iswt2(coeffs, current_wavelet)
        assert_allclose(Y, X, rtol=1e-5, atol=1e-5)


def test_wavedec2():
    coeffs = pywt.wavedec2(np.ones((4, 4)), 'db1')
    assert_(len(coeffs) == 3)
    assert_allclose(pywt.waverec2(coeffs, 'db1'), np.ones((4, 4)), rtol=1e-12)


if __name__ == '__main__':
    run_module_suite()
