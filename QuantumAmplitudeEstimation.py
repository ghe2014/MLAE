# Copyright (c) 2023 Guangliang He
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from fractions import Fraction
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt


def log_likelihood_func(theta: float | npt.NDArray[float],
                        m: int | List[int],
                        s: int | List[int],
                        h: int | List[int]) -> float | npt.NDArray[float]:
    """
    log likelihood function
    theta: either float or np.ndarray of float
    m: either int or list of int, the evaluation schedule
    s: either int or list of int, matching m, the shots
    h: either int or list of int, matching m, the hits
    return:
    logL(theta) as a ndarray(float), same shape as theta
    """

    if isinstance(m, list):
        # m, s, h are lists
        return sum([log_likelihood_func(theta, m[k], s[k], h[k])
                    for k in range(len(m))])

    # m, s, h are scalars
    theta_m = (2*m+1)*theta
    sin_theta_m_2 = np.sin(theta_m)**2
    cos_theta_m_2 = np.cos(theta_m)**2
    return h*np.log(sin_theta_m_2) + (s-h)*np.log(cos_theta_m_2)


def log_likelihood_prime(theta: float | npt.NDArray[float],
                         m: int | List[int],
                         s: int | List[int],
                         h: int | List[int]) -> float | npt.NDArray[float]:
    """
    first derivative of log likelihood function
    theta: either float or np.ndarray of float
    m: either int or list of int, the evaluation schedule
    s: either int or list of int, matching m, the shots
    h: either int or list of int, matching m, the hits
    return:
    d logL(theta)/d theta as a ndarray(float), same shape as theta
    """

    if isinstance(m, list):
        # m, N, h are lists
        return sum([log_likelihood_prime(theta, m[k], s[k], h[k])
                    for k in range(len(m))])

    # m, s, h are scalars
    theta_m = (2*m+1)*theta
    tan_theta_m = np.tan(theta_m)
    cot_theta_m = 1/tan_theta_m
    return 2*(2*m+1)*(h*cot_theta_m-(s-h)*tan_theta_m)


def maximize_likelihood(m: List[int],
                        s: List[int],
                        h: List[int]) -> Tuple[float, float, int]:
    """
    find the global maximum of log L by search local maxima in
    regions separated by singularities.  
    m: list of int, the evaluation schedule
    s: list of int, matching m, the shots
    h: list of int, matching m, the hits
    return:
    tuple of (theta_hat, logL(theta_hat), n_calls)
    """

    # the set of singularities (without the scale of pi/2)
    # in fractions
    singularity_set = set()
    for k in range(len(m)):
        if h[k] > 0:
            singularity_set.update([Fraction(2*i, 2*m[k]+1)
                                    for i in range(m[k]+1)])
        if h[k] < s[k]:
            singularity_set.update([Fraction(2*i+1, 2*m[k]+1)
                                    for i in range(m[k]+1)])

    if Fraction(0) not in singularity_set:
        # 0 is not a singularity, then 0 must be the maximum
        return 0.0, 0.0, 0

    if Fraction(1) not in singularity_set:
        # pi/2 is not a singularity, pi/2 must be the maximum
        return np.pi/2, 0.0, 0

    # sort the singularities, also multiply by pi/2
    sorted_singularities = sorted([f*np.pi/2
                                   for f in singularity_set])

    eps = 1e-15
    theta_hat = None
    max_log_l = -float('INF')
    n_calls = 0

    for i in range(len(sorted_singularities)-1):
        bracket = [sorted_singularities[i]+eps,
                   sorted_singularities[i+1]-eps]
        rf_result = opt.root_scalar(log_likelihood_prime,
                                    args=(m, s, h),
                                    method="brentq",
                                    bracket=bracket)
        n_calls += rf_result.function_calls
        if not rf_result.converged:
            message = (
                f"maximize_likelihood({m}, {s}, {h}) "
                f"did not converge\n"
                f"bracket = {bracket}\n"
                + str(rf_result))
            warnings.warn(message)

        log_likelihood = log_likelihood_func(rf_result.root,
                                             m, s, h)
        if log_likelihood > max_log_l:
            theta_hat = rf_result.root
            max_log_l = log_likelihood

    return theta_hat, max_log_l, n_calls


def iterative_maximize_likelihood(m: List[int],
                                  s: List[int],
                                  h: List[int],
                                  n: int = 1) -> Tuple[float, float, int]:
    """
    m: list of int, the evaluation schedule
    s: list of int, matching m, the shots
    h: int or list of int, matching m, the hits
    n: the number of additional search region on one side of center
       search region.  Total search region is 2n+1
    return:
    tuple of (theta_hat, logL(theta_hat), n_calls)
    """

    eps = 1e-15
    m_len = len(m)
    
    theta_hat = None
    max_log_l = -float('INF')
    n_calls = 0
    singularity_set = set()
    for k in range(m_len):
        if h[k] > 0:
            singularity_set.update([Fraction(2*i, 2*m[k]+1)
                                    for i in range(m[k]+1)])
        if h[k] < s[k]:
            singularity_set.update([Fraction(2*i+1, 2*m[k]+1)
                                    for i in range(m[k]+1)])

        if Fraction(0) not in singularity_set:
            theta_hat = 0
            max_log_l = 0
            continue

        if Fraction(1) not in singularity_set:
            theta_hat = np.pi/2
            max_log_l = 0
            continue

        # determine search region
        sorted_singularities = sorted([frac*np.pi/2
                                       for frac in singularity_set])

        r_start = None
        r_end = None
        if theta_hat is None:
            r_start = 0
            r_end = len(sorted_singularities)-1
        else:
            # find the region of theta_hat
            for i in range(len(sorted_singularities)):
                if theta_hat < sorted_singularities[i]:
                    # theta_hat falls in region i-1,
                    # between singularity i-1 and i
                    r_start = max(0, i-1-n)
                    r_end = min(i+n, len(sorted_singularities)-1)
                    break

        if r_start is None:
            message = (f"r_start not set. theta_hat = {theta_hat}"
                       f"sorted_singularities = {sorted_singularities}")
            raise ValueError(message)

        max_log_l = -float('INF')
        for i in range(r_start, r_end):
            bracket = [sorted_singularities[i]+eps,
                       sorted_singularities[i+1]-eps]
            rf_result = opt.root_scalar(log_likelihood_prime,
                                        args=(m[:k+1], s[:k+1], h[:k+1]),
                                        method='brentq',
                                        bracket=bracket)
            if not rf_result.converged:
                message = (
                    f"maximize_likelihood({m[:k+1]}, {s[:k+1]}, {h[:k+1]}) "
                    f"did not converge\nbracket = {bracket}\n"
                    + str(rf_result))
                warnings.warn(message)

            n_calls += rf_result.function_calls
            log_l = log_likelihood_func(rf_result.root,
                                        m[:k+1], s[:k+1], h[:k+1])
            if log_l > max_log_l:
                theta_hat = rf_result.root
                max_log_l = log_l

        if theta_hat is None:
            message = ("iterative_maximize_likelihood_failed"
                       f"m = {m}\ns = {s}\nh = {h}\n")
            raise ValueError(message)

    return theta_hat, max_log_l, n_calls
