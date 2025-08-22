import time, math

import numpy as np
from tqdm import tqdm
import mpmath
from mpmath import mp
from mpmath import pi as Pi
from mpmath import sin as Sin
from mpmath import asin as Asin
from mpmath import sqrt as Sqrt
import gmpy2

from icecream import ic

from data import get_scatter_data, plot_data

#***** utils *****

class MinMaxScaler:
    def __init__(self): self.min = self.max = None
    def scale(self, X):
        self.min, self.max = X.min(axis=0), X.max(axis=0)
        return (X - self.min) / (self.max - self.min)
    def unscale(self, X): return X * (self.max - self.min) + self.min

# class MinMaxScaler:
#     def __init__(self): self.min = self.max = None
#     def fit(self, X):
#         self.min, self.max = X.min(axis=0), X.max(axis=0)
#         return self
#     def transform(self, X): return (X - self.min) / (self.max - self.min)
#     def fit_transform(self, X): return self.fit(X).transform(X)
#     def inverse_transform(self, X): return X * (self.max - self.min) + self.min

#***** math *****

# on 5_000_000 elements, gmpy2 is 2x faster than mpmath
def phi(theta): return gmpy2.sin(theta * gmpy2.const_pi() * 2) ** 2
def phi_inverse(z): return np.arcsin(np.sqrt(z)) / (2.0 * np.pi)

# https://sandbox.mc.edu/%7Ebennet/cs110/flt/dtof.html
# 13x speedup on input with 10M elements
# alternatively, you can use gmpy2.digits(y_decimal, 2) but this
# cannot be vectorized and is likely slower (have not tested yet)
def decimal_to_binary(y_decimal, precision):
    powers = 2**np.arange(precision)
    y_powers = y_decimal[:, np.newaxis] * powers[np.newaxis, :]
    y_fractional = y_powers % 1 # extract the fractional part of y_powers
    binary_digits = (y_fractional >= 0.5).astype(int)
    binary_str = ''.join(map(str, binary_digits.flatten().tolist()))
    return binary_str

# 4000x faster on input with 10_000 elements
def binary_to_decimal(y_binary):
    fractional_binary = "0." + y_binary # indicates this binary number is <1
    return gmpy2.mpfr(fractional_binary, base=2)

def logistic_decoder(alpha, sample_idx, precision):
    return float(Sin(2 ** (sample_idx * precision) * Asin(Sqrt(alpha))) ** 2)

#***** model *****

# scalar reasoning model
class SRM:
    def __init__(self, precision, verbose=True):
        self.precision = precision # binary precision, not decimal precision
        self.verbose = verbose

        self.scaler = MinMaxScaler()
        self.alpha = None
        self.total_precision = None

        assert mpmath.libmp.BACKEND == 'gmpy', 'mpmath is painfully slow without gmpy backend. Run `pip install gmpy2` to install gmpy.'

    def _get_alpha(self, y_decimal):
        # compute φ^(-1)(x) and convert to binary
        phi_inverses_decimal = phi_inverse(y_decimal)
        phi_inverses_binary = decimal_to_binary(phi_inverses_decimal, self.precision)

        # set precision for arbitrary floating-point math
        self.total_precision = len(y_decimal) * self.precision # binary precision, not decimal precision
        if len(phi_inverses_binary) != self.total_precision:
            raise ValueError(f"expected the total number of binary digits for the training labels binary to be {self.total_precision} but got {len(phi_inverses_binary)}.")
        mp.prec = self.total_precision # precision in bits
        gmpy2.get_context().precision = self.total_precision
        if self.verbose:
            print(f'To represent φ^(-1)(y) for all {len(y_decimal)} training labels with binary precision={self.precision}, requires {self.total_precision} binary digits or {mp.dps} decimal digits.')

        # convert to decimal with arbitrary number of floating point precision
        phi_inverse_decimal = binary_to_decimal(phi_inverses_binary)
        phi_decimal = phi(phi_inverse_decimal)
        return phi_decimal

    def fit(self, X, y):
        y_scaled = self.scaler.scale(y)
        self.alpha = self._get_alpha(y_scaled)
        return self

    def transform(self, X):
        # if use np.int64 for sample_idx values, then we get an overflow in 2 ** (sample_idx * precision) of logistic_decoder
        y_pred_unscaled = np.array([logistic_decoder(self.alpha, sample_idx.item(), self.precision) for sample_idx in tqdm(X)])
        y_pred = self.scaler.unscale(y_pred_unscaled)
        return y_pred

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

#***** main *****

def main():
    precision = 50
    X, y = np.arange(100_000), np.arange(100_000)
    # X, y = get_scatter_data()
    # plot_data(X, y)

    srm = SRM(precision)
    y_pred = srm.fit_transform(X, y)
    # ic(y, y_pred)
    plot_data(X, y, y_pred)


if __name__ == '__main__':
    main()

# currently, we learn f(idx) -> y
# need x -> idx
# to get x->index, just use an overiffted polynomial!
# do polynomial interpolation on (x[idx], [idx]) and then map this polynomial over!
# but then we need to represent this polynomial as one scalar too!

# instead of learning f(idx) -> y
# let's learn an overfitted polynomial to our data g(x) -> y which has params theta = [theta_1, ..., theta_n]
# now let's encode theta into a single number using f(theta) = alpha
# now when we get a number, we do f(x)
# f(i) = theta_i
# sum_{i=1} theta_i x_i = y_i

# so we have \sum_{i=1}^d f(i) x_i = y_i

# todo:
# 1. can we use mpmath or gmpy2 for decimal_to_binary: gmpy2.digits()?
# 2. vectorize all functions
# 3. vectorize
