import time, math

import numpy as np
from tqdm import tqdm
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

# 4000x faster on input with 10_000 inputs
def binary_to_decimal(y_binary):
    fractional_binary = "0." + y_binary # indicates this binary number is <1
    return gmpy2.mpfr(fractional_binary, base=2)

# gmpy2 1.15x faster on 50_000 inputs
def logistic_decoder(alpha, sample_idx, precision):
   return float(gmpy2.sin(gmpy2.mpfr(2) ** (sample_idx * precision) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2)

def logistic_decoder_list(alpha, sample_idxs, precision):
    exponents = (sample_idxs*precision).tolist()
    mod = gmpy2.mpz(gmpy2.mpfr(2) ** (max(exponents) + 1)) # make mod so big we don't use it
    samples = gmpy2.powmod_exp_list(2, exponents, mod)
    const = gmpy2.asin(gmpy2.sqrt(alpha))
    y_pred = np.array([float(gmpy2.sin(sample * const) **2) for sample in tqdm(samples)])
    ic(y_pred.shape)
    return y_pred

#***** model *****

# scalar reasoning model
class SRM:
    def __init__(self, precision, verbose=True):
        self.precision = precision # binary precision, not decimal precision
        self.verbose = verbose
        self.scaler = None
        self.alpha = None

    def _get_alpha(self, y_decimal):
        # compute φ^(-1)(x) and convert to binary
        phi_inverses_decimal = phi_inverse(y_decimal)
        phi_inverses_binary = decimal_to_binary(phi_inverses_decimal, self.precision)

        # set precision for arbitrary floating-point math
        total_precision = len(y_decimal) * self.precision
        gmpy2.get_context().precision = total_precision
        if len(phi_inverses_binary) != total_precision:
            raise ValueError(f"Expected {total_precision} binary digits for y but got {len(phi_inverses_binary)}.")

        # convert to decimal with arbitrary number of floating point precision
        phi_inverse_decimal = binary_to_decimal(phi_inverses_binary)
        phi_decimal = phi(phi_inverse_decimal)
        return phi_decimal

    def fit(self, X, y):
        self.scaler = MinMaxScaler()
        y_scaled = self.scaler.scale(y)
        self.alpha = self._get_alpha(y_scaled)
        return self

    def transform(self, X_idxs):
        y_pred_unscaled = logistic_decoder_list(self.alpha, X_idxs, self.precision)
        y_pred = self.scaler.unscale(y_pred_unscaled)
        return y_pred

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

#***** main *****

def main():
    precision = 15
    # X, y = np.arange(3_000), np.arange(3_000)
    X, y = get_scatter_data()
    X_idxs = np.arange(len(X))
    ic(X.shape, y.shape)
    # plot_data(X, y)

    srm = SRM(precision)
    srm.fit(X, y)
    y_pred = srm.transform(X_idxs)
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

