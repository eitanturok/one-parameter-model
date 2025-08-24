# Run with  `uv run src/arc-agi2/srm.py`
from functools import partial
import numpy as np
from multiprocessing import Pool
import gmpy2

from icecream import ic

from data import get_scatter_data, get_arc_agi2, plot_data
from utils import getenv, MinMaxScaler, Precision, Timing, tqdm

VERBOSE = getenv("VERBOSE", 1)
WORKERS = getenv("WORKERS", 8)

#***** math *****

def phi(theta): return gmpy2.sin(theta * gmpy2.const_pi() * 2) ** 2
def phi_inverse(z): return np.arcsin(np.sqrt(z)) / (2.0 * np.pi)

# convert decimal floats in [0, 1] to binary via https://sandbox.mc.edu/%7Ebennet/cs110/flt/dtof.html]
# cannot use python's bin() function because it only converts int to binary
def decimal_to_binary(y_decimal, precision):
    powers = 2**np.arange(precision)
    y_powers = y_decimal[:, np.newaxis] * powers[np.newaxis, :]
    y_fractional = y_powers % 1 # extract the fractional part of y_powers
    binary_digits = (y_fractional >= 0.5).astype(int).astype('<U1')
    return np.apply_along_axis(''.join, axis=1, arr=binary_digits).tolist()

def binary_to_decimal(y_binary):
    fractional_binary = "0." + y_binary # indicate the binary number is a float in [0, 1], not an int
    return gmpy2.mpfr(fractional_binary, base=2)

def logistic_decoder_simple(alpha, sample_idx, precision):
   return float(gmpy2.sin(gmpy2.mpfr(2) ** (sample_idx * precision) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2)

def logistic_decoder(alpha, sample_idxs, precision):
    exponents = (sample_idxs * precision).astype(int).tolist()
    mod = gmpy2.mpz(gmpy2.mpfr(2) ** (max(exponents) + 1)) # make mod so big we don't use it
    samples = gmpy2.powmod_exp_list(2, exponents, mod)
    const = gmpy2.asin(gmpy2.sqrt(alpha))
    y_pred = np.array([float(gmpy2.sin(sample * const) **2) for sample in tqdm(samples, desc="Decoding")])
    return y_pred

# @Precision(17288)
def logistic_decoder_single(total_prec, alpha, prec, idx):
  gmpy2.get_context().precision = total_prec # set the precision in each pool/thread
  return float(gmpy2.sin(gmpy2.mpfr(2) ** (idx * prec) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2)

# @Precision(17288)
def logistic_decoder_parallel(total_prec, alpha, prec, idxs):
  fxn = partial(logistic_decoder_single, total_prec, alpha, prec)
  with Pool(WORKERS) as p:
      return np.array(list(tqdm(p.imap(fxn, idxs), total=len(idxs), desc="Logistic Decoder")))

#***** model *****

# scalar reasoning model
# EpsilonNet - cause it's tiny
# NanoGPT - already done
# GPT-0.1 - cause tiny
class SRM:
    def __init__(self, precision):
        self.precision = precision # binary precision, not decimal precision, for a single number

    @Timing("fit: ", enabled=VERBOSE)
    def fit(self, X, y):

        # compute alpha with arbitrary floating-point precision
        self.y_shape, self.total_precision, self.scaler = y.shape[1:], y.size * self.precision, MinMaxScaler()
        ic(self.total_precision)
        with gmpy2.context(precision=self.total_precision):
            y_scaled = self.scaler.scale(y).flatten() # scale labels to be in [0, 1]
            phi_inv_decimal_list = phi_inverse(y_scaled) # compute φ^(-1)(y) for all labels
            phi_inv_binary_list = decimal_to_binary(phi_inv_decimal_list, self.precision) # convert to a binary list
            phi_inv_binary = ''.join(phi_inv_binary_list) # concatenate all binary strings together to get a scalar binary number
            if len(phi_inv_binary) != self.total_precision: raise ValueError(f"Expected {self.total_precision} binary digits but got {len(phi_inv_binary)}.")
            phi_inv_scalar = binary_to_decimal(phi_inv_binary) # convert to a scalar decimal
            self.alpha = phi(phi_inv_scalar) # apply φ to φ^(-1)(y) to recover y but now y is a scalar

            if VERBOSE >= 2: print(f'With {self.precision} digits of binary precision, alpha has {len(str(self.alpha))} digits of decimal precision.')
            if VERBOSE >= 3: print(f'{self.alpha=}')
        return self

    @Timing("predict: ", enabled=VERBOSE)
    def transform(self, X_idxs):
        y_size = np.array(self.y_shape).prod()
        sample_idxs = np.tile(np.arange(y_size), (len(X_idxs), 1)) + X_idxs[:, None] * y_size
        with gmpy2.context(precision=self.total_precision): # TODO: setting precision here doesn't work rn
            raw_pred = logistic_decoder_parallel(self.total_precision, self.alpha, self.precision, sample_idxs.flatten())
            # raw_pred = logistic_decoder(self.alpha, sample_idxs.flatten(), self.precision)
        return self.scaler.unscale(raw_pred.reshape(sample_idxs.shape))

    def fit_transform(self, X, y): return self.fit(X, y).transform(X)

#***** main *****

def main():
    precision = 8
    # X, y = np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)
    X, y = get_scatter_data()
    # X, y = get_arc_agi2()
    # X, y = X[:30], y[:30]
    X_idxs = np.arange(len(X))
    ic(X.shape, y.shape)

    srm = SRM(precision)
    srm.fit(X, y)
    y_pred = srm.transform(X_idxs)
    plot_data(X, y, y_pred)


if __name__ == '__main__':
    main()

"""

currently, we learn f(idx) -> y
need x -> idx
to get x->index, just use an overiffted polynomial!
do polynomial interpolation on (x[idx], [idx]) and then map this polynomial over!
but then we need to represent this polynomial as one scalar too!

instead of learning f(idx) -> y
let's learn an overfitted polynomial to our data g(x) -> y which has params theta = [theta_1, ..., theta_n]
now let's encode theta into a single number using f(theta) = alpha
now when we get a number, we do f(x)
f(i) = theta_i
sum_{i=1} theta_i x_i = y_i

so we have \sum_{i=1}^d f(i) x_i = y_i



-----------------------------------------------------

Consider the dataset {(x_1, y_1), ..., (x_n, y_n)} where x_i is a d-dimensional data point and y_i is a scalar
Also, consider a polynomial of degree t with t+1 coefficients C = [c_0, ..., c_t]. defined as
g(C, x) = \sum_{i=0}^{t+1} = c_i * x_i

Now assume t is high enough that we perfectly overfit on our data, i.e. g(x_j) = y_j for all j
Then we want to learn a function f such that
f(i) = c_i for i=1, ..., t+1
In this way we learn (memorize) C.

Then we can just jut do
y_j = g(C, x_j) for j=1, ..., n

Putting it all together:

Class Model:
    def fit(X, y):
        g, c = learn_polynomial(X, y) # coefficients c, polynomial function g
        t = polynomial_degree(y)
        f = encode_polynomial(g, t)
    def transform(X):
        for i in range(t): c[i] = f(i)
        for j in range(len(x)): y[j] = g(c, X[j])


so instead of learning x_i themslves, we learn the params for a coefficient
The difference is that f maps indices to real numbers.
So if we learn f(i) = x_i, then if we shuffle i, we are in trouble.



todo: multi-core parallelize code

# coeffs = np.polynomial.polynomial.polyfit(X, y, deg=len(X)-1)
# y_pred = np.polynomial.polynomial.polyval(X, coeffs)  # Use matching polyval function
# ic(X.shape, y.shape, y_pred.shape)
# plot_data(X, y, y_pred)

# coeffs = np.polynomial.chebyshev.chebfit(X, y, deg=100)
# y_pred = np.polynomial.chebyshev.chebval(X, coeffs)
# ic(X.shape, y.shape, y_pred.shape)
# plot_data(X, y, y_pred)


spline = UnivariateSpline(X, y, s=len(X)*0.1)
y_pred = spline(X)
plot_data(X, y, y_pred)

"""

