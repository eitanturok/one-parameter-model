# Run with  `uv run src/arc-agi2/srm.py`
from functools import reduce
import numpy as np
from tqdm import tqdm
import gmpy2

from icecream import ic

from data import get_scatter_data, get_arc_agi2, plot_data

#***** utils *****

class MinMaxScaler:
    def __init__(self, epsilon:float=1e-20):
        self.min = self.max = None
        self.epsilon = epsilon # to prevent division by zero

    def scale(self, X):
        self.min, self.max = X.min(axis=0), X.max(axis=0)
        return (X - self.min) / ((self.max - self.min) + self.epsilon)

    def unscale(self, X):
        return X * ((self.max - self.min) + self.epsilon) + self.min

#***** math *****

def phi(theta):
    return gmpy2.sin(theta * gmpy2.const_pi() * 2) ** 2

def phi_inverse(z):
    return np.arcsin(np.sqrt(z)) / (2.0 * np.pi)

# https://sandbox.mc.edu/%7Ebennet/cs110/flt/dtof.html
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
    exponents = (sample_idxs * precision).tolist()
    mod = gmpy2.mpz(gmpy2.mpfr(2) ** (max(exponents) + 1)) # make mod so big we don't use it
    samples = gmpy2.powmod_exp_list(2, exponents, mod)
    const = gmpy2.asin(gmpy2.sqrt(alpha))
    y_pred = np.array([float(gmpy2.sin(sample * const) **2) for sample in tqdm(samples, desc="Decoding")])
    return y_pred

#***** model *****

# scalar reasoning model
class SRM:
    def __init__(self, precision, verbose=True):
        self.precision = precision # binary precision, not decimal precision
        self.verbose = verbose
        self.scaler = None
        self.y_data_shape = None
        self.alpha = None

    def _get_alpha(self, y_decimal):
        # encode all labels with φ^(-1) and convert to binary
        phi_inv_list = phi_inverse(y_decimal)
        phi_inv_binary = decimal_to_binary(phi_inv_list, self.precision)

        # set precision for arbitrary floating-point precision
        total_precision = len(y_decimal) * self.precision
        gmpy2.get_context().precision = total_precision
        if len(phi_inv_binary) != total_precision:
            raise ValueError(f"Expected {total_precision} binary digits but got {len(phi_inv_binary)}.")

        # convert to decimal with arbitrary number of floating point precision
        phi_inv_scalar = binary_to_decimal(phi_inv_binary)
        return phi(phi_inv_scalar)

    def fit(self, X, y):
        self.scaler = MinMaxScaler()
        y_scaled = self.scaler.scale(y)
        self.y_data_shape = y_scaled.shape[1:] # dimension 0 is batch size
        self.alpha = self._get_alpha(y_scaled.flatten())
        return self

    def transform(self, X_idxs):
        y_size = np.array(self.y_data_shape).prod()
        base_indices = np.tile(np.arange(y_size), (len(X_idxs), 1))
        flat_indices = (base_indices + X_idxs[:, None] * y_size).flatten()
        batch_shape = (len(X_idxs),) + self.y_data_shape
        raw_pred = logistic_decoder_list(self.alpha, flat_indices, self.precision).reshape(batch_shape)
        return self.scaler.unscale(raw_pred)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

#***** main *****

def main():
    precision = 10
    # X, y = np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)
    # X, y = get_scatter_data()
    X, y = get_arc_agi2()
    X, y = X[:3], y[:3]
    X_idxs = np.arange(len(X))
    ic(X.shape, y.shape)

    srm = SRM(precision)
    srm.fit(X, y)
    y_pred = srm.transform(np.array([2]))
    # plot_data(X[0], y[0], y_pred[0])
    plot_data(X[2], y[2], y_pred[0])


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

