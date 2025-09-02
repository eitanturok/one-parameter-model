# Run with ` uv run src/one_parameter_model/main.py`
from functools import partial
from multiprocessing import Pool

import gmpy2
import numpy as np

from utils import MinMaxScaler, Timing, tqdm, VERBOSE, WORKERS

#***** math *****

# convert decimal floats in [0, 1] to binary via https://sandbox.mc.edu/%7Ebennet/cs110/flt/dtof.html]
# cannot use python's bin() function because it only converts int to binary
def decimal_to_binary(y_decimal, precision):
    if not isinstance(y_decimal, np.ndarray): y_decimal = np.array(y_decimal)
    if y_decimal.ndim == 0: y_decimal = np.expand_dims(y_decimal, 0)

    powers = 2**np.arange(precision)
    y_powers = y_decimal[:, np.newaxis] * powers[np.newaxis, :]
    y_fractional = y_powers % 1 # extract the fractional part of y_powers
    binary_digits = (y_fractional >= 0.5).astype(int).astype('<U1')
    return np.apply_along_axis(''.join, axis=1, arr=binary_digits).tolist()

def binary_to_decimal(y_binary):
    fractional_binary = "0." + y_binary # indicate the binary number is a float in [0, 1], not an int
    return gmpy2.mpfr(fractional_binary, base=2)

def phi(theta): return gmpy2.sin(theta * gmpy2.const_pi() * 2) ** 2

def phi_inverse(z): return np.arcsin(np.sqrt(z)) / (2.0 * np.pi)

# def logistic_decoder_simple(alpha, sample_idx, precision):
#    return float(gmpy2.sin(gmpy2.mpfr(2) ** (sample_idx * precision) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2)

# def logistic_decoder(alpha, sample_idxs, precision):
#     exponents = (sample_idxs * precision).astype(int).tolist()
#     mod = gmpy2.mpz(gmpy2.mpfr(2) ** (max(exponents) + 1)) # make mod so big we don't use it
#     samples = gmpy2.powmod_exp_list(2, exponents, mod)
#     const = gmpy2.asin(gmpy2.sqrt(alpha))
#     y_pred = np.array([float(gmpy2.sin(sample * const) **2) for sample in tqdm(samples, desc="Decoding")])
#     return y_pred

def logistic_decoder_single(total_prec, alpha, prec, idx):
    gmpy2.get_context().precision = total_prec # set the precision in each pool/thread
    # gmpy2.get_context().precision = prec # set the precision in each pool/thread
    return float(gmpy2.sin(gmpy2.mpfr(2) ** (idx * prec) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2)

def logistic_decoder_parallel(total_prec, alpha, prec, idxs):
    fxn = partial(logistic_decoder_single, total_prec, alpha, prec)
    with Pool(WORKERS) as p:
        return np.array(list(tqdm(p.imap(fxn, idxs), total=len(idxs), desc="Logistic Decoder")))

#***** model *****

# scalar reasoning model
class SRM:
    def __init__(self, precision):
        self.precision = precision # binary precision, not decimal precision, for a single number

    @Timing("fit: ", enabled=VERBOSE)
    def fit(self, X, y):

        # compute alpha with arbitrary floating-point precision
        self.y_shape, self.total_precision, self.scaler = y.shape[1:], y.size * self.precision, MinMaxScaler() # pylint: disable=attribute-defined-outside-init
        with gmpy2.context(precision=self.total_precision):
            # scale labels to be in [0, 1]
            y_scaled = self.scaler.scale(y.flatten())
            # compute φ^(-1)(y_i) for all labels i=1, ..., n
            phi_inv_decimal_list = phi_inverse(y_scaled)
            # convert to a binary list bin(φ^(-1)(y_i)) i=1, ..., n
            phi_inv_binary_list = decimal_to_binary(phi_inv_decimal_list, self.precision)
            # concatenate all binary strings together to get a scalar binary number bin(φ^(-1)(y))
            phi_inv_binary = ''.join(phi_inv_binary_list)
            if len(phi_inv_binary) != self.total_precision:
                raise ValueError(f"Expected {self.total_precision} binary digits but got {len(phi_inv_binary)}.")
            # convert to a scalar decimal
            phi_inv_scalar = binary_to_decimal(phi_inv_binary)
            # apply φ to φ^(-1)(y) to recover y=[y_1, ..., y_n] but now y is a scalar
            self.alpha = phi(phi_inv_scalar) # pylint: disable=attribute-defined-outside-init

            if VERBOSE >= 2: print(f'With {self.precision} digits of binary precision, alpha has {len(str(self.alpha))} digits of decimal precision.')
            if VERBOSE >= 3: print(f'{self.alpha=}')
        return self

    @Timing("predict: ", enabled=VERBOSE)
    def transform(self, X_idxs):
        y_size = np.array(self.y_shape).prod()
        sample_idxs = np.tile(np.arange(y_size), (len(X_idxs), 1)) + X_idxs[:, None] * y_size
        raw_pred = logistic_decoder_parallel(self.total_precision, self.alpha, self.precision, sample_idxs.flatten())
        return self.scaler.unscale(raw_pred).reshape((-1,) + self.y_shape)

    def fit_transform(self, X, y): return self.fit(X, y).transform(X)
