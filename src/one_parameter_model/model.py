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
    # indicate the binary number is a float in [0, 1], not an int
    fractional_binary = "0." + y_binary
    return gmpy2.mpfr(fractional_binary, base=2)

def phi(x): return gmpy2.sin(x * 2 * gmpy2.const_pi()) ** 2

def phi_inverse(x): return np.arcsin(np.sqrt(x)) / (2.0 * np.pi)

# compute the value using `total_precision` precision
# then truncate to `prec` bits of precision and cast to a regular python float
def logistic_decoder_single(total_prec, alpha, prec, idx):
    gmpy2.get_context().precision = total_prec
    val = gmpy2.sin(gmpy2.mpfr(2) ** (idx * prec) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2
    return float(gmpy2.mpfr(val, precision=prec))

def logistic_decoder_parallel(total_prec, alpha, prec, idxs):
    fxn = partial(logistic_decoder_single, total_prec, alpha, prec)
    with Pool(WORKERS) as p:
        return np.array(list(tqdm(p.imap(fxn, idxs), total=len(idxs), desc="Logistic Decoder")))

def logistic_decoder_sequential(total_prec, alpha, prec, idxs):
    return np.array([logistic_decoder_single(total_prec, alpha, prec, idx) for idx in idxs])

#***** model *****

# one parameter model
class ScalarModel:
    def __init__(self, precision):
        self.precision = precision # binary precision, not decimal precision, for a single number

    @Timing("fit: ", enabled=VERBOSE)
    def fit(self, X, y):
        self.y_shape = y.shape[1:] # pylint: disable=attribute-defined-outside-init
        self.total_precision = y.size * self.precision # pylint: disable=attribute-defined-outside-init

        # scale labels to be in [0, 1]
        self.scaler = MinMaxScaler() # pylint: disable=attribute-defined-outside-init
        y_scaled = self.scaler.scale(y.flatten())

        # compute alpha with arbitrary floating-point precision
        with gmpy2.context(precision=self.total_precision):

            # 1. compute φ^(-1) for all labels
            phi_inv_decimal_list = phi_inverse(y_scaled)
            # 2. convert to a binary
            phi_inv_binary_list = decimal_to_binary(phi_inv_decimal_list, self.precision)
            # 3. concatenate all binary strings together
            phi_inv_binary = ''.join(phi_inv_binary_list)
            if len(phi_inv_binary) != self.total_precision:
                raise ValueError(f"Expected {self.total_precision} binary digits but got {len(phi_inv_binary)}.")

            # 4. convert to a scalar decimal
            phi_inv_scalar = binary_to_decimal(phi_inv_binary)
            # 5. apply φ to the scalar
            self.alpha = phi(phi_inv_scalar) # pylint: disable=attribute-defined-outside-init

            if VERBOSE >= 2: print(f'With {self.precision} digits of binary precision, alpha has {len(str(self.alpha))} digits of decimal precision.')
            if VERBOSE >= 3: print(f'{self.alpha=}')
        return self

    @Timing("predict: ", enabled=VERBOSE)
    def predict(self, X_idxs):
        y_size = np.array(self.y_shape).prod()
        sample_idxs = (np.tile(np.arange(y_size), (len(X_idxs), 1)) + X_idxs[:, None] * y_size).flatten()
        raw_pred = logistic_decoder_parallel(self.total_precision, self.alpha, self.precision, sample_idxs)
        # raw_pred = logistic_decoder_sequential(self.total_precision, self.alpha, self.precision, sample_idxs)
        return self.scaler.unscale(raw_pred).reshape((-1, *self.y_shape))

    def fit_predict(self, X, y): return self.fit(X, y).predict(X)
