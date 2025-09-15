# Run with ` uv run src/one_parameter_model/main.py`
from functools import partial
from multiprocessing import Pool

import gmpy2
import numpy as np

from .utils import binary_to_decimal, decimal_to_binary, MinMaxScaler, Timing, tqdm, VERBOSE

#***** math *****

def phi(x): return gmpy2.sin(x * 2 * gmpy2.const_pi()) ** 2

def phi_inverse(x): return np.arcsin(np.sqrt(x)) / (2.0 * np.pi)

# def _logistic_decoder(idx, precision, alpha):
#     return gmpy2.sin(gmpy2.mpfr(2) ** (idx * precision) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2

# compute the value using `total_precision` precision
# then truncate to `prec` bits of precision and cast to a regular python float
def logistic_decoder_single(total_precision, alpha, precision, idx):
    # set precision to np bits
    gmpy2.get_context().precision = total_precision
    # compute the logistic map
    val = gmpy2.sin(gmpy2.mpfr(2) ** (idx * precision) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2
    # set precision to p bits
    return float(gmpy2.mpfr(val, precision=precision))

def logistic_decoder(total_precision, alpha, precision, idxs, workers):
    # sequential if workers is 0
    if workers == 0:
        return np.array([logistic_decoder_single(total_precision, alpha, precision, idx) for idx in tqdm(idxs)])
    fxn = partial(logistic_decoder_single, total_precision, alpha, precision)
    print(__name__)
    with Pool(workers) as p:
        return np.array(list(tqdm(p.imap(fxn, idxs), total=len(idxs), desc="Logistic Decoder")))

#***** model *****

# one parameter model
class OneParameterModel:
    def __init__(self, precision, workers=0):
        self.precision = precision # binary precision, not decimal precision, for a single number
        self.workers = workers

    @Timing("fit: ", enabled=VERBOSE)
    def fit(self, X, y=None):
        # if the dataset is unsupervised, treat the data X like the labels y
        if y is None: y = X

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
    def predict(self, idxs):
        y_size = np.array(self.y_shape).prod()
        full_idxs = (np.tile(np.arange(y_size), (len(idxs), 1)) + idxs[:, None] * y_size).flatten()
        raw_pred = logistic_decoder(self.total_precision, self.alpha, self.precision, full_idxs, self.workers)
        return self.scaler.unscale(raw_pred).reshape((-1, *self.y_shape))

    def fit_predict(self, X, y): return self.fit(X, y).predict(X)
