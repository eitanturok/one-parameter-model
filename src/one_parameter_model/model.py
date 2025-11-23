# Run with ` uv run src/one_parameter_model/main.py`
import functools, multiprocessing
import gmpy2
import numpy as np
from .utils import binary_to_decimal, decimal_to_binary, MinMaxScaler, Timing, tqdm, VERBOSE

#***** math *****

def phi(x): return gmpy2.sin(x * 2 * gmpy2.const_pi()) ** 2

def phi_inverse(x): return np.arcsin(np.sqrt(x)) / (2.0 * np.pi)

def _logistic_decoder(i, p, alpha):
    return gmpy2.sin(gmpy2.mpfr(2) ** (i * p) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2

# compute the value using `full_precision` precision
# then truncate to `prec` bits of precision and cast to a regular python float
def logistic_decoder_single(y_size, alpha, precision, idx):
    # set precision to np bits
    full_precision = y_size * (idx + 1) * precision
    alpha = gmpy2.mpfr(alpha, precision=full_precision)
    gmpy2.get_context().precision = full_precision
    # compute the logistic map
    val = gmpy2.sin(gmpy2.mpfr(2) ** (idx * precision) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2
    # set precision to p bits
    val = gmpy2.mpfr(val, precision=precision)
    return float(val)

def logistic_decoder(y_size, alpha, precision, idxs, workers):
    # sequential if workers == 0 else parallelized
    if workers == 0: return np.array([logistic_decoder_single(y_size, alpha, precision, idx) for idx in tqdm(idxs)])
    fxn = functools.partial(logistic_decoder_single, y_size, alpha, precision)
    with multiprocessing.Pool(workers) as p:
        return np.array(list(tqdm(p.imap(fxn, idxs), total=len(idxs), desc="Logistic Decoder")))

#***** model *****

# one parameter model
class OneParameterModel:
    def __init__(self, precision, workers=8):
        self.precision = precision # binary precision, not decimal precision, for a single number
        self.workers = workers

    @Timing("fit: ", enabled=VERBOSE)
    def fit(self, X, Y=None):
        # if the dataset is unsupervised, treat the data X like the labels y
        if Y is None: Y = X

        # store shape/size of a single label y
        self.y_shape = Y.shape[1:]  # pylint: disable=attribute-defined-outside-init
        self.y_size = np.array(self.y_shape, dtype=int).prod().item()  # pylint: disable=attribute-defined-outside-init

        # scale labels to be in [0, 1]
        self.scaler = MinMaxScaler()  # pylint: disable=attribute-defined-outside-init
        Y_scaled = self.scaler.scale(Y.flatten())

        # compute alpha with arbitrary floating-point precision of np bits
        full_precision = Y.size * self.precision
        with gmpy2.context(precision=full_precision):

            # 1. apply φ^(-1)
            phi_inv_decimal_list = phi_inverse(Y_scaled)
            # 2. convert to binary
            phi_inv_binary_list = decimal_to_binary(phi_inv_decimal_list, self.precision)
            # 3. concatenate all binary strings together into a scalar
            phi_inv_binary = ''.join(phi_inv_binary_list)
            if len(phi_inv_binary) != full_precision:
                raise ValueError(f"Expected {full_precision} binary digits but got {len(phi_inv_binary)}.")
            # 4. convert to decimal
            phi_inv_scalar = binary_to_decimal(phi_inv_binary)
            # 5. apply φ
            self.alpha = phi(phi_inv_scalar) # pylint: disable=attribute-defined-outside-init
        return self

    @Timing("predict: ", enabled=VERBOSE)
    def predict(self, idxs):
        # full_idxs = (np.tile(np.arange(self.y_size), (len(idxs), 1)) + idxs[:, None] * self.y_size).flatten().tolist()
        full_idxs = [0]
        raw_pred = logistic_decoder(self.y_size, self.alpha, self.precision, full_idxs, self.workers)
        return self.scaler.unscale(raw_pred).reshape((-1, *self.y_shape))
        """
        instead of full_idxs including every single index upon which we call logistic_decoder
        what if full_idxs just includes the indices of the one specific example we are looking at
        then, when we call logistic_decoder, we set a high enough precision that we extract all the elements of that single example

        so logistic decoder at the end doesn't extract just p bits, but rather y_size * p bits so we get all the bits of the entire
        multi-dimensional vector.

        this means we overall call logistic_decoder fewer times.

        mayve we should compute the full logistic decoder with all the np bits of precision once
        and then all we have to do is extract the ip bits, multiple times. this removes the more complex operations of
        actually computing the logistic decoder value.

        can we cache part of the logistic map so we don't need to compute all of it?
        """
