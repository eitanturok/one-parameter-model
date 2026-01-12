import functools, multiprocessing, contextlib, time, sys
import numpy as np
from mpmath import mp, asin as Arcsin, sqrt as Sqrt, sin as Sin, pi as Pi
from tqdm import tqdm

#***** utilities *****

class MinMaxScaler:
    def __init__(self, epsilon=1e-10):
        self.min = self.max = self.range = None
        self.epsilon = epsilon
    def fit(self, X):
        self.min, self.max = X.min(axis=0), X.max(axis=0)
        self.range = np.maximum(self.max - self.min, self.epsilon)  # Prevent div by zero
        return self
    def transform(self, X):
        X_scaled = (X - self.min) / self.range
        return np.clip(X_scaled, self.epsilon, 1 - self.epsilon)  # Keep away from exact boundaries
    def fit_transform(self, X): return self.fit(X).transform(X)
    def inverse_transform(self, X):
        X_clipped = np.clip(X, self.epsilon, 1 - self.epsilon)
        return X_clipped * self.range + self.min

#***** binary *****

def dyadic_map(X:np.ndarray): return (2 * X) % 1

def decimal_to_binary(x_decimal:np.ndarray|float|int|list|tuple, precision:int):
    # converts a 1D sequence from decimal to binary, assume all values in [0, 1]
    if isinstance(x_decimal, (float, int)): x_decimal = np.array([x_decimal], dtype=float)
    elif isinstance(x_decimal, (list, tuple)): x_decimal = np.array(x_decimal, dtype=float)
    assert 0 <= x_decimal.min() <= x_decimal.max() <= 1, f"expected x_decimal to be in [0, 1] but got [{x_decimal.min()}, {x_decimal.max()}]"
    bits = []
    for _ in range(precision):
        bits.append(np.round(x_decimal))
        x_decimal = dyadic_map(x_decimal)
    return ''.join(map(str, np.array(bits).astype(int).T.ravel()))

def binary_to_decimal(x_binary:np.ndarray):
    # converts an arbitrary-precision scalar from binary to decimal
    return mp.fsum(int(b) * mp.mpf(0.5) ** (i+1) for i, b in tqdm(enumerate(x_binary), desc="Encoding", total=len(x_binary)))

#***** math *****

def phi(x): return Sin(2 * Pi * x) ** 2

def phi_inverse(x): return np.arcsin(np.sqrt(x)) / (2.0 * np.pi)

def dyadic_decoder(alpha, p, i): return float((2 ** (i * p) * alpha) % 1)

def logistic_decoder(alpha, full_precision, p, i):
    mp.prec = full_precision
    return float(Sin(2 ** (i * p) * Arcsin(Sqrt(alpha))) ** 2)

def logistic_decoder_fast(arcsin_sqrt_alpha, p, i):
    mp.prec = p * (i + 1) + 2  # extra bits to reduce numerical errors ??
    return float(Sin(2 ** (i * p) * arcsin_sqrt_alpha) ** 2)

# todo: fix this
def dyadic_encoder(X, precision, full_precision):
    # set the arbitrary precision before computing anything
    mp.prec = full_precision

    # 1. convert to binary
    binary_list = decimal_to_binary(X, precision)

    # 2. concatenate all binary strings together into a scalar
    binary_scalar = ''.join(binary_list)
    if len(binary_scalar) != full_precision:
        raise ValueError(f"Expected {full_precision} bits but got {len(binary_scalar)} bits.")

    # 3. convert to decimal
    decimal_scalar = binary_to_decimal(binary_scalar)
    return decimal_scalar

def logistic_encoder(X, precision, full_precision):
    # set the arbitrary precision before computing anything
    mp.prec = full_precision

    # 1. apply φ^(-1)
    phi_inv_decimal_list = phi_inverse(X)

    # 2. convert to binary
    phi_inv_binary_list = decimal_to_binary(phi_inv_decimal_list, precision)

    # 3. concatenate all binary strings together into a scalar
    phi_inv_binary_scalar = ''.join(phi_inv_binary_list)
    if len(phi_inv_binary_scalar) != full_precision:
        raise ValueError(f"Expected {full_precision} bits but got {len(phi_inv_binary_scalar)} bits.")

    # 4. convert to decimal
    phi_inv_decimal_scalar = binary_to_decimal(phi_inv_binary_scalar)

    # 5. apply φ
    alpha = phi(phi_inv_decimal_scalar)
    return alpha

# only works for one ARC-AGI-2 example, i.e. we only encoded one ARC-AGI-2 example into alpha
def decode(alpha, full_precision, p, y_scaled):
    y_idxs = list(range(len(y_scaled)))
    return np.array([logistic_decoder(alpha, full_precision, p, i) for i in tqdm(y_idxs, total=len(y_idxs), desc="Decoding")], dtype=np.float32)

# only works for one ARC-AGI-2 example, i.e. we only encoded one ARC-AGI-2 example into alpha
def fast_decode(alpha, p, y_scaled, n_workers=8):
    y_idxs = list(range(len(y_scaled)))
    mp.prec = p * len(y_scaled) # compute arcsin(sqrt(alpha)) with full precision
    decoder = functools.partial(logistic_decoder_fast, Arcsin(Sqrt(alpha)), p)
    with multiprocessing.Pool(n_workers) as p:
        y_pred = np.array(list(tqdm(p.imap(decoder, y_idxs), total=len(y_idxs), desc="Decoding")), dtype=np.float32)
    return y_pred

#***** model *****

class OneParameterModel:
    def __init__(self, precision:int=8, n_workers:int=8):
        self.precision = precision # number of bits per sample
        self.n_workers = n_workers
        self.scaler = MinMaxScaler()

    def fit(self, X:np.ndarray, y:np.ndarray|None=None):
        # if the dataset is unsupervised, treat X like the y
        if y is None: y = X

        # store shape/size of a single label y
        self.y_shape = y.shape[1:] # pylint: disable=attribute-defined-outside-init
        self.y_size = np.array(self.y_shape, dtype=int).prod().item() # pylint: disable=attribute-defined-outside-init

        # scale labels to be in [0, 1]
        y_scaled = self.scaler.fit_transform(y.flatten())
        assert 0 <= y_scaled.min() <= y_scaled.max() <= 1, f"y_scaled must be in [0, 1] but got [{y_scaled.min()}, {y_scaled.max()}]"

        # compute alpha with arbitrary floating-point precision
        self.full_precision = y.size * self.precision # pylint: disable=attribute-defined-outside-init
        self.alpha = logistic_encoder(y_scaled, self.precision, self.full_precision) # pylint: disable=attribute-defined-outside-init
        return self

    def predict(self, idxs:np.ndarray, fast:bool=True):
        full_idxs = (np.tile(np.arange(self.y_size), (len(idxs), 1)) + idxs[:, None] * self.y_size).flatten().tolist()

        # choose the fast or slow logistic decoder
        if fast:
            mp.prec = self.full_precision # compute arcsin(sqrt(alpha)) with full precision
            decoder = functools.partial(logistic_decoder_fast, Arcsin(Sqrt(self.alpha)), self.precision)
        else: decoder = functools.partial(logistic_decoder, self.alpha, self.full_precision, self.precision)

        # run the decoder sequentially/in parallel and then unscale+reshape y_pred
        if self.n_workers == 0:
            y_pred = np.array([decoder(idx) for idx in tqdm(full_idxs)])
        else:
            with multiprocessing.Pool(self.n_workers) as p:
                y_pred = np.array(list(tqdm(p.imap(decoder, full_idxs), total=len(full_idxs), desc="Decoding"))) # dtype=np.float32?
        return self.scaler.inverse_transform(y_pred).reshape((-1, *self.y_shape))

    def verify(self, y_pred:np.ndarray, y:np.ndarray):
        # check logistic decoder error is within the theoretical bounds (section 2.5 https://arxiv.org/pdf/1904.12320)
        # compare in scaled space because error bounds are only defined in [0, 1]
        tolerance = np.pi / 2 ** (self.precision - 1)
        np.testing.assert_allclose(self.scaler.transform(y_pred), self.scaler.transform(y), atol=tolerance, rtol=0)
