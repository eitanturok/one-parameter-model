import functools, multiprocessing, contextlib, time, sys
import numpy as np
from mpmath import mp, asin as Arcsin, sqrt as Sqrt, sin as Sin, pi as Pi
from tqdm import tqdm

#***** utilities *****

class MinMaxScaler:
    def __init__(self, feature_range=(1e-10, 1-1e-10), epsilon=1e-10):
        self.min = self.max = self.range = None
        self.feature_range, self.epsilon = feature_range, epsilon
    def fit(self, X):
        self.min, self.max = X.min(axis=0), X.max(axis=0)
        self.range = np.maximum(self.max - self.min, self.epsilon)  # Prevent div by zero
        return self
    def transform(self, X):
        X_scaled = (X - self.min) / self.range
        return np.clip(X_scaled, *self.feature_range)  # Keep away from exact boundaries
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        X_clipped = np.clip(X, *self.feature_range)
        return X_clipped * self.range + self.min

# https://github.com/tinygrad/tinygrad/blob/44bc7dc73d7d03a909f0cc5c792c3cdd2621d787/tinygrad/helpers.py
class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
    def __enter__(self): self.st = time.perf_counter() # pylint: disable=attribute-defined-outside-init
    def __exit__(self, *exc):
        self.et = time.perf_counter() - self.st # pylint: disable=attribute-defined-outside-init
        if self.enabled: print(f"{self.prefix}{self.et:6.3f} sec"+(self.on_exit(self.et) if self.on_exit else ""), file=sys.stderr)


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
    return mp.fsum(int(b) * mp.mpf(0.5) ** (i+1) for i, b in enumerate(x_binary))

#***** math *****

def phi(x): return Sin(2 * Pi * x) ** 2

def phi_inverse(x): return np.arcsin(np.sqrt(x)) / (2.0 * np.pi)

def dyadic_decoder(alpha, p, i): return float((2 ** (i * p) * alpha) % 1)

def logistic_decoder(alpha, full_precision, p, i):
    mp.prec = full_precision
    return float(Sin(2 ** (i * p) * Arcsin(Sqrt(alpha))) ** 2)

# This is 10x+ faster than logistic_decoder because:
# 1) arcsin(sqrt(alpha)) is precomputed once instead of every iteration
# 2) mp.prec is set to minimum needed precision per iteration instead of full precision

# Explanation:
# 1) Why is mp.prec = p * (i + 1) + 1?
#    Each iteration decodes another p bits of alpha.
#    So in iteration i, we need the first (i+1)*p bits of alpha (i is 0-indexed).
#    The +1 at the end adds an extra bit for numerical stability.
#    This is instead of using mp.prec = full_precision = p*len(X) bits.
# 2) When can we use lower precision?
#    Alpha is stored in logistic space and requires full precision.
#    We transform alpha to dyadic space by computing arcsin_sqrt_alpha = φ⁻¹(alpha) in full precision.
#    In dyadic space, iteration i only needs the first (i+1)*p bits of alpha.
#    Therefore, we can compute everything else using p*(i+1) bits instead of full_precision = p*len(X) bits.
#    Note: we can only use this reduced precision in dyadic space, not logistic space.
def logistic_decoder_fast(arcsin_sqrt_alpha, p, i):
    mp.prec = p * (i + 1) + 1
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

#***** model *****

class OneParameterModel:
    def __init__(self, precision:int=8, n_workers:int=8):
        self.precision = precision # number of bits per sample
        self.n_workers = n_workers
        self.scaler = MinMaxScaler()

    @Timing("fit: ")
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

    @Timing("predict: ")
    def predict(self, idxs:np.ndarray, fast:bool=True):
        full_idxs = (np.tile(np.arange(self.y_size), (len(idxs), 1)) + idxs[:, None] * self.y_size).flatten().tolist()

        # choose the fast or slow logistic decoder
        if fast: decoder = functools.partial(logistic_decoder_fast, Arcsin(Sqrt(self.alpha)), self.precision)
        else: decoder = functools.partial(logistic_decoder, self.alpha, self.full_precision, self.precision)

        # run the decoder sequentially/in parallel and then unscale+reshape y_pred
        if self.n_workers == 0:
            y_pred = np.array([decoder(idx) for idx in tqdm(full_idxs)])
        else:
            with multiprocessing.Pool(self.n_workers) as p:
                y_pred = np.array(list(tqdm(p.imap(decoder, full_idxs), total=len(full_idxs), desc="Decoding")))
        return self.scaler.inverse_transform(y_pred).reshape((-1, *self.y_shape))

    @Timing("verify: ")
    def verify(self, y: np.ndarray, y_pred: np.ndarray):
        # check logistic decode error is within theoretical bounds (section 2.5 https://arxiv.org/pdf/1904.12320)
        # |y - y_pred| <= π / 2^(p-1) when y, y_pred ∈ [0, 1]
        # |y - y_pred| <= (π / 2^(p-1)) * range when y, y_pred ∈ [min, max], range = max - min

        # multiply the tolerance by the scaler range to account for scaling
        tolerance = np.pi / 2 ** (self.precision - 1) * self.scaler.range
        np.testing.assert_allclose(y_pred, y, atol=tolerance, rtol=0)

def fast_decode(alpha, p, y_scaled, n_workers=8):
    y_idxs = list(range(len(y_scaled)))
    decoder = functools.partial(logistic_decoder_fast, Arcsin(Sqrt(alpha)), p)
    with multiprocessing.Pool(n_workers) as p:
        y_pred = np.array(list(tqdm(p.imap(decoder, y_idxs), total=len(y_idxs), desc="Decoding")))
    return y_pred
