import functools, multiprocessing
import numpy as np
from mpmath import mp, asin as Arcsin, sqrt as Sqrt, sin as Sin, pi as Pi
from .utils import MinMaxScaler, Timing, tqdm

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
    ret = Sin(2 ** (i * p) * Arcsin(Sqrt(alpha))) ** 2
    # mp.prec = p
    return float(ret)

def logistic_decoder_fast(arcsin_sqrt_alpha, p, i):
    # this is more than 10x faster because
    # 1) we precompute arcsin(sqrt(alpha)) only once
    # 2) we only set the precision to the minimial needed for the current iteration
    # Why is mp.prec = p * (i + 1) + 1?
    # Each iteration i requires p bits of precision
    # So iteration i requires p * (i+1) bits of precision (i is 0-indexed)
    # The +1 at the end adds an extra bit for numerical stability
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
        assert y is not None

        # store shape/size of a single label y
        self.y_shape = y.shape[1:] # pylint: disable=attribute-defined-outside-init
        self.y_size = np.array(self.y_shape, dtype=int).prod().item() # pylint: disable=attribute-defined-outside-init

        # scale labels to be in [0, 1]
        y_scaled = self.scaler.fit_transform(y.flatten())
        assert 0 <= y_scaled.min() <= y_scaled.max() <= 1, f"y_scaled must be in [0, 1] but got [{y_scaled.min()}, {y_scaled.max()}]"

        # compute alpha with arbitrary floating-point precision
        self.full_precision: int = y.size * self.precision # pylint: disable=attribute-defined-outside-init
        self.alpha = logistic_encoder(y_scaled, self.precision, self.full_precision) # pylint: disable=attribute-defined-outside-init
        return self

    @Timing("predict: ")
    def predict(self, idxs:np.ndarray, fast:bool=True):
        assert self.scaler is not None
        full_idxs = (np.tile(np.arange(self.y_size), (len(idxs), 1)) + idxs[:, None] * self.y_size).flatten().tolist()

        # choose the fast or slow logistic decoder
        if fast: decoder = functools.partial(logistic_decoder_fast, Arcsin(Sqrt(self.alpha)), self.precision)
        else: decoder = functools.partial(logistic_decoder, self.alpha, self.full_precision, self.precision)

        # run the decoder sequentially/in parallel and then unscale+reshape y_pred
        if self.n_workers == 0:
            y_pred = np.array([decoder(idx) for idx in tqdm(full_idxs)])
        else:
            with multiprocessing.Pool(self.n_workers) as p:
                y_pred = np.array(list(tqdm(p.imap(decoder, full_idxs), total=len(full_idxs), desc="Logistic Decoder")))
        return self.scaler.inverse_transform(y_pred).reshape((-1, *self.y_shape))

    @Timing("verify: ")
    def verify(self, y: np.ndarray, y_pred: np.ndarray):
        # check logistic decode error is within theoretical bounds (section 2.5 https://arxiv.org/pdf/1904.12320)
        # |y - y_pred| <= π / 2^(p-1) when y, y_pred ∈ [0, 1]
        # |y - y_pred| <= (π / 2^(p-1)) * range when y, y_pred ∈ [min, max], range = max - min

        # multiply the tolerance by the scaler range to account for scaling
        tolerance = np.pi / 2 ** (self.precision - 1) * self.scaler.range

        try:
            np.testing.assert_allclose(y_pred, y, atol=tolerance, rtol=0, err_msg=f"Predictions don't match within tolerance")
            print(f"Passes with {tolerance=:.2e}")
        except AssertionError:
            errors = np.abs(y_pred - y)
            bad_idx = np.where(errors >= tolerance)[0]
            raise AssertionError(
                f"Errors at {len(bad_idx)} indices with precision={self.precision}, {tolerance=:.2e}\n"
                f"  indices: {bad_idx[:10].tolist()}{'...' if len(bad_idx) > 10 else ''}\n\n"
                f"  errors: {errors[bad_idx][:10].tolist()}{'...' if len(bad_idx) > 10 else ''}\n\n"
                f"  y: {y[bad_idx][:10].tolist()}{'...' if len(bad_idx) > 10 else ''}\n\n"
                f"  y_pred: {y_pred[bad_idx][:10].tolist()}{'...' if len(bad_idx) > 10 else ''}\n\n"
                f"  max_error: {errors.max():.2e}"
            )
