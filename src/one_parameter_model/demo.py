import time, contextlib, sys, multiprocessing
from functools import partial
# from tqdm import tqdm
import numpy as np
from mpmath import mp, workprec
from mpmath import asin as Arcsin
from mpmath import sqrt as Sqrt
from mpmath import sin as Sin
from mpmath import pi as Pi
from .utils import MinMaxScaler, Timing, tqdm

#***** binary *****

# converts a 1D np.array from decimal to binary
def decimal_to_binary(x, prec):
    bits = []
    for _ in range(prec):
        bits.append(np.round(x))
        x = dyadic_map(x)
    return ''.join(map(str, np.array(bits).astype(int).T.ravel()))

# converts an arbitrary-precision scalar from binary to decimal
def binary_to_decimal(y_binary):
    return mp.fsum(int(b) * mp.mpf(0.5) ** (i+1) for i, b in enumerate(y_binary))

#***** math *****

def dyadic_map(x): return (2 * x) % 1

def phi_inverse(x): return np.arcsin(np.sqrt(x)) / (2.0 * np.pi)

def phi(x): return Sin(2 * Pi * x) ** 2

def dyadic_decoder(alpha, i, p): return float((2 ** (i * p) * alpha) % 1)

def logistic_decoder(alpha, full_precision, p, i):
    mp.prec = full_precision
    ret = Sin(2 ** (i * p) * Arcsin(Sqrt(alpha))) ** 2
    mp.prec = p
    return float(ret)

def logistic_decoder_fast(arcsin_sqrt_alpha, p, i):
    # (i+1) because i is 0-indexed
    # +1 at the end adds an extra bit of precision for numerical stability
    mp.prec = p * (i + 1) + 1
    return float(Sin(2 ** (i * p) * arcsin_sqrt_alpha) ** 2)

#***** model *****

class OneParameterModel:
    def __init__(self, precision:int=8, n_workers:int=8):
        self.scaler = self.alpha = self.full_precision = self.y_shape = self.y_size = None
        self.precision, self.n_workers = precision, n_workers # number of bits per sample

    @Timing("fit: ")
    def fit(self, X:np.ndarray, Y:np.ndarray|None=None):
        # if the dataset is unsupervised, treat X like your y
        if Y is None: Y = X

        # store shape/size of a single label y
        self.y_shape = Y.shape[1:]
        self.y_size = np.array(self.y_shape, dtype=int).prod().item()

        # scale labels to be in [0, 1]
        self.scaler = MinMaxScaler()
        Y_scaled = self.scaler.scale(Y.flatten())
        assert 0 <= Y_scaled.min() <= Y_scaled.max() <= 1, f"Y_scaled must be in [0, 1] but got [{Y_scaled.min()}, {Y_scaled.max()}]"

        # compute alpha with arbitrary floating-point precision set to full_precision bits
        self.full_precision = Y.size * self.precision # number of bits in whole dataset
        mp.prec = self.full_precision

        # 1. apply φ^(-1)
        phi_inv_decimal_list = phi_inverse(Y_scaled)

        # 2. convert to binary
        phi_inv_binary_list = decimal_to_binary(phi_inv_decimal_list, self.precision)

        # 3. concatenate all binary strings together into a scalar
        phi_inv_binary_scalar = ''.join(phi_inv_binary_list)
        if len(phi_inv_binary_scalar) != self.full_precision:
            raise ValueError(f"Expected {self.full_precision} bits but got {len(phi_inv_binary_scalar)} bits.")

        # 4. convert to decimal
        phi_inv_decimal_scalar = binary_to_decimal(phi_inv_binary_scalar)

        # 5. apply φ
        self.alpha = phi(phi_inv_decimal_scalar)
        return self

    @Timing("predict: ")
    def predict(self, idxs:np.array, fast:bool=True):
        assert self.scaler is not None
        full_idxs = (np.tile(np.arange(self.y_size), (len(idxs), 1)) + idxs[:, None] * self.y_size).flatten().tolist()

        if fast: fxn = partial(logistic_decoder_fast, Arcsin(Sqrt(self.alpha)), self.precision)
        else: fxn = partial(logistic_decoder, self.alpha, self.full_precision, self.precision)
        if self.n_workers == 0:
            y_pred = np.array([fxn(idx) for idx in tqdm(full_idxs)])
        else:
            with multiprocessing.Pool(self.n_workers) as p:
                y_pred = np.array(list(tqdm(p.imap(fxn, full_idxs), total=len(full_idxs), desc="Logistic Decoder")))
        return self.scaler.unscale(y_pred).reshape((-1, *self.y_shape))

    @Timing("verify: ")
    def verify(self, X, Y_pred):
        tolerance = np.pi / 2 ** (self.precision-1)
        errors = np.abs(np.array(Y_pred) - np.array(X))
        bad_idx = np.where(errors >= tolerance)[0]
        assert len(bad_idx) == 0, f"Errors at {len(bad_idx)} indices with precision={self.precision}, {tolerance=:.2e}\n" \
                                   f"  indices: {bad_idx[:10].tolist()}{'...' if len(bad_idx) > 10 else ''}\n" \
                                   f"  errors: {errors[bad_idx][:10].tolist()}{'...' if len(bad_idx) > 10 else ''}\n" \
                                   f"  X: {np.array(X)[bad_idx][:10].tolist()}{'...' if len(bad_idx) > 10 else ''}\n" \
                                   f"  Y_pred: {np.array(Y_pred)[bad_idx][:10].tolist()}{'...' if len(bad_idx) > 10 else ''}\n" \
                                   f"  X_bin: {[decimal_to_binary(np.array([X[i]]), self.precision) for i in bad_idx[:10]]}{'...' if len(bad_idx) > 10 else ''}\n" \
                                   f"  Y_bin: {[decimal_to_binary(np.array([Y_pred[i]]), self.precision) for i in bad_idx[:10]]}{'...' if len(bad_idx) > 10 else ''}\n" \
                                   f"  max_error: {errors.max():.2e}"
        print(f"Passes with {tolerance=}")


def main():
    np.random.seed(0)
    n_samples = 5000
    X = np.random.uniform(0, 1, n_samples)

    model = OneParameterModel(precision=12)
    model.fit(X)
    Y_pred = model.predict(np.arange(n_samples))
    model.verify(X, Y_pred)


if __name__ == '__main__':
    main()
