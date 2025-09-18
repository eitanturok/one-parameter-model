# from tinygrad helpers https://github.com/tinygrad/tinygrad/blob/44bc7dc73d7d03a909f0cc5c792c3cdd2621d787/tinygrad/helpers.py
import contextlib, time, os, math, sys, shutil
from typing import Iterable, Iterator, Generic, TypeVar
import numpy as np
import gmpy2

T = TypeVar("T")

#***** binary *****

def decimal_to_binary(y_decimal, precision):
    # convert decimal floats in [0, 1] to binary via https://sandbox.mc.edu/%7Ebennet/cs110/flt/dtof.html
    # cannot use python's bin() function because it only converts int to binary
    if not isinstance(y_decimal, np.ndarray): y_decimal = np.array(y_decimal)
    if y_decimal.ndim == 0: y_decimal = np.expand_dims(y_decimal, 0)

    powers = 2**np.arange(precision)
    y_powers = y_decimal[:, np.newaxis] * powers[np.newaxis, :]
    y_fractional = y_powers % 1 # extract the fractional part of y_powers
    binary_digits = (y_fractional >= 0.5).astype(int).astype('<U1')
    return np.apply_along_axis(''.join, axis=1, arr=binary_digits).tolist()

def binary_to_decimal(y_binary):
    # 0. shows that the binary number is a float in [0, 1], not an int
    fractional_binary = "0." + y_binary
    return gmpy2.mpfr(fractional_binary, base=2)

#**** miscellaneous *****

def getenv(key:str, default=0): return type(default)(os.getenv(key, default))
def flatten(l:Iterable[Iterable[T]]): return [item for sublist in l for item in sublist]

class MinMaxScaler:
    def __init__(self, epsilon:float=1e-20):
        self.min = self.max = None
        self.epsilon = epsilon # to prevent division by zero
    def scale(self, X):
        self.min, self.max = X.min(axis=0), X.max(axis=0)
        return (X - self.min) / ((self.max - self.min) + self.epsilon)
    def unscale(self, X):
        return X * ((self.max - self.min) + self.epsilon) + self.min

class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
    def __enter__(self): self.st = time.perf_counter_ns() # pylint: disable=attribute-defined-outside-init
    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st # pylint: disable=attribute-defined-outside-init
        if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""), file=sys.stderr)

#***** tqdm *****

class tqdm(Generic[T]):
    def __init__(self, iterable:Iterable[T]|None=None, desc:str='', disable:bool=False,
                unit:str='it', unit_scale=False, total:int|None=None, rate:int=100):
        self.iterable, self.disable, self.unit, self.unit_scale, self.rate = iterable, disable, unit, unit_scale, rate
        self.st, self.i, self.n, self.skip, self.t = time.perf_counter(), -1, 0, 1, getattr(iterable, "__len__", lambda:0)() if total is None else total
        self.set_description(desc)
        self.update(0)
    def __iter__(self) -> Iterator[T]:
        assert self.iterable is not None, "need an iterable to iterate"
        for item in self.iterable:
            yield item
            self.update(1)
        self.update(close=True)
    def __enter__(self): return self
    def __exit__(self, *_): self.update(close=True)
    def set_description(self, desc:str): self.desc = f"{desc}: " if desc else ""
    def update(self, n:int=0, close:bool=False):
        self.n, self.i = self.n+n, self.i+1
        if self.disable or (not close and self.i % self.skip != 0): return
        prog, elapsed, ncols = self.n/self.t if self.t else 0, time.perf_counter()-self.st, shutil.get_terminal_size().columns
        if elapsed and self.i/elapsed > self.rate and self.i: self.skip = max(int(self.i/elapsed)//self.rate,1)
        def HMS(t): return ':'.join(f'{x:02d}' if i else str(x) for i,x in enumerate([int(t)//3600,int(t)%3600//60,int(t)%60]) if i or x)
        def SI(x):
            return (f"{x/1000**int(g:=round(math.log(x,1000),6)):.{int(3-3*math.fmod(g,1))}f}"[:4].rstrip('.')+' kMGTPEZY'[int(g)].strip()) if x else '0.00'
        prog_text = f'{SI(self.n)}{f"/{SI(self.t)}" if self.t else self.unit}' if self.unit_scale else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
        est_text = f'<{HMS(elapsed/prog-elapsed) if self.n else "?"}' if self.t else ''
        it_text = (SI(self.n/elapsed) if self.unit_scale else f"{self.n/elapsed:5.2f}") if self.n else "?"
        suf = f'{prog_text} [{HMS(elapsed)}{est_text}, {it_text}{self.unit}/s]'
        sz = max(ncols-len(self.desc)-3-2-2-len(suf), 1)
        bar = '\r' + self.desc + (f'{100*prog:3.0f}%|{("█"*int(num:=sz*prog)+" ▏▎▍▌▋▊▉"[int(8*num)%8].strip()).ljust(sz," ")}| ' if self.t else '') + suf
        print(bar[:ncols+1], flush=True, end='\n'*close, file=sys.stderr)
    @classmethod
    def write(cls, s:str): print(f"\r\033[K{s}", flush=True, file=sys.stderr)

class trange(tqdm):
    def __init__(self, n:int, **kwargs): super().__init__(iterable=range(n), total=n, **kwargs)


#***** environment variables *****

VERBOSE = getenv("VERBOSE", 1)
