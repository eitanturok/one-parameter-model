import time
import numpy as np
import gmpy2
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from joblib import Parallel, delayed

def logistic_decoder_single(y_size, alpha, precision, idx):
    total_precision = y_size * (idx + 1) * precision
    gmpy2.get_context().precision = total_precision
    val = gmpy2.sin(gmpy2.mpfr(2) ** (idx * precision) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2
    return float(gmpy2.mpfr(val, precision=precision))

# 1. Original multiprocessing
def method_multiprocessing(y_size, alpha, precision, idxs, workers):
    fxn = partial(logistic_decoder_single, y_size, alpha, precision)
    with Pool(workers) as p:
        return np.array(list(p.map(fxn, idxs)))

# 2. Threading
def method_threading(y_size, alpha, precision, idxs, workers):
    fxn = partial(logistic_decoder_single, y_size, alpha, precision)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return np.array(list(ex.map(fxn, idxs)))

# 3. ProcessPoolExecutor (cleaner multiprocessing)
def method_process_executor(y_size, alpha, precision, idxs, workers):
    fxn = partial(logistic_decoder_single, y_size, alpha, precision)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return np.array(list(ex.map(fxn, idxs)))

# 4. Joblib
def method_joblib(y_size, alpha, precision, idxs, workers):
    return np.array(Parallel(n_jobs=workers)(
        delayed(logistic_decoder_single)(y_size, alpha, precision, idx) for idx in idxs
    ))

# 5. Sequential baseline
def method_sequential(y_size, alpha, precision, idxs, workers):
    return np.array([logistic_decoder_single(y_size, alpha, precision, idx) for idx in idxs])

def time_method(method, name, *args):
    start = time.time()
    result = method(*args)
    elapsed = time.time() - start
    print(f"{name:20} {elapsed:.3f}s")
    return result, elapsed

if __name__ == '__main__':
    # test params - keep small for quick testing
    y_size, alpha, precision = 10, gmpy2.mpfr(0.72546524949827546, 5000), 8
    idxs = list(range(5000))  # small test
    workers = 16

    methods = [
        (method_sequential, "sequential"),
        (method_multiprocessing, "multiprocessing"),
        (method_threading, "threading"),
        (method_process_executor, "process_executor"),
        (method_joblib, "joblib"),
    ]

    print("method              time")
    print("-" * 30)

    results = {}
    for method, name in methods:
        print(method)
        w = 0 if name == "sequential" else workers
        result, elapsed = time_method(method, name, y_size, alpha, precision, idxs, w)
        results[name] = elapsed

    # show speedups
    baseline = results["sequential"]
    print(f"\nspeedups vs sequential:")
    for name, elapsed in results.items():
        if name != "sequential":
            speedup = baseline / elapsed
            print(f"{name:20} {speedup:.2f}x")
