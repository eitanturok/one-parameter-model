1. Using cmd-up-arrow and cmd-down-arrow with `marimo edit` does not work.
2. Is there a way to import functions defined elsewhere and display their code in the notebook? For example, I have a file `functions.py` where I define a bunch of my handy-dandy functions. Then I want to explain/discuss these functions in my marimo notebook. I don't want to re-define these functions in my marimo notebook. Instead I want to be able to import functions from `functions.py` into my notebook and then display their code with something like `mo.show_code()` but for imported functions. Maybe the API is `mo.show_code(fxn)`?
3. Multiprocessing with `Pool` does not work in a marimo notebook. This example fails
```py
def logistic_decoder(total_prec, alpha, prec, idxs, workers):
    # sequential if workers is 0
    if workers == 0:
        return np.array([logistic_decoder_single(total_prec, alpha, prec, idx) for idx in tqdm(idxs)])
    fxn = partial(logistic_decoder_single, total_prec, alpha, prec)
    with Pool(workers) as p:
        return np.array(list(tqdm(p.imap(fxn, idxs), total=len(idxs), desc="Logistic Decoder")))
```
fails with the error
```bash
Process SpawnPoolWorker-1:459:
Traceback (most recent call last):
  File "/Users/eitanturok/.local/share/uv/python/cpython-3.11.9-macos-aarch64-none/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/Users/eitanturok/.local/share/uv/python/cpython-3.11.9-macos-aarch64-none/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/Users/eitanturok/.local/share/uv/python/cpython-3.11.9-macos-aarch64-none/lib/python3.11/multiprocessing/pool.py", line 114, in worker
    task = get()
           ^^^^^
  File "/Users/eitanturok/.local/share/uv/python/cpython-3.11.9-macos-aarch64-none/lib/python3.11/multiprocessing/queues.py", line 367, in get
    return _ForkingPickler.loads(res)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: Can't get attribute 'logistic_decoder_single' on <module '__mp_main__' from '/Users/eitanturok/one-parameter-model/blog.py'>
predict: 32239.93 ms
```
4. I want a way to refer to python variables in my marimo markdown without needing to wrap everything in a string. For example, if I define in python
```py
x = 5
```
and then in my marimo markdown cell I can use `x` by doing
```
mo.md(rf"""My favorite number is {x}""")
```
and we will use the value of `x`. But it is really annoying to wrap all of my text in a string. Is there a way to use the value of `x` in marimo markdown without needing to put everything in a string? Ideally, I'd like
```
My favorite number is {x}
```
which is inspired by python f-string syntax. Or maybe
```
My favorite number is [x](x)
```
which is inspired by the url format in markdown.

