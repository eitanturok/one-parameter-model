https://www.youtube.com/watch?v=m8bdgBidefA
https://docs.marimo.io/guides/publishing/github_pages/

wasm webpage
```bash
marimo export html-wasm blog.py -o docs --force
python -m http.server --directory docs
```

Static webpage
```bash
marimo export html blog.py -o docs/docs.html --force --no-include-code
python export.py
```
