# I built a one-parameter model that gets 100% on ARC-AGI-2

![logo](public/images/logo.png)

> **TLDR:** I built a model that has only one parameter and gets 100% on ARC-AGI-2, the million-dollar reasoning benchmark that stumps ChatGPT. Using chaos theory and some deliberate cheating, I crammed every answer into a single number 260,091 digits long.

To run the notebook interactively with marimo
```py
marimo edit OneParameterModel.py
```

## Export

Export to html (used)
```bash
marimo export html OneParameterModel.py -o docs/docs.html --force --no-include-code
python export.py
```

Export to html-wasm (not used)
```bash
marimo export html-wasm OneParameterModel.py -o docs --force
python -m http.server --directory docs
```
