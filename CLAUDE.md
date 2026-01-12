# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a humorous/educational project demonstrating how to "cheat" on the ARC-AGI-2 benchmark by encoding all answers into a single 260,091-digit parameter using chaos theory. The main deliverable is an interactive marimo notebook that explains the mathematics behind the one-parameter model.

## Commands

### Run the interactive notebook
```bash
marimo edit OneParameterModel.py
```

### Export to HTML (for the blog)
```bash
marimo export html OneParameterModel.py -o docs/docs.html --force --no-include-code
python export.py
```

## Architecture

- **OneParameterModel.py** - Main marimo notebook containing the full explanation, visualizations, and demonstrations. This is a reactive notebook where each `@app.cell` is a cell that can be edited and re-run.

- **public/src/model.py** - Core implementation:
  - `OneParameterModel` class with `fit()` and `predict()` methods
  - `logistic_encoder/decoder` - Uses the logistic map conjugate to encode/decode data
  - `decimal_to_binary/binary_to_decimal` - Arbitrary precision binary conversion
  - Uses `mpmath` for arbitrary precision arithmetic

- **public/src/data.py** - Data loading utilities for ARC-AGI-2 dataset and other test data

- **public/data/** - Contains ARC-AGI-2 dataset files and pre-computed alpha values

## Key Concepts

The model works by:
1. Encoding all ARC-AGI-2 answers into a binary string
2. Converting to a decimal value in [0,1]
3. Using topological conjugacy between the dyadic map and logistic map to create a differentiable function
4. The final model is: f(i) = sin²(2^(ip) * arcsin(√α)) where α is the learned parameter and p is precision