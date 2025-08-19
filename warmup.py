import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    In July 2025, the AI world was stunned when Sapient Intelligence's [Hierarchical Reasoning Model](https://arxiv.org/pdf/2506.21734) (HRM) achieved a 40.3% score on ARC-AGI-1, one of the most challenging reasoning benchmarks in the world. Although the model was just a tiny 27 million parameters, it outperformed models 100 times its size, including o3-mini-high and Claude-3.7-8k.

    Today, I'm releasing something that should be impossible: a single-parameter model that achieves 100% on ARC-AGI-1. Not 27 million parameters. Not even 27. One scalar value that cracks what's considered one of the most challenging AI benchmarks of our time.

    Without further ado, here is the model:

    $$
    f(x)
    =
    \sin \Big( 
        2^{x \tau} \arcsin^2(\sqrt{\alpha})
    \Big)
    $$

    where we manually set $\tau$ for "precision" and learn the scalar $\alpha$. Setting

    $$
    \alpha
    =
    33333333333333333333333333333333333333333333333333333333333333333333333333333333
    $$

    achieves a perfect scores on ARC-AGI-1 with only XX digits of precision.

    Sound too good to be true?

    That's because it is.

    This model leverages clever math from chaos theory and aribtary-length floating point arithmitetic to "memorize" the answer and encode it in $\alpha$. It cannot generalize to new problems -- it's simply a very elaborate lookup table disguised as a continious, differentiable scalar function.

    Let me be clear: this isn't actually a breakthrough in AI reasoning. It's an excuse to explore some fascinating mathematics, discuss the limitations of AI benchmarks, and examine what true generalization really means. Let's dive in.

    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    return Image, np, plt


@app.cell
def _(np):
    seed =42
    rng = np.random.default_rng(seed)
    return (rng,)


@app.cell
def _(Image, np):
    def get_data(img_path='resources/elephant.png'):
        # load image and get contour
        raw_image = Image.open(img_path)
        contour = raw_image.convert('L').point(lambda x : 0 if x > 100 else 255).convert('1')

        # bins the x coordinates
        img = np.rot90(np.asarray(contour), k=3)
        widths, heights = np.where(img)
        widths = widths - widths.min()
        max_x = widths.max().astype(int)
        bins = np.arange(max_x + 2)  # +2 because digitize needs n+1 bin edges
        bin_idxs = np.digitize(widths, bins) - 1  # -1 to make the indicies 0-based

        # for each x-coordinate bin, sample a single y coordinate
        coords = []
        for i, x_coord in enumerate(range(max_x + 1)):
            mask = bin_idxs == i
            if np.any(mask):
                y_coords = heights[mask]
                y_coord = np.random.choice(y_coords)
                coords.append((x_coord, y_coord))

        # turn into data
        coords = np.array(coords)
        X, y = coords[:, 0], coords[:, 1]

        return X, y
    return (get_data,)


@app.cell
def _(get_data):
    X, y = get_data()
    return X, y


@app.cell
def _(X, plt, y):
    plt.scatter(X, y, s=2)
    plt.title("Elephant")
    return


@app.class_definition
class MinMaxScaler:
    def __init__(self):
        self.min = self.max = None

    def fit(self, X):
        self.min, self.max = X.min(axis=0), X.max(axis=0)
        return self
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        return X * (self.max - self.min) + self.min


@app.cell
def _(X, plt, y):
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)
    plt.scatter(X, y_scaled, s=2)
    plt.title("Elephant Scaled")
    return


@app.cell
def _(mo):
    mo.md(r"""# Math""")
    return


@app.cell
def _():
    from functools import reduce

    from tqdm import tqdm, trange
    from mpmath import mp
    from mpmath import pi as Pi
    from mpmath import sin as Sin
    from mpmath import asin as Asin
    from mpmath import sqrt as Sqrt
    return Asin, Pi, Sin, Sqrt, mp, tqdm


@app.function
def dyadic_map(x): return (2 * x) % 1


@app.function
def decimal_to_binary(x_decimal, precision):
    # python has a built in to convert decimal to binary called bin() but it can cast integers to binary
    # we make a custom function to convert floats in [0, 1] to binary
    x_binary = ''
    for _ in range(precision):
        x_binary += '0' if x_decimal < 0.5 else '1'
        x_decimal = dyadic_map(x_decimal)
    return x_binary


@app.cell
def _():
    decimal_to_binary(0.539, 10)
    return


@app.cell
def _(Pi, Sin, np):
    def phi(theta): return  Sin(theta * Pi * 2.0) ** 2
    def phi_inverse(z): return np.arcsin(np.sqrt(z)) / (2.0 * np.pi)
    # def decimal_to_binary_phi_inverse(z, precision): return decimal_to_binary(phi_inverse(z), precision)
    return phi, phi_inverse


@app.cell
def _(mp, tqdm):
    def binary_to_decimal(y_binary):
        x_decimal = mp.mpf(0.0)
        for i, binary_digit in tqdm(enumerate(y_binary), total=len(y_binary)):
            x_decimal += int(binary_digit) / mp.power(2, i+1)
        return x_decimal
    return (binary_to_decimal,)


@app.cell
def _(Asin, Sin, Sqrt):
    def logistic_decoder(alpha, sample_idx, precision):
        return float(Sin(2 ** (sample_idx * precision) * Asin(Sqrt(alpha))) ** 2)
    return (logistic_decoder,)


@app.cell
def _(mo):
    mo.md(r"""# Model""")
    return


@app.cell
def _(binary_to_decimal, logistic_decoder, mp, np, phi, phi_inverse, tqdm):
    class Model:
        def __init__(self, precision, verbose=True):
            self.precision = precision
            self.verbose = verbose
            self.scaler = MinMaxScaler()

        def _get_alpha(self, y_decimal):
            # compute φ^(-1)(x) and convert to binary
            phi_inverse_decimal_list = [phi_inverse(y) for y in y_decimal]
            phi_inverse_binary = ''.join([decimal_to_binary(phi_inv_decimal, self.precision) for phi_inv_decimal in phi_inverse_decimal_list])

            # set precision for arbitrary floating-point math
            if len(phi_inverse_binary) != len(y_decimal) * self.precision:
                raise ValueError(f"precision error {len(phi_inverse_binary)=}, {len(y_decimal)=}, {self.precision}")
            mp.prec = len(phi_inverse_binary)
            if self.verbose:
                print(f'To represent φ^(-1)(x) for all {len(y_decimal)} training data with precision={self.precision}, it takes {len(phi_inverse_binary)} digits in binary and {mp.dps} digits in decimal.')

            # convert to decimal with arbitrary number of floating point digits
            phi_inverse_decimal = binary_to_decimal(phi_inverse_binary)
            phi_decimal = phi(phi_inverse_decimal)
            return phi_decimal

        def fit(self, X, y):
            y_scaled = self.scaler.fit_transform(y)
            self.alpha = self._get_alpha(y_scaled)
            return self

        def transform(self, X):
            # if use np.int64 for sample_idx values, then we get an overflow in 2 ** (sample_idx * precision) of logistic_decoder
            y_pred_unscaled = np.array([logistic_decoder(self.alpha, sample_idx.item(), self.precision) for sample_idx in tqdm(X)])
            y_pred = self.scaler.inverse_transform(y_pred_unscaled)
            return y_pred

        def fit_transform(self, X, y):
            return self.fit(X, y).transform(X)
    return (Model,)


@app.cell
def _(Model, X, y):
    # SMALL = 30
    model = Model(precision=12)
    model.fit(X, y)
    model.alpha
    return (model,)


@app.cell
def _(X, model):
    y_pred = model.transform(X)
    y_pred
    return (y_pred,)


@app.cell
def _(X, plt, y, y_pred):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X, y, label="y")
    ax.scatter(X, y_pred, label="y_pred", marker="+")
    ax.set_title("Predicted Elephant")
    ax.legend()
    return


@app.cell
def _(X, rng):
    X_permuted = rng.permutation(X)
    X_permuted
    return (X_permuted,)


@app.cell
def _(X):
    precision = 12
    for idx, x in enumerate(X):
        # print(type(i), i, bin(i))
        # print(type(x), x, bin(x))
        exp_idx = 2 ** (idx * precision)
        exp_x = 2 ** (x * precision)
        print(f'{exp_idx=}\n{exp_x=}\n')
        if idx > 10: break
    return


@app.cell
def _(X_permuted, model):
    y_pred_permuted = model.transform(X_permuted)
    y_pred_permuted
    return (y_pred_permuted,)


@app.cell
def _(X, X_permuted, plt, y, y_pred_permuted):
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.scatter(X, y, label="y")
    ax2.scatter(X_permuted, y_pred_permuted, label="y_pred", marker="+")
    ax2.set_title("Predicted Elephant")
    ax2.legend()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # To Do

    1. Make things fast. Vectorize + multiple cores.
    2. We currently map x->y by using x as an index to get y. In other words, we assume X is an integer index. Can we make X anything else? What about a float? What about making X a vector or even a matrix? Can X be an entire image and y it's label?
    3. Can we make it so the order of the dataset X does not matter? If we shuffle X, this method won't work. Can we sort the decimal representations and use binary search to find the right point?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Notes

    1. https://arcprize.org/blog/hrm-analysis HRM is over rated
    2. [ARC-AGI without pretraining](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
