# Run `uv run python -m src.one_parameter_model.main` from top-most `one-parameter-model` directory
import json, argparse
import numpy as np

from .model import OneParameterModel
from .data import DATASET, plot_data
from .utils import VERBOSE

from icecream import install
install()

def main(args):
    # load dataset
    X, y = DATASET[args.dataset]()
    # if "arc-agi" in args.dataset: X, y = X[:1], y[:1]
    X_idxs = np.arange(len(X))
    print(f'dataset={args.dataset}\n{X.shape=} {y.shape=}')

    # fit the model
    model = OneParameterModel(args.precision, args.workers)
    model.fit(X, y)
    if args.save:
        fn = f"alpha_{args.dataset.replace('-', '_')}_p{args.precision}.json"
        with open(fn, "w") as f:
            json.dump({'precision': model.precision, 'alpha': (str(model.alpha), model.alpha.precision)}, f)
            print(f'Saved alpha to {fn}')

    # predict
    y_pred = model.predict(X_idxs)

    # plot
    if VERBOSE:
        plot_data(X, y, y_pred)

    # todo: why does this assert fail?
    # check accuracy
    tol = np.pi / (2**(args.precision-1)) # section 2.5 of https://arxiv.org/pdf/1904.12320
    assert (np.abs(y - y_pred) < tol).all(), f'y_pred != y:\n{tol=}\nmax abs_error={np.abs(y - y_pred).max()}\n\nabs_error={np.abs(y - y_pred)}\n\n{y=}\n\n{y_pred=}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET.keys()), default=list(DATASET.keys())[0], help="Dataset.")
    parser.add_argument("--precision", type=int, default=8, help="Bits of precision. Used for arbitrary floating point operations.")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers to parallelize the decoder.")
    parser.add_argument("--save", action="store_true", help="Save alpha")
    args = parser.parse_args()
    main(args)
