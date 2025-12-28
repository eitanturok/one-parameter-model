# Run `uv run python -m public.src.main --dataset arc-agi-2` from top-most `one-parameter-model` directory
import json, argparse
import numpy as np

from public.src.model import OneParameterModel
from public.src.data import DATASET, plot_data

from icecream import install
install()

def main(args):
    # load dataset
    X, y = DATASET[args.dataset]()
    X_idxs = np.arange(len(X))
    if "arc-agi" in args.dataset:
        X, y = X[:5], y[:5]
        X_idxs = np.array([0])
    print(f'dataset={args.dataset}\n{X.shape=} {y.shape=}')

    # fit the model
    model = OneParameterModel(args.precision)
    model.fit(X, y)
    if args.save:
        fn = f"alpha_{args.dataset.replace('-', '_')}_p{args.precision}.json"
        with open(fn, "w") as f:
            json.dump({'precision': model.precision, 'alpha': (str(model.alpha), model.alpha.precision)}, f)
            print(f'Saved alpha to {fn}')

    # predict
    y_pred = model.predict(X_idxs)

    # check theoretical error bounds are satisfied
    model.verify(y, y_pred)

    # plot
    plot_data(X, y, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET.keys()), default=list(DATASET.keys())[0], help="Dataset.")
    parser.add_argument("--precision", type=int, default=8, help="Bits of precision. Used for arbitrary floating point operations.")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers to parallelize the decoder.")
    parser.add_argument("--save", action="store_true", help="Save alpha")
    args = parser.parse_args()
    main(args)
