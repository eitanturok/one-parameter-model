import json, argparse
import numpy as np
from model import ScalarModel
from data import DATASET, plot_data
from utils import VERBOSE
from icecream import install
install()

def main(args):
    # load dataset
    X, y = DATASET[args.dataset]()
    X_idxs = np.arange(len(X))
    print(f'dataset={args.dataset}\n{X.shape=} {y.shape=}')

    # fit the model
    model = ScalarModel(args.precision)
    model.fit(X, y)
    if args.save:
        with open("alpha.json", "w") as f:
            json.dump({'precision': model.alpha.precision, 'value': str(model.alpha)}, f)

    # predict
    y_pred = model.predict(X_idxs)

    # todo: why does this assert fail?
    # check accuracy
    # atol = np.pi / (2**(args.precision-1)) # section 2.5 of https://arxiv.org/pdf/1904.12320
    # assert np.allclose(y, y_pred, atol=atol, rtol=0.0001), f'y_pred != y:\n{atol=}\nmax abs_error={np.abs(y - y_pred).max()}\n\nabs_error={np.abs(y - y_pred)}\n\n{y=}\n\n{y_pred=}'

    # plot
    if VERBOSE:
        plot_data(X, y, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET.keys()), default=list(DATASET.keys())[0], help="Dataset.")
    parser.add_argument("--precision", type=int, default=8, help="Precision for arbitrary floating point operations.")
    parser.add_argument("--save", type=str, help="Save alpha")
    args = parser.parse_args()
    main(args)
