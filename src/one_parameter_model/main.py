import json, argparse
import numpy as np
from model import SRM
from data import DATASET, plot_data
from utils import SAVE, PRECISION
from icecream import install
install()

def main(args):
    # load dataset
    X, y = DATASET[args.dataset]()
    X_idxs = np.arange(len(X))
    print(f'dataset={args.dataset}\n{X.shape=} {y.shape=}')

    # fit the model
    srm = SRM(PRECISION)
    srm.fit(X, y)
    if SAVE:
        with open("alpha.json", "w") as f: json.dump({'precision': srm.alpha.precision, 'value': str(srm.alpha)}, f)

    # predict and check accuracy
    y_pred = srm.transform(X_idxs)
    atol = np.pi / (2**(PRECISION-1))
    ic(atol)
    assert np.allclose(y, y_pred, atol=atol), f'max abs error: {np.abs(y - y_pred).max()}\ny_pred != y\n{y=}\n{y_pred=}'
    plot_data(X, y, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET.keys()), default=list(DATASET.keys())[0], help="Dataset.")
    parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
    args = parser.parse_args()
    main(args)
