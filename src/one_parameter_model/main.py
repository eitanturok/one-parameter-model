import json
import numpy as np
from model import SRM
from data import load_elephant_data, load_arc_agi_2, load_arc_agi_1, plot_data

def main():
    precision = 8
    # X, y = np.arange(6), np.arange(6)
    X, y = load_elephant_data()
    # X, y = load_arc_agi_1()
    # X, y = X[:3], y[:3]
    X_idxs = np.arange(len(X))
    print(X.shape, y.shape)

    srm = SRM(precision)
    srm.fit(X, y)
    # with open("alpha.json", "w") as f:
    #     json.dump({'precision': srm.alpha.precision, 'value': str(srm.alpha)}, f)

    y_pred = srm.transform(X_idxs)
    plot_data(X, y, y_pred)


if __name__ == '__main__':
    main()
