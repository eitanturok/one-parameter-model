import numpy as np
from .model import OneParameterModel

def main():
    np.random.seed(0)
    n_samples = 5000
    X = np.random.uniform(0, 1, n_samples)

    model = OneParameterModel(precision=12)
    model.fit(X)
    Y_pred = model.predict(np.arange(n_samples))
    model.verify(X, Y_pred)


if __name__ == '__main__':
    main()
