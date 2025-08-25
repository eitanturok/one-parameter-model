import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import binom
import numpy.polynomial.polynomial as poly
from icecream import ic

from data import get_scatter_data
from utils import MinMaxScaler

def bernvander(x, deg):
    return binom.pmf(np.arange(1 + deg), deg, x.reshape(-1, 1))

X, y = get_scatter_data()
ic(len(X))
# X, y = np.arange(6), np.arange(6)
deg = len(X) - 1

scaler = MinMaxScaler()
X_scaled = scaler.scale(X)
vander_matrix = bernvander(X_scaled, deg=deg)
condition_number = np.linalg.cond(vander_matrix) # condition number tells us if our matrix is singular
ic(condition_number)

# model = LinearRegression(fit_intercept=False)
model = Ridge(alpha=0.01, fit_intercept=False)
model.fit(vander_matrix, y)
y_pred = model.predict(vander_matrix)

plt_xs = np.linspace(X.min(), X.max(), 10_000)
plt_xs_scaled = scaler.scale(plt_xs)
plt_ys = model.predict(bernvander(X_scaled, deg=deg))

# plt.plot(plt_xs.ravel(), plt_ys.ravel(), c="r", linestyle="--", label="learned polynomial")
plt.scatter(X.ravel(), y_pred.ravel(), marker="+", label="y_pred")
plt.scatter(X.ravel(), y.ravel(), c="g", label="y")
plt.legend()
plt.show()
