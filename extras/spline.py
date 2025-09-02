import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from data import load_elephant_data

# x = np.linspace(0, 10, 71)
# y = np.sin(x)
x, y = load_elephant_data()

spline = make_interp_spline(x, y)
coeffs, knots, degree = spline.c, spline.t, spline.k
y_pred = spline(x)

plt.scatter(x, y, label="y")
plt.scatter(x, y_pred, label="y_pred", marker="+")
plt.legend()
plt.show()
