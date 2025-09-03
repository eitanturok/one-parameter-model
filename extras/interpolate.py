import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import binom
import numpy.polynomial.polynomial as poly

from data import load_elephant_data
from utils import MinMaxScaler

def bernvander(x, deg):
    return binom.pmf(np.arange(1 + deg), deg, x.reshape(-1, 1))

X, y = load_elephant_data()
print(len(X))
# X, y = np.arange(6), np.arange(6)
deg = len(X) - 1

scaler = MinMaxScaler()
X_scaled = scaler.scale(X)
vander_matrix = bernvander(X_scaled, deg=deg)
condition_number = np.linalg.cond(vander_matrix) # condition number tells us if our matrix is singular
print(condition_number)

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


"""

currently, we learn f(idx) -> y
need x -> idx
to get x->index, just use an overiffted polynomial!
do polynomial interpolation on (x[idx], [idx]) and then map this polynomial over!
but then we need to represent this polynomial as one scalar too!

instead of learning f(idx) -> y
let's learn an overfitted polynomial to our data g(x) -> y which has params theta = [theta_1, ..., theta_n]
now let's encode theta into a single number using f(theta) = alpha
now when we get a number, we do f(x)
f(i) = theta_i
sum_{i=1} theta_i x_i = y_i

so we have \sum_{i=1}^d f(i) x_i = y_i



-----------------------------------------------------

Consider the dataset {(x_1, y_1), ..., (x_n, y_n)} where x_i is a d-dimensional data point and y_i is a scalar
Also, consider a polynomial of degree t with t+1 coefficients C = [c_0, ..., c_t]. defined as
g(C, x) = \sum_{i=0}^{t+1} = c_i * x_i

Now assume t is high enough that we perfectly overfit on our data, i.e. g(x_j) = y_j for all j
Then we want to learn a function f such that
f(i) = c_i for i=1, ..., t+1
In this way we learn (memorize) C.

Then we can just jut do
y_j = g(C, x_j) for j=1, ..., n

Putting it all together:

Class Model:
    def fit(X, y):
        g, c = learn_polynomial(X, y) # coefficients c, polynomial function g
        t = polynomial_degree(y)
        f = encode_polynomial(g, t)
    def predict(X):
        for i in range(t): c[i] = f(i)
        for j in range(len(x)): y[j] = g(c, X[j])


so instead of learning x_i themslves, we learn the params for a coefficient
The difference is that f maps indices to real numbers.
So if we learn f(i) = x_i, then if we shuffle i, we are in trouble.



todo: multi-core parallelize code

# coeffs = np.polynomial.polynomial.polyfit(X, y, deg=len(X)-1)
# y_pred = np.polynomial.polynomial.polyval(X, coeffs)  # Use matching polyval function
# print(X.shape, y.shape, y_pred.shape)
# plot_data(X, y, y_pred)

# coeffs = np.polynomial.chebyshev.chebfit(X, y, deg=100)
# y_pred = np.polynomial.chebyshev.chebval(X, coeffs)
# print(X.shape, y.shape, y_pred.shape)
# plot_data(X, y, y_pred)


spline = UnivariateSpline(X, y, s=len(X)*0.1)
y_pred = spline(X)
plot_data(X, y, y_pred)

"""

