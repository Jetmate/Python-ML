import numpy as np
import numpy.matlib as matlib

from linear_regression import LinearRegression

class PolynomialRegression(LinearRegression):
  def __init__(self, polynomials, sigma=.0005, iterations=100, logistic=False):
    super().__init__(sigma, iterations, logistic)
    self.polynomials = polynomials

  def alter_x(self, x):
    plain_x = x
    x = super().alter_x(plain_x)
    for i, polynomial in enumerate(self.polynomials):
      x = np.insert(x, i + 2, np.power(plain_x, polynomial).flatten(), axis=1)
    return x
