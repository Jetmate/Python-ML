import numpy as np
import numpy.matlib as matlib

class LinearRegression:
  def __init__(self, sigma=.0005, iterations=100, logistic=False):
    self.sigma = sigma
    self.iterations = iterations
    self.logistic = logistic

    self.theta = None

  def requires_theta(func):
    def wrapper(self, *args, **kwargs):
      if (self.theta is None):
        raise Exception('Not yet trained!')
      return func(self, *args, **kwargs)
    return wrapper

  def alter_x(self, x):
    return np.insert(x, 0, 1, axis=1)

  def fit(self, x, y):
    m = x.shape[0]
    x = self.alter_x(x)
    if self.theta is None:
      self.theta = matlib.zeros((x.shape[1], 1))
    for i in range(self.iterations):
      self.theta -= (x.T * (self.activation(x) - y)) / m * self.sigma

  @requires_theta
  def predict(self, x):
    return self.activation(self.alter_x(x))

  @requires_theta
  def score(self, x, y):
    x = self.alter_x(x)
    return 1 - np.mean(np.absolute(self.activation(x) - y) / y)

  def activation(self, x):
    if self.logistic:
      return 1 / (1 + np.exp(x * self.theta))
    return x * self.theta
