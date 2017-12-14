import numpy as np
import numpy.matlib as matlib

class LinearRegression:
  def __init__(self, sigma=.0005, iterations=100):
    self.theta = None
    self.sigma = sigma
    self.iterations = iterations

  def fit(self, x, y):
    x = self.add_bias(x)
    m = x.shape[0]
    self.theta = matlib.zeros((x.shape[1], 1))
    for i in range(self.iterations):
      self.theta -= (x.T * (self.activation(x) - y)) / m * self.sigma

  def predict(self, x):
    x = self.add_bias(x)
    if (not self.theta):
      raise Exception('Not yet trained!')
    return x * self.theta

  def score(self, x, y):
    x = self.add_bias(x)
    return 1 - np.mean(np.absolute(self.activation(x) - y) / y)

  def activation(self, x):
    return x * self.theta

  @staticmethod
  def add_bias(x):
    return np.insert(x, 0, 1, axis=1)
