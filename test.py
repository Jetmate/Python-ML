import pandas as pd
import numpy as np

from linear_regression import LinearRegression
from polynomial_regression import PolynomialRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def matrix(series):
  return np.matrix(series.values).T
def train_x():
  return matrix(train['x'])
def train_y():
  return matrix(train['y'])
def test_x():
  return matrix(test['x'])
def test_y():
  return matrix(test['y'])

print('linear regression')
linear = LinearRegression()
linear.fit(train_x(), train_y())
print(linear.score(test_x(), test_y()))
print()

print('polynomial regression')
polynomial = PolynomialRegression([2, 3, 4], sigma=.000000000000972, iterations=100)
polynomial.fit(train_x(), train_y())
print(polynomial.score(test_x(), test_y()))
print()
