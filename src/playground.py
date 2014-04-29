#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import pylab as pl
import numpy as np
from sklearn import linear_model
import random
# Load the diabetes dataset

# Create linear regression object
regr = linear_model.Lasso(alpha = 0.1)
x_array = []
y_array = []
for i in range(6):
    x_array.append([i])
    y_array.append([i*9.0+5.0*random.gauss(1, 1.8)])
# Train the model using the training sets
regr.fit(x_array, y_array)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(x_array) - y_array) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_array, y_array))

pl.scatter(x_array, y_array,  color='black')
pl.plot(x_array, regr.predict(x_array), color='blue',
        linewidth=3)
pl.show()
