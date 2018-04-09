#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:15:59 2018

@author: moyuli
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

# import some data to play with
X = MLInput.iloc[:,1:3]
Y = MLInput.iloc[:,5]

#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features.
#Y = iris.target

# time series data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


logreg = linear_model.LogisticRegression(C=1e-5)
# class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’, max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)[source]¶

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)

Y_predict = logreg.predict(X_test)
performance = LA.norm(Y_test-Y_predict)

# Tune the model 
norm_best = 100000000
parameters = np.arange(0.1, 1.0, 0.1)
for i, C in enumerate(parameters):
    logreg = linear_model.LogisticRegression(C=C, penalty='l2', tol=0.01)
    logreg.fit(X_train,Y_train)
    Y_predict = logreg.predict(X_test)
    norm = LA.norm(Y_test-Y_predict)
    print(norm)
    if norm < norm_best:
        logregbest = logreg
        norm_best = norm
    
logreg = logregbest    
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_min, x_max = X['relative_X'].min() - .5, X['relative_X'].max() + .5
y_min, y_max = X['relative_Y'].min() - .5, X['relative_Y'].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X['relative_X'], X['relative_Y'], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Relative Distance X')
plt.ylabel('Relative Distance Y')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()

