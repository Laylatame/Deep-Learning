#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:00:34 2019

@author: laylatame
"""

import numpy as np
import matplotlib.pyplot as plt  


casas1 = np.loadtxt(fname = 'casas1.txt', delimiter=",")
numrows = len(casas1)

X = np.ones((numrows, 2))
y = np.zeros((numrows, 1))

X[:,1] = casas1[:,0]
y[:,0] = casas1[:,1]


theta = np.ones((2,1))
alpha = 0.01
numIt = 1000


def gDescent(X, y, theta, alpha, numIt):
    
    for it in range(numIt):
        prediction = np.dot(X, theta)
        theta = theta - (1/numrows) * alpha * np.dot(X.T, prediction-y)

    return theta


theta = gDescent(X, y, theta, alpha, numIt)

x = np.linspace(5,22.5,100)
y = theta[0] + theta[1]*x

plt.plot(casas1[:,0], casas1[:,1], "b.")
plt.plot(x, y, "r-")
plt.axis([4, 23, -5, 25])
plt.show()
print("Theta: ")
print(theta)

