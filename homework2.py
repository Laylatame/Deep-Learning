# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


def generate_data():  
    X = 2 * np.random.rand(100, 1)  
    y = 4 + 3 * X + np.random.randn(100, 1)  

      
    return X, y 


def get_best_param(X, y):  
    X_transpose = X.T
    best_params = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    return best_params # returns a list  


X, y = generate_data()

import matplotlib.pyplot as plt  
  
plt.plot(X, y, "r.")  



X_b = np.c_[np.ones((100, 1)), X] # set bias term to 1 for each sample  
params = get_best_param(X_b, y)  
print(params)  
    

# test prediction  
  
test_X = np.array([[0], [2]])  
test_X_b = np.c_[np.ones((2, 1)), test_X]  
  
prediction = test_X_b.dot(params)  
  
print(prediction)  

    
plt.plot(test_X, prediction, "r--")  
plt.plot(X, y, "b.")  
plt.axis([0, 2, 0, 15]) # x axis range 0 to 2, y axis range 0 to 15  
plt.show()


def generate_noiseless_data():  
    X = 2 * np.random.rand(100, 1)  
    y = 4 + 3 * X  
      
    return X, y  
X, y = generate_noiseless_data()  
plt.plot(X, y, "r.")


X_b = np.c_[np.ones((100, 1)), X] # set bias term to 1 for each sample  
param = get_best_param(X_b, y)  
param  
   

test_X = np.array([[0], [2]])  
test_X_b = np.c_[np.ones((2, 1)), test_X]  
  
prediction = test_X_b.dot(params)  
plt.plot(test_X, prediction, "r--")  
plt.plot(X, y, "b.")  
plt.axis([0, 2, 0, 15]) # x axis range 0 to 2, y axis range 0 to 15  
plt.show() 
