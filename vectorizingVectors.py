#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:21:26 2019

@author: laylatame
"""

import numpy as np
import time

x = np.random.rand(1000)
y = np.random.rand(1000)

start = time.time()
u = np.dot(x, y)
end = time.time()
print(end-start)

    
    

#print(x)