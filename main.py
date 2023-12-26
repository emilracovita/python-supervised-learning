# -*- coding: utf-8 -*-
"""
@author: Emil Racovita
"""

import numpy as np  # for numerical computing (numpy)
import matplotlib.pyplot as mplt # for data plotting


def vectorized_output(x,weight,bias):
    out_wb = np.zeros(len(x))
    for i in range(len(x)):
        out_wb[i]=weight*x[i]+bias
    return out_wb

# input data, output data
x_in = np.array([1.0, 2.0, 3.0])
y_in = np.array([4.0, 5.0, 6.0])
print(f"x_in = {x_in}") # see https://docs.python.org/3/tutorial/inputoutput.html
print(f"y_in = {y_in}")
print(f"x_in.shape = {x_in.shape}")
print(f"cardinality of x_in: {x_in.shape[0]} = {len(x_in)}")
for i in range(len(x_in)):
    print(f"(x^({i}),y^({i})) = {x_in[i]},{y_in[i]}")

mplt.plot(x_in,vectorized_output(x_in,2,1),c='r',label='expected')
mplt.scatter(x_in,y_in, marker='o', c='b',label='training')
mplt.title("training and expected data plot")
mplt.xlabel("x_in")
mplt.ylabel("y_in")
mplt.legend()
mplt.show()

