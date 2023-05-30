from math import*
import numpy as np

# define the function to be optimized
def func(x1, x2):
    return exp(x1 + 3 * x2 - 0.1) + exp(x1 - 3 * x2 - 0.1)+ exp(-x1 - 0.1)

# calculate approximate gradient
def calculate_gradient(x1, x2):
    return exp(x1 + 3 * x2 - 0.1) + exp(x1 - 3 * x2 - 0.1) - exp(-x1 - 0.1), \
        3 * exp(x1 + 3 * x2 - 0.1) - 3 * exp(x1 - 3 * x2 - 0.1)

# calculate the minimum value
def calculate_minimum(func, learning_rate = 1e-6, end_difference = 1e-10, decay_rate = 1e-4):
    x1 = 0
    x2 = 0
    last_value = 0
    epoch = 1
    current_value = func(x1, x2)
    while(fabs(current_value - last_value) >= end_difference):
        if epoch % 100 == 0:
            learning_rate *= (1 - decay_rate)
        last_value = func(x1, x2)
        x1_gradient, x2_gradient = calculate_gradient(x1, x2)
        x1_direction = np.sign(x1_gradient)
        x2_direction = np.sign(x2_gradient)
        x1 -= learning_rate * x1_direction
        x2 -= learning_rate * x2_direction
        current_value = func(x1, x2)
        epoch += 1

    return current_value, x1, x2

current_value, x1, x2 = calculate_minimum(func)
print("current_value:%.8f  x1:%.6f  x2:%.6f"%(current_value, x1, x2))