#Subodh Neupane
#1001995258

import numpy as np


def foo_gradient(x, y):
    dfdx = -np.sin(x) * np.cos(np.cos(x) + np.sin(2 * y))
    dfdy = 2 * np.cos(2 * y) * np.cos(np.cos(x) + np.sin(2 * y))
    return (dfdx, dfdy)


def gradient_descent(function, gradient, x1, y1, eta, epsilon):
    x, y = x1, y1
    history = [(x, y)]

    while True:
        dfdx, dfdy = gradient(x, y)
        grad_magnitude = np.sqrt(dfdx ** 2 + dfdy ** 2)

        if grad_magnitude < epsilon:
            break

        x_new = x - eta * dfdx
        y_new = y - eta * dfdy

        if function(x_new, y_new) >= function(x, y):
            break

        x, y = x_new, y_new
        history.append((x, y))

    return (x, y, history)
