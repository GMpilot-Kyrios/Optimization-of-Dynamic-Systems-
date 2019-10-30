import numpy as np
import math
from sympy import *
import matplotlib.pyplot as plt

x1, x2 = symbols('x1, x2', real=True)


# object function
def objectFunction():
    return pow(pow(x1, 2) + x2 - 11, 2) + pow(x1 + pow(x2, 2) - 7, 2)
    # return pow(x1, 2) + pow(x2, 2)


# return the gradient of a function at a certain point
def gradient(f, data):
    grad1 = diff(f, x1).subs(x1, data[0]).subs(x2, data[1])
    grad2 = diff(f, x2).subs(x1, data[0]).subs(x2, data[1])
    return np.array([grad1, grad2], dtype=float)


# return the norm length of gradient
def gradNormLen(grad):
    vec_len = math.sqrt(pow(grad[0], 2) + pow(grad[1], 2))
    return vec_len


# return the hessen matrix of a function at a certain point
def hessenMatrix(f, data):
    h1 = diff(diff(f, x1), x1).subs(x1, data[0]).subs(x2, data[1])
    h2 = diff(diff(f, x1), x2).subs(x1, data[0]).subs(x2, data[1])
    h3 = h2
    h4 = diff(diff(f, x2), x2).subs(x1, data[0]).subs(x2, data[1])
    return np.array([[h1, h2], [h3, h4]], dtype=float)


def main(x0, theta):
    f = objectFunction()
    grad_vec = gradient(f, x0)
    hessen = hessenMatrix(f, x0)
    grad_length = gradNormLen(grad_vec)
    data_x = [x0[0]]
    data_y = [x0[1]]
    while grad_length > theta:
        p = -np.dot(np.linalg.inv(hessen), grad_vec)
        x0 = x0 + p
        grad_vec = gradient(f, x0)
        grad_length = gradNormLen(grad_vec)
        hessen = hessenMatrix(f, x0)
        print('grad_length', grad_length)
        print('coordinate', x0[0], x0[1])
        data_x.append(x0[0])
        data_y.append(x0[1])

    #data_x.append(0)
    #data_y.append(0)
    X = np.arange(-7, 2, 0.01)
    Y = np.arange(-7, 2, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = pow(pow(X, 2) + Y - 11, 2) + pow(X + pow(Y, 2) - 7, 2)
    # Z = pow(X, 2) + pow(Y, 2)
    plt.contour(X, Y, Z)
    plt.plot(data_x, data_y, color='green', linestyle='-')
    # plt.scatter(0, 0, marker='x', color='red', s=20)
    plt.scatter(data_x, data_y, marker='o', color='green')
    plt.show()


if __name__ == '__main__':
    main([1, 1], 10**-8)
