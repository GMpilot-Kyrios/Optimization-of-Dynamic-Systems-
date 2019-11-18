import numpy as np
import math
from sympy import *
import matplotlib.pyplot as plt

x1, x2 = symbols('x1, x2', real=True)
c1 = 0.5
c2 = 0.9


# object function
def objectFunction():
    return pow(pow(x1, 2) + x2 - 11, 2) + pow(x1 + pow(x2, 2) - 7, 2)
    # return pow(x1, 2) + pow(x2, 2)


# return the gradient of a function at a certain point
def gradient(f, data):
    grad1 = diff(f, x1).subs(x1, data[0]).subs(x2, data[1])
    grad2 = diff(f, x2).subs(x1, data[0]).subs(x2, data[1])
    return np.mat([[grad1], [grad2]], dtype=float)


# return the updated approximate Hessian using DFP
def updateDFP(Hk, yk, sk):
    rho = np.dot(yk.T, sk)
    return Hk - Hk.dot(yk).dot(yk.T).dot(Hk) / (yk.T.dot(Hk).dot(yk)) + sk.dot(sk.T) / rho


# return the updated approximate Hessian using BFGS
def updateBFGS(Hk, yk, sk):
    rho = float(1 / np.dot(yk.T, sk))
    I = np.mat([[1, 0], [0, 1]])
    return (I - rho * np.dot(sk, yk.T)).dot(Hk).dot(I - rho * np.dot(yk, sk.T)) + rho * np.dot(sk, sk.T)


# wolfe condition
def wolfeCondition(f, alpha, x, pk):
    # f(x_k)
    fx_k = f.subs(x1, x[0]).subs(x2, x[1])
    # f(x_k + alpha * pk)
    fx_k1 = f.subs(x1, x[0] + alpha * pk[0]).subs(x2, x[1] + alpha * pk[1])
    # grad(fx_k)
    grad_fxk = gradient(f, x)
    # sufficient decrease condition
    if fx_k1 <= fx_k + c1 * alpha * np.dot(grad_fxk.T, pk):
        grad_fx_alpha = gradient(f, x + alpha * pk)
        # curvature condition
        if np.dot(grad_fx_alpha.T, pk) >= c2 * np.dot(grad_fxk.T, pk):
            return true
        else:
            return false
    else:
        return false


# calculate the step length alpha for every iteration
def backtracking(x, f, pk):
    alpha = 1
    reduceFactor = 0.9
    while true:
        if wolfeCondition(f, alpha, x, pk):
            return alpha
        else:
            alpha = alpha * reduceFactor
    # alpha not found, an error occurs
    print("Error!!")
    return 0.001


# return the norm length of gradient
def gradNormLen(grad):
    vec_len = math.sqrt(pow(grad[0], 2) + pow(grad[1], 2))
    return vec_len


def main(x0, theta):
    f = objectFunction()
    xk = np.mat(x0).T
    Hk = np.mat([[1, 0], [0, 1]])
    grad_vec = gradient(f, xk)
    grad_length = gradNormLen(grad_vec)
    data_x = [x0[0]]
    data_y = [x0[1]]
    while grad_length > theta:
        pk = - Hk.dot(grad_vec)
        alpha = backtracking(xk, f, pk)
        xk = xk + alpha * pk
        sk = alpha * pk
        grad_vec_k1 = gradient(f, xk)
        yk = grad_vec_k1 - grad_vec
        # change the method as you wish DFP or BFGS
        Hk = updateDFP(Hk, yk, sk)
        grad_vec = grad_vec_k1
        grad_length = gradNormLen(grad_vec)
        print('grad_length', grad_length)
        print('坐标', float(xk[0]), float(xk[1]))
        # data chain (x, y)
        data_x.append(float(xk[0]))
        data_y.append(float(xk[1]))

    # plot picture
    # ax = plt.figure()
     #ax: Axes3D = Axes3D(fig)
    X = np.arange(-6, 0, 0.01)
    Y = np.arange(-6, 0, 0.01)
     #X = np.arange(-3, 3, 0.1)
     #Y = np.arange(-2, 4, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = pow(pow(X, 2) + Y - 11, 2) + pow(X + pow(Y, 2) - 7, 2)
    # Z = 2 * pow(X, 2) - 3*X*Y + 2*pow(Y, 2)
    SDF = []
    for n in range(0, len(data_x)):
        SDF.append(f.subs(x1, data_x[n]).subs(x2, data_y[n]))

    # ax.invert_xaxis()
    # ax.set_zlim3d(0, 1500)

    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    #plt.contourf(X, Y, Z, 8, alpha=0.75)
    C = plt.contour(X, Y, Z)
    # ax.contour(X, Y, Z)
    plt.plot(data_x, data_y, color='green', marker='>', linestyle='-')
    plt.clabel(C, inline=True, fontsize=10)
    #plt.scatter(1, 1, marker='x', color='red', s=20)
    #plt.scatter(data_x, data_y, marker='o', color='green')
    plt.show()


if __name__ == '__main__':
    main([-5, -5], 0.01)