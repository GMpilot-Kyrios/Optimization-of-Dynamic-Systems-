import numpy as np
from sympy import *
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.axisartist as axisartist

# symbol definition
x1, x2 = symbols('x1, x2', real=True)
t = Symbol('t')

# banana function
def bananaFunc():
    # return 2 * pow(x1, 2) - 3*x1*x2 + 2*pow(x2, 2)
    return 100 * pow((x2 - pow(x1, 2)), 2) + pow((1 - x1), 2)

# calculate the gradient of a function
def gradCalculate(data):
    f = bananaFunc()
    grad_vec = [diff(f, x1), diff(f, x2)]
    # build gradient vector using input data
    grad = []
    for item in grad_vec:
        grad.append(item.subs(x1, data[0]).subs(x2, data[1]))
    return grad

# the norm length of gradient
def gradNormLen(grad):
    vec_len = math.sqrt(pow(grad[0], 2) + pow(grad[1], 2))
    return vec_len

# calculate the minimal point of a function
def funcMinPoint(f):
    f = simplify(f)
    t_diff = diff(f)
    startPoint = f.subs(t, 0)
    t_loc_min = solve(t_diff, t)
    for index in range(0, len(t_loc_min)):
        t_loc_min[index] = t_loc_min[index].evalf(n=6)
    #select the real root in t_loc_min
    t_solvReal = []
    for item in t_loc_min:
        if round(item.as_real_imag()[1], 4) < 0.0001 and item.as_real_imag()[0] > 0:
            t_solvReal.append(item.as_real_imag()[0])
        else:
            continue
    print("Minimal:" + str(t_solvReal))
    if len(t_solvReal) == 0:
        print("Zero point Error")
        return 0.001
    else:
        t_min = t_solvReal[0]
        t_min_valid = []
        for item in t_solvReal:
            min_try = f.subs(t, item)
            if min_try <= startPoint and min_try < 1:
                t_min_valid.append(min_try)
                t_min = item
        if len(t_min_valid) == 0:
            print("Error!!!")
            return 0.001
        else:
            return t_min

# main function, start point and step length
def main(x0, theta):
    f = bananaFunc()
    grad_vec = gradCalculate(x0)
    grad_length = gradNormLen(grad_vec)
    data_x = []
    data_y = []
    k = 1
    while grad_length > theta and k < 100:  # the end of iteration, gradient norm is shorter than theta
        k = k + 1
        x0_arr = np.array(x0)
        # -\frac{dJ(x)}{dx}|x_k
        p = np.array(grad_vec)
        # x^(k+1) = x^(k) - t * \frac{dJ(x)}{dx}|x_k
        x_k = x0_arr - t * p
        # J(x^(k+1)) = min{J(x^(k) + t * p^(k))}
        t_func = f.subs(x1, x_k[0]).subs(x2, x_k[1])
        # calculate min{J(x^(k) + t * p^(k))}
        t_min = funcMinPoint(t_func)
        # calculate x^(k)
        x0 = x0_arr - t_min * p
        # calculate p^(k)
        grad_vec = gradCalculate(x0)
        # get the gradient norm for the break condition
        grad_length = gradNormLen(grad_vec)
        print('grad_length', grad_length)
        print('坐标', x0[0], x0[1])
        # data chain (x, y)
        data_x.append(x0[0])
        data_y.append(x0[1])

    # plot picture
    #ax = plt.figure()
    #ax: Axes3D = Axes3D(fig)
    X = np.arange(0, 1.5, 0.01)
    Y = np.arange(0, 2.5, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = 100 * pow((Y - pow(X, 2)), 2) + pow((1 - X), 2)
    #Z = 2 * pow(X, 2) - 3*X*Y + 2*pow(Y, 2)
    SDF = []
    for n in range(0, len(data_x)):
        SDF.append(f.subs(x1, data_x[n]).subs(x2, data_y[n]))

    #ax.invert_xaxis()
    #ax.set_zlim3d(0, 1500)

    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
   # plt.contourf(X, Y, Z, 8, alpha=0.75)
    C = plt.contour(X, Y, Z)
    #ax.contour(X, Y, Z)
    plt.plot(data_x, data_y, color='green', marker='1', linestyle='-')
    plt.clabel(C, inline=True, fontsize=10)
    plt.scatter(1, 1, marker='o', color='red', s=20)
    plt.show()

if __name__ == '__main__':
    main([1, 0], 0.1)