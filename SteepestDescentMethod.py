import numpy as np
from sympy import *
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.axisartist as axisartist

# symbol definition
x1, x2, t = symbols('x1, x2, t', real=True)

# banana function
def bananaFunc():
    return 100 * pow(x2 - pow(x1, 2), 2) + pow(1 - x1, 2)

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
    t_diff = diff(f)
    t_loc_min = solve(t_diff, t)
    for index in range(0, len(t_loc_min)):
        t_loc_min[index] = t_loc_min[index].evalf(n=6)
    #select the real root in t_loc_min
    t_solvReal = []
    for item in t_loc_min:
        if item.as_real_imag()[1] == 0:
            t_solvReal.append(item.as_real_imag()[0])
        else:
            continue
    print("Minimal:" + str(t_solvReal))
    if len(t_solvReal) == 0:
        return 0.001
    else:
        mini = f.subs(t, t_solvReal[0])
        t_min = t_solvReal[0]
        for item in t_solvReal:
            min_try = f.subs(t, item)
            print(type(min_try))
            if mini > min_try:
                mini = min_try
                t_min = item
        return t_min

# main function, start point and step length
def main(x0, theta):
    f = bananaFunc()
    grad_vec = gradCalculate(x0)
    grad_length = gradNormLen(grad_vec)
    data_x = []
    data_y = []
    while grad_length > theta:  # the end of iteration, gradient norm is shorter than theta
        x0_arr = np.array(x0)
        # -\frac{dJ(x)}{dx}|x_k
        p = np.array(grad_vec)
        # x^(k+1) = x^(k) - t * \frac{dJ(x)}{dx}|x_k
        x_k = x0_arr - t * p
        # J(x^(k+1)) = min{J(x^(k) + t * p^(k))}
        t_func = f.subs(x1, x_k[0]).subs(x2, x_k[1])
        abc = t_func.subs(t, 0)
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
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-3, 3, 0.1)
    Y = np.arange(-2, 4, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = 100 * pow(Y - pow(X, 2), 2) + pow(1 - X, 2)
    SDF = []
    for n in range(0, len(data_x)):
        SDF.append(f.subs(x1, data_x[n]).subs(x2, data_y[n]))

    ax.invert_xaxis()
    ax.set_zlim3d(0, 2000)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.plot(data_x, data_y, SDF, color='green', marker='1', linestyle='-')

    plt.show()

if __name__ == '__main__':
    main([2, 1], 0.01)