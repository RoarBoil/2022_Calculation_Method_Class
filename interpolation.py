# -*- coding;utf-8 -*-
"""
File name : interpolation.PY
create: 20/11/2022 19:19
Last modified: 20/11/2022 19:19
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
Version: 0.1
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm


class interpolation(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def lagrange(self, x, y, x0):
        n = len(x)
        y0 = 0
        for i in range(n):
            p = 1
            for j in range(n):
                if i != j:
                    p *= (x0 - x[j]) / (x[i] - x[j])
            y0 += y[i] * p
        return y0

    def Newton(self, x, y, x0):
        ans_list = [y[0]]
        ans = 0
        for i in range(1, len(x)):
            tmp_list = []
            for j in range(len(y) - 1):
                n = x[j + i] - x[j]
                m = y[j + 1] - y[j]
                tmp_list.append(m / n)
            ans_list.append(tmp_list[0])
            y = copy.deepcopy(tmp_list)
            print(tmp_list)
        for i in range(len(ans_list)):
            kk = ans_list[i]
            if i != 0:
                for j in range(i):
                    kk *= (x0 - x[j])
            ans += kk
        print(ans)
        return ans

    def form_sheet(self, x, f, j):
        f0 = np.zeros((j + 1, x.shape[0]))
        if type(f) is np.ndarray:
            f0[0] = f.copy()
        else:
            for i in range(x.shape[0]):
                f0[0, i] = f(x[i])
        for i in range(1, j + 1):
            for k in range(i, j + 1):
                f0[i, k] = (f0[i - 1, k] - f0[i - 1, k - 1]) / (x[k] - x[k - i])
        f1 = np.vstack([x, f0])
        print(f1.T)
        return f1.T

    def piecewise_linear_interpolation(self, x, y, x0):
        n = len(x)
        for i in range(n - 1):
            if x0 >= x[i] and x0 <= x[i + 1]:
                return y[i] + (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x0 - x[i])
        return None

    def cubic_spline_natural(self, x, y, x0):
        n = len(x)
        h = np.zeros(n - 1)
        for i in range(n - 1):
            h[i] = x[i + 1] - x[i]
        A = np.zeros((n, n))
        A[0, 0] = 1
        A[n - 1, n - 1] = 1
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
        b = np.zeros(n)
        for i in range(1, n - 1):
            b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        M = np.linalg.solve(A, b)
        for i in range(n - 1):
            if x0 >= x[i] and x0 <= x[i + 1]:
                a = (M[i + 1] - M[i]) / (6 * h[i])
                b = M[i] / 2
                c = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
                d = y[i]
                return a * (x0 - x[i]) ** 3 + b * (x0 - x[i]) ** 2 + c * (x0 - x[i]) + d
        return None

    def cubic_spline_Clamped(self, x, y, x0, alpha, beta):
        n = len(x)
        h = np.zeros(n - 1)
        for i in range(n - 1):
            h[i] = x[i + 1] - x[i]
        A = np.zeros((n, n))
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        A[n - 1, n - 2] = h[n - 2]
        A[n - 1, n - 1] = 2 * h[n - 2]
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
        b = np.zeros(n)
        b[0] = 6 * ((y[1] - y[0]) / h[0] - alpha)
        b[n - 1] = 6 * (beta - (y[n - 1] - y[n - 2]) / h[n - 2])
        for i in range(1, n - 1):
            b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        M = np.linalg.solve(A, b)
        for i in range(n - 1):
            if x0 >= x[i] and x0 <= x[i + 1]:
                a = (M[i + 1] - M[i]) / (6 * h[i])
                b = M[i] / 2
                c = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
                d = y[i]
                return a * (x0 - x[i]) ** 3 + b * (x0 - x[i]) ** 2 + c * (x0 - x[i]) + d
        return None

if "__main__" == __name__:
    # test using the Lagrange interpolation
    print("test using the Lagrange interpolation")
    x = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    y = [0.13793103, 0.30769231, 0.8, 0.8, 0.30769231, 0.13793103]
    interpolation_object = interpolation(x, y)
    print(interpolation_object.lagrange(x, y, 0.1))

    # test part of the points
    print("test part of the points")
    x = [-2.5, -1.5, 1.5, 2.5]
    y = [0.13793103, 0.30769231, 0.30769231, 0.13793103]
    interpolation_object = interpolation(x, y)
    print(interpolation_object.lagrange(x, y, 0.1))

    # test using the Newton interpolation
    print("test using the Newton interpolation")
    x = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    y = [0.13793103, 0.30769231, 0.8, 0.8, 0.30769231, 0.13793103]
    interpolation_object = interpolation(x, y)
    print(interpolation_object.Newton(x, y, 0.1))

    # test part of the points
    print("test part of the points")
    x = [-2.5, -1.5, 1.5, 2.5]
    y = [0.13793103, 0.30769231, 0.30769231, 0.13793103]
    interpolation_object = interpolation(x, y)
    print(interpolation_object.Newton(x, y, 0.1))

    # plot the figure
    x = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    y = [0.13793103, 0.30769231, 0.8, 0.8, 0.30769231, 0.13793103]
    interpolation_object = interpolation(x, y)
    x0 = np.linspace(-2.5, 2.5, 100)
    y0 = np.zeros(x0.shape[0])
    for i in range(x0.shape[0]):
        y0[i] = interpolation_object.lagrange(x, y, x0[i])
    plt.Figure(dpi=300)
    plt.plot(x0, y0, label="Lagrange interpolation")
    plt.scatter(x, y, label="data points")
    for i in range(x0.shape[0]):
        y0[i] = interpolation_object.Newton(x, y, x0[i])
    plt.plot(x0, y0, label="Newton interpolation")
    x0 = np.linspace(-2.5, 2.5, 10000)
    y0 = 1 / (1 + x0 ** 2)
    plt.plot(x0, y0, label="Ground Truth")
    plt.legend()
    plt.show()

    # Runge phenomenon
    print("Runge phenomenon")
    plt.Figure(dpi=500)
    x_t = np.linspace(-5, 5, 10000)
    y_t = 1 / (1 + x_t ** 2)
    plt.plot(x_t, y_t, label="Ground Truth")
    for n in tqdm(range(5, 15, 3)):
        x = np.linspace(-5, 5, n)
        y = 1 / (1 + x ** 2)
        interpolation_object = interpolation(x, y)
        x0 = np.linspace(-5, 5, 10000)
        y0 = np.zeros(x0.shape[0])
        for i in range(x0.shape[0]):
            y0[i] = interpolation_object.lagrange(x, y, x0[i])
        plt.scatter(x, y, label="data points" + str(n))
        plt.plot(x0, y0, label="Lagrange interpolation" + str(n))
    plt.legend()
    plt.show()

    # test using the piecewise linear interpolation
    print("test using the piecewise linear interpolation")
    X = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    Y = [0.13793103, 0.30769231, 0.8, 0.8, 0.30769231, 0.13793103]
    interpolation_object = interpolation(X, Y)
    x = np.linspace(-2.5, 2.5, 10000)
    y = []
    for i in x:
        y.append(interpolation_object.piecewise_linear_interpolation(X, Y, i))
    plt.figure(dpi=300)
    x_t = np.linspace(-2.5, 2.5, 10000)
    y_t = 1 / (1 + x_t ** 2)
    plt.plot(x_t, y_t, label="Ground Truth")
    plt.scatter(X, Y, label="data points")
    l1 = plt.plot(x, y, label='piecewise linear')
    plt.legend()
    plt.show()

    # Runge phenomenon 2
    print("Runge phenomenon 2")
    plt.Figure(dpi=500)
    x_t = np.linspace(-5, 5, 10000)
    y_t = 1 / (1 + x_t ** 2)
    plt.plot(x_t, y_t, label="Ground Truth")
    for n in tqdm(range(5, 15, 3)):
        x = np.linspace(-5, 5, n)
        y = 1 / (1 + x ** 2)
        interpolation_object = interpolation(x, y)
        x0 = np.linspace(-5, 5, 10000)
        y0 = np.zeros(x0.shape[0])
        for i in range(x0.shape[0]):
            y0[i] = interpolation_object.piecewise_linear_interpolation(x, y, x0[i])
        plt.scatter(x, y, label="data points" + str(n))
        plt.plot(x0, y0, label="piecewise linear" + str(n))
    plt.legend()
    plt.show()

    # test using the cubic_spline_natural
    print("test using the cubic_spline_natural")
    X = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    Y = [0.13793103, 0.30769231, 0.8, 0.8, 0.30769231, 0.13793103]
    interpolation_object = interpolation(X, Y)
    x = np.linspace(-2.5, 2.5, 10000)
    y = []
    for i in x:
        y.append(interpolation_object.cubic_spline_natural(X, Y, i))
    plt.figure(dpi=300)
    x_t = np.linspace(-2.5, 2.5, 10000)
    y_t = 1 / (1 + x_t ** 2)
    plt.plot(x_t, y_t, label="Ground Truth")
    plt.scatter(X, Y, label="data points")
    l1 = plt.plot(x, y, label='cubic_spline_natural')
    plt.legend()
    plt.show()

    # test using the cubic_spline_clamped
    print("test using the cubic_spline_clamped")
    X = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    Y = [0.13793103, 0.30769231, 0.8, 0.8, 0.30769231, 0.13793103]
    interpolation_object = interpolation(X, Y)
    x = np.linspace(-2.5, 2.5, 10000)
    y = []
    for i in x:
        y.append(interpolation_object.cubic_spline_Clamped(X, Y, i, 2, -2))
    plt.figure(dpi=300)
    x_t = np.linspace(-2.5, 2.5, 10000)
    y_t = 1 / (1 + x_t ** 2)
    plt.plot(x_t, y_t, label="Ground Truth")
    plt.scatter(X, Y, label="data points")
    l1 = plt.plot(x, y, label='cubic_spline_clamped')
    plt.legend()
    plt.title("the first derivative of the end point is 2 and -2")
    plt.show()

    # Runge phenomenon 3
    print("Runge phenomenon 3")
    plt.Figure(dpi=1000)
    x_t = np.linspace(-5, 5, 10000)
    y_t = 1 / (1 + x_t ** 2)
    plt.plot(x_t, y_t, label="Ground Truth")
    for n in tqdm(range(5, 15, 3)):
        x = np.linspace(-5, 5, n)
        y = 1 / (1 + x ** 2)
        interpolation_object = interpolation(x, y)
        x0 = np.linspace(-5, 5, 10000)
        y0 = np.zeros(x0.shape[0])
        for i in range(x0.shape[0]):
            y0[i] = interpolation_object.cubic_spline_natural(x, y, x0[i])
        plt.scatter(x, y, label="data points" + str(n))
        plt.plot(x0, y0, label="cubic_spline_natural" + str(n))
    plt.legend()
    plt.show()


