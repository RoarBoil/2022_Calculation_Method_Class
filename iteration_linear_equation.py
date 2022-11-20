# -*- coding;utf-8 -*-
"""
File name : iteration_linear_equation.PY
create: 20/11/2022 12:47
Last modified: 20/11/2022 12:47
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
Version: 0.1
"""
import matplotlib.pyplot as plt
import numpy as np
import time

class iteration_linear_equations(object):
    """
    This class is used to solve the linear equations using direct methods.
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.L = np.zeros((len(A), len(A)))
        self.U = np.zeros((len(A), len(A)))
        for i in range(len(A)):
            for j in range(len(A)):
                if i > j:
                    self.L[i, j] = -A[i, j]
                elif i < j:
                    self.U[i, j] = -A[i, j]
        self.D = A + self.L + self.U

    def check_spectral_radius(self, X):
        """
        Check whether the spectral radius is less than 1.
        :return: True or False
        """
        if len(X) != len(X):
            return False
        if np.max(np.abs(np.linalg.eigvals(X))) >= 1:
            return False
        return True

    def jacobi(self, x0, tol=1e-8, max_iter=100):
        """
        Jacobi method.
        :param x0: Initial value.
        :param tol: Tolerance.
        :param max_iter: Maximum number of iterations.
        :return: The solution of the linear equations.
        """
        x = x0
        J = np.linalg.inv(self.D).dot(self.L + self.U)
        f = np.linalg.inv(self.D).dot(self.b)
        if not self.check_spectral_radius(J):
            print("The spectral radius is larger than 1, not converge")
            return None
        for i in range(max_iter):
            x_new = J.dot(x) + f
            if np.linalg.norm(x_new - x) < tol:
                print(i)
                return x_new
            x = x_new
            print("The iteration:", i, "=======")
            print(x)
        return x

    def gauss_seiel(self, x0, tol=1e-8, max_iter=100):
        """
        gauss_seiel method.
        :param x0: Initial value.
        :param tol: Tolerance.
        :param max_iter: Maximum number of iterations.
        :return: The solution of the linear equations.
        """
        x = x0
        G = np.linalg.inv(self.D - self.L).dot(self.U)
        f = np.linalg.inv(self.D - self.L).dot(self.b)
        if not self.check_spectral_radius(G):
            print("The spectral radius is larger than 1, not converge")
            return None
        for i in range(max_iter):
            x_new = G.dot(x) + f
            if np.linalg.norm(x_new - x) < tol:
                print(i)
                return x_new
            x = x_new
            print("The iteration:", i, "=======")
            print(x)
        return x

    def SOR(self, x0, w, tol=1e-8, max_iter=100):
        """
        SOREL method.
        :param x0: Initial value.
        :param tol: Tolerance.
        :param max_iter: Maximum number of iterations.
        :return: The solution of the linear equations.
        """
        x = x0
        S = np.linalg.inv(self.D - w * self.L).dot((1- w) * self.D + w *self.U)
        f = w * np.linalg.inv(self.D - w * self.L).dot(self.b)
        if not self.check_spectral_radius(S):
            print("The spectral radius is larger than 1, not converge")
            return None
        for i in range(max_iter):
            x_new = S.dot(x) + f
            if np.linalg.norm(x_new - x) < tol:
                print(i)
                return x_new
            x = x_new
            print("The iteration:", i, "=======")
            print(x)
        return x

if "__main__" == __name__:
    # test using the example above using the Jacobi method
    print("test the example above using the Jacobi method")
    A = np.array([[1, 2, -2], [1, 1, 1], [2, 2, 1]])
    b = np.array([7, 2, 5])
    x0 = np.array([0, 0, 0])
    iteration = iteration_linear_equations(A, b)
    print("The solution of the linear equations is: ", iteration.jacobi(x0, 1e-6, 100))

    # test using the example above using the Gauss-Seidel method
    print("test the example above using the Gauss-Seidel method")
    A = np.array([[2, -1, 1], [2, 2, 2], [-1, -1, 2]])
    b = np.array([-1, 4, 5])
    x0 = np.array([0, 0, 0])
    iteration = iteration_linear_equations(A, b)
    print("The solution of the linear equations is: ", iteration.gauss_seiel(x0, 1e-6, 100))

    # limitation of the Jacobi method
    print("limitation of the Jacobi method")
    A = np.array([[2, -1, 1], [2, 2, 2], [-1, -1, 2]])
    b = np.array([-1, 4, 5])
    x0 = np.array([0, 0, 0])
    iteration = iteration_linear_equations(A, b)
    print("The solution of the linear equations is: ", iteration.jacobi(x0, 1e-6, 100))

    # # limitation of the Gauss-Seidel method
    print("limitation of the Gauss-Seidel method")
    A = np.array([[1, 2, -2], [1, 1, 1], [2, 2, 1]])
    b = np.array([7, 2, 5])
    x0 = np.array([0, 0, 0])
    iteration = iteration_linear_equations(A, b)
    print("The solution of the linear equations is: ", iteration.gauss_seiel(x0, 1e-6, 100))

    # test using the example above using the SOR method
    print("test the example above using the SOR method")
    A = np.array([[2, -1, 1], [2, 2, 2], [-1, -1, 2]])
    b = np.array([-1, 4, 5])
    x0 = np.array([0, 0, 0])
    iteration = iteration_linear_equations(A, b)
    print("The solution of the linear equations is: ", iteration.SOR(x0, 1.03, 1e-6, 100))

    # test under different w
    print("test under different w")
    A = np.array([[2, -1, 1], [2, 2, 2], [-1, -1, 2]])
    b = np.array([-1, 4, 5])
    x0 = np.array([0, 0, 0])
    iteration = iteration_linear_equations(A, b)
    iteration_list = []
    for i in range(50, 150):
        tmp_result = iteration.SOR(x0, i/100, 1e-6, 100)
        if type(tmp_result) is int:
            iteration_list.append(tmp_result)
    plt.Figure(dpi=300)
    plt.plot([i/100 for i in range(50, 50 + len(iteration_list))], iteration_list, label="w")
    plt.legend()
    plt.show()

    # compare Gauss-Seidel method and SOR method
    print("compare Gauss-Seidel method and SOR method")
    n = 5
    for kk in range(1000):
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        x0 = np.array([0 for i in range(n)])
        iteration = iteration_linear_equations(A, b)
        iteration_list = []
        for i in range(50, 150):
            tmp_result = iteration.SOR(x0, i/100, 1e-6, 200)
            if type(tmp_result) is int:
                iteration_list.append(tmp_result)
        if len(iteration_list) < 10:
            continue
        plt.Figure(dpi=300)
        plt.plot([i/100 for i in range(50, 50 + len(iteration_list))], iteration_list, label="w")
        plt.legend()
        plt.show()
        break



