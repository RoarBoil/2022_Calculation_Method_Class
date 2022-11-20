# -*- coding;utf-8 -*-
"""
File name : direct_linear_equations.PY
create: 19/11/2022 15:42
Last modified: 19/11/2022 15:42
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
Version: 0.1
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plts


class direct_linear_equations(object):
    """
    This class is used to solve the linear equations using direct methods.
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def check_singular(self):
        """
        Check whether the matrix is singular.
        :return: True or False
        """
        if len(self.A) != len(self.A[0]):
            return True
        elif np.linalg.matrix_rank(self.A) != len(self.A):
            return True
        else:
            return False

    def check_diagonally_dominant(self):
        """
        Check whether the matrix is diagonally_dominant.
        :return: True or False
        :return:
        """
        if len(self.A) != len(self.A[0]):
            return False
        for i in range(len(self.A)):
            if abs(self.A[i][i]) < sum(abs(self.A[i])) - abs(self.A[i][i]):
                return False
        return True

    def check_positive_definite(self):
        """
        Check whether the matrix is positive definite.
        :return: True or False
        """
        if len(self.A) != len(self.A[0]):
            return False
        for i in range(len(self.A)):
            if np.linalg.det(self.A[:i + 1, :i + 1]) <= 0:
                return False
        return True

    def gaussian_elimination(self):
        """
        Solve the linear equations using Gaussian elimination method.
        :return: the solution of the linear equations.
        """
        if self.check_singular():
            raise ValueError("The matrix is singular.")
        if not self.check_diagonally_dominant():
            print(Warning("Warning: The matrix is not diagonally dominant, results may be inaccurate using normal Gaussian_elimination."))
        if not self.check_positive_definite():
            print(Warning("Warning: The matrix is not positive_definite, results may be inaccurate using normal Gaussian_elimination."))
        A = self.A.copy()
        b = self.b.copy()
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                b[j] = b[j] - b[i] * A[j][i] / A[i][i]
                A[j] = A[j] - A[i] * A[j][i] / A[i][i]
                if n < 10:
                    print(A)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i], x)) / A[i][i]
        return x

    def elimination_with_maximal_column_pivoting(self):
        """
        Solve the linear equations using elimination_with_maximal_column_pivoting method.
        :return: the solution of the linear equations.
        """
        if self.check_singular():
            raise ValueError("The matrix is singular.")
        n = len(self.A)
        A = self.A.copy()
        b = self.b.copy()
        for i in range(n):
            if n < 10:
                print("====iteration", i + 1)
                print("start:")
                print(A)
            max_index = np.argmax(abs(A[i:, i])) + i
            A[[i, max_index]] = A[[max_index, i]]
            b[[i, max_index]] = b[[max_index, i]]
            if n < 10:
                print("after pivoting:")
                print(A)
            for j in range(i + 1, n):
                b[j] = b[j] - b[i] * A[j][i] / A[i][i]
                A[j] = A[j] - A[i] * A[j][i] / A[i][i]
            if n < 10:
                print("after elimination:")
                print(A)

        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i], x)) / A[i][i]
        return x

    def check_LU(self):
        """
        Check whether the matrix can be use for LU decomposition.
        :return: True or False
        """
        for i in range(1, len(self.A)):
            if np.linalg.det(self.A[:i, :i]) == 0:
                return False
        return True

    def LU_decomposition(self):
        """
        Solve the linear equations using LU_decomposition method.
        :return: the solution of the linear equations.
        """
        if self.check_singular():
            raise ValueError("The matrix is singular.")
        if not self.check_LU():
            raise ValueError("some of the sequential principal minor of the matrix are zero.")
        n = len(self.A)
        A = self.A.copy()
        b = self.b.copy()
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        for i in range(n):
            L[i][i] = 1
            for j in range(i, n):
                U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            for j in range(i + 1, n):
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
        if n < 10:
            print("L:  ", L)
            print("U:  ", U)
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))) / U[i][i]
        return x


if "__main__" == __name__:
    # test using the example above
    print("test using the example above")
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    linear_equations = direct_linear_equations(A, b)
    print(linear_equations.gaussian_elimination())

    # test if the code can detect singular matrix
    print("test if the code can detect singular matrix")
    A = np.array([[0, 1, -1], [0, -1, 2], [0, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    linear_equations = direct_linear_equations(A, b)
    print(linear_equations.gaussian_elimination())

    # test the gaussian_elimination on large matrix
    print("test the gaussian_elimination on large matrix")
    A = np.random.rand(4000, 4000)
    b = np.random.rand(4000)
    initial_time = time.time()
    linear_equations = direct_linear_equations(A, b)
    linear_equations.gaussian_elimination()
    print("The time cost of gaussian_elimination is: ", time.time() - initial_time)

    # test using the example above, but with elimination_with_maximal_column_pivoting
    print("test using the example above, but with elimination_with_maximal_column_pivoting")
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    linear_equations = direct_linear_equations(A, b)
    print(linear_equations.elimination_with_maximal_column_pivoting())

    # compare the elimination_with_maximal_column_pivoting with gaussian_elimination
    print("compare the elimination_with_maximal_column_pivoting with gaussian_elimination")
    A = np.array([[0.0000000000000001, 2, 3], [-1, 3.712, 4.623], [-2, 1.072, 5.643]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)
    linear_equations = direct_linear_equations(A, b)
    print("===============gaussian_elimination================")
    print(linear_equations.gaussian_elimination())
    print("===============elimination_with_maximal_column_pivoting================")
    print(linear_equations.elimination_with_maximal_column_pivoting())

    # compare the time cost
    print("compare the time cost")
    time_list_gaussian = []
    time_list_column = []
    for n in range(100, 3000, 100):
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        linear_equations = direct_linear_equations(A, b)
        print("n = ", n)
        initial_time = time.time()
        linear_equations.gaussian_elimination()
        time_list_gaussian.append(time.time() - initial_time)
        print("The time cost of gaussian_elimination is: ", time.time() - initial_time)
        initial_time = time.time()
        linear_equations.elimination_with_maximal_column_pivoting()
        time_list_column.append(time.time() - initial_time)
        print("The time cost of elimination_with_maximal_column_pivoting is: ", time.time() - initial_time)
    plt.Figure(dpi=300)
    plt.plot(range(100, 3000, 100), time_list_gaussian, label="gaussian_elimination")
    plt.plot(range(100, 3000, 100), time_list_column, label="elimination_with_maximal_column_pivoting")
    plt.legend()
    plt.show()

    # test using the example above, but with LU_decomposition
    print("test using the example above, but with LU_decomposition")
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    linear_equations = direct_linear_equations(A, b)
    print(linear_equations.LU_decomposition())



