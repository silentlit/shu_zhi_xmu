import numpy as np


def case_1(func=None, x_0=np.pi / 6, n=3, h=1e-2):
    # x_0 = np.pi / 6
    ori_func = np.sin if func is None else func
    diff_func_with_x_0 = (lambda _h: (ori_func(x_0 + _h) - ori_func(x_0 - _h)) * 0.5 / _h)

    # h = 1e-6
    # print('(f(x + h) - f(x - h)) / 2 = {}, x = {}, h = {}'.format(diff_func_with_x_0(h), x_0, h))
    # print(math.cos(x_0))

    # n = 3
    # h = 1e-2

    # list type
    res_list = []
    for i in range(n):
        h_i = h * 0.5 ** i
        res_list.insert(0, diff_func_with_x_0(h_i))
    for i in range(1, n):
        for j in range(n - i):
            res_list[j] = ((4 ** i) * res_list[j] - res_list[j + 1]) / ((4 ** i) - 1)
    res_list.reverse()
    print(res_list)

    # table type
    matrix_total = np.zeros(shape=(n, n))
    for i in range(n):
        h_i = h * 0.5 ** i
        matrix_total[i][0] = diff_func_with_x_0(h_i)
    for j in range(1, n):
        for i in range(j, n):
            # matrix_total[i][j] = ((4 ** i) * matrix_total[i][j - 1] - matrix_total[i - 1][j - 1]) / ((4 ** i) - 1)
            matrix_total[i][j] = (matrix_total[i][j - 1] + (matrix_total[i][j - 1] - matrix_total[i - 1][j - 1]) /
                                  ((4 ** i) - 1))
    print(matrix_total)


def run_case_1():
    case_1(func=np.sin, x_0=np.pi / 3, n=3, h=1e-3)


if __name__ == '__main__':
    run_case_1()
