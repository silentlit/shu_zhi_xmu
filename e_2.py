import numpy as np


def rich_ex(func=None, x_0=np.pi / 6, n=3, h=1e-2):
    ori_func = np.sin if func is None else func
    diff_func_with_x_0 = (lambda _h: (ori_func(x_0 + _h) - ori_func(x_0 - _h)) * 0.5 / _h)

    res_list = []
    for i in range(n):
        h_i = h * 0.5 ** i
        res_list.insert(0, diff_func_with_x_0(h_i))
    for i in range(1, n):
        for j in range(n - i):
            res_list[j] = ((4 ** i) * res_list[j] - res_list[j + 1]) / ((4 ** i) - 1)
    res_list.reverse()
    print('list: {}'.format(res_list))

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
    print('matrix:\n{}'.format(matrix_total))


def run_case_1():
    print('{a}{b}{a}'.format(a='*' * 30, b='例2 利用理查森外推法计算开始'))
    rich_ex(func=np.sin, x_0=np.pi / 3, n=3, h=1e-3)
    print('{a}{b}{a}'.format(a='*' * 30, b='例2 利用理查森外推法计算结束'))


def rom_brg(func=None, a=0, b=1, n=10, eps=1e-12):
    func = (lambda _x: np.exp(-1.0 * _x * _x)) if func is None else func
    matrix_total = np.zeros(shape=(n, n))

    matrix_total[0][0] = 0.5 * (b - a) * (func(a) + func(b))
    h = b - a
    for i in range(1, n):
        h *= 0.5
        sum_h = 0
        k = 0
        while 2 * k < 2 ** i:
            sum_h += func(a + h * (2 * k + 1))
            k += 1
        matrix_total[i][0] = 0.5 * matrix_total[i - 1][0] + h * sum_h

    m = 1.0
    for j in range(1, n):
        m *= 4
        for i in range(j, n):
            matrix_total[i][j] = matrix_total[i][j - 1] + (matrix_total[i][j - 1] -
                                                           matrix_total[i - 1][j - 1]) / (m - 1)
            if abs(matrix_total[i][j] - matrix_total[i][j - 1]) < eps:
                # print(matrix_total)
                print('res: {}'.format(matrix_total[i][j]))
                return
    # print(matrix_total)
    print('未达到精度要求, res: {}'.format(matrix_total[n - 1][n - 1]))


def run_case_2():
    print('{a}{b}{a}'.format(a='*' * 30, b='例:用龙贝格算法计算开始'))
    rom_brg(func=lambda _x: np.exp(-1.0 * _x * _x), a=0, b=1, n=10, eps=1e-12)
    print('{a}{b}{a}'.format(a='*' * 30, b='例:用龙贝格算法计算结束'))


def auto_simp(func=None, a=0, b=2 * np.pi, eps=1e-10, level=0, level_max=32):
    func = (lambda x: np.cos(2 * x) * np.exp(-x)) if func is None else func
    eps_i = 15 * eps * (0.5 ** level)
    if level > level_max:
        print('level > level_max: {} > {}'.format(level, level_max))
        return 0
    else:
        h = b - a
        h_4 = h * 0.25
        x_array = np.array(list(map(lambda _i: a + _i * h_4, range(5))))
        y_array = func(x_array)
        sum_2 = h * (y_array[0] + 4 * y_array[2] + y_array[4]) / 6.0
        sum_4 = h * (y_array[0] + 4 * y_array[1] + 2 * y_array[2] + 4 * y_array[3] + y_array[4]) / 12.0
        if abs(sum_2 - sum_4) < eps_i:
            return sum_4
        else:
            return (auto_simp(func, x_array[0], x_array[2], eps, level + 1, level_max) +
                    auto_simp(func, x_array[2], x_array[4], eps, level + 1, level_max))


def run_case_3():
    print('{a}{b}{a}'.format(a='*' * 30, b='例:用自适应辛普生法计算开始'))
    eps_1 = 0.5 * 1e-4
    res_1 = auto_simp(func=lambda x: np.cos(2 * x) * np.exp(-x), a=0, b=2 * np.pi,
                      eps=eps_1, level=0, level_max=32)
    eps_2 = 0.5 * 1e-9
    res_2 = auto_simp(func=lambda x: np.cos(2 * x) * np.exp(-x), a=0, b=2 * np.pi,
                      eps=eps_2, level=0, level_max=32)
    print('eps = {}, res = {}\neps = {}, res = {}'.format(eps_1, res_1,
                                                          eps_2, res_2))
    print('{a}{b}{a}'.format(a='*' * 30, b='例:用自适应辛普生法计算结束'))


if __name__ == '__main__':
    run_case_1()
    run_case_2()
    run_case_3()
