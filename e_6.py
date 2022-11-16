import numpy as np
from scipy.linalg import solve
from scipy.sparse import diags


def crank_nicolson(func, t_step, x_step, t_upper=1, x_upper=1, c=1):
    t_N = int(1.0 * t_upper / t_step)
    x_N = int(1.0 * x_upper / x_step)
    t_list = list()

    ss = 1.0 * c * t_step / (x_step ** 2)
    cof_a_matrix = diags((-ss, 2.0 + 2 * ss, -ss), (-1, 0, 1), shape=(x_N - 2, x_N - 2))
    cof_b_matrix = diags((ss, 2.0 - 2 * ss, ss), (-1, 0, 1), shape=(x_N - 2, x_N - 2))
    t_0_array = np.array(list(map(lambda x_: [func(0, x_)], np.arange(x_step, x_upper - x_step, x_step))))
    t_list.append(t_0_array)

    for i in range(1, t_N):
        b_matrix = np.dot(cof_b_matrix.toarray(), t_list[i - 1])
        t_i_res = solve(cof_a_matrix.toarray(), b_matrix)
        t_list.append(t_i_res)
        if i % int(t_N / 4) == 0:
            print('done {:.2f}..'.format(i / t_N))

    t_list = np.array(t_list)
    return t_list.reshape((t_N, -1))


def show_res(res, t_step, x_step):
    def find_by_value(value_matrix, t_value, x_value):
        return value_matrix[int(t_value / t_step)][int(x_value / x_step)]
    t_s = 0.02
    x_s = 0.2
    t_list = list(map(lambda x_: t_s * x_, range(1, 6)))
    x_list = list(map(lambda x_: x_s * x_, range(1, 5)))
    tem_res = list()
    tem_real = list()
    for t_v in t_list:
        tem_list = list()
        tem_r_list = list()
        for x_v in x_list:
            real = np.exp(-(np.pi ** 2) * t_v) * np.sin(np.pi * x_v)
            calc = find_by_value(res, t_v, x_v)
            tem_list.append(calc)
            tem_r_list.append(real)
            print('t = {}, x = {}, calc = {}, real = {}, err = {}'.format(t_v, x_v, calc, real, real - calc))
        tem_res.append(tem_list)
        tem_real.append(tem_r_list)
    print('计算: ')
    print(np.array(tem_res))
    print('实际: ')
    print(np.array(tem_real))


def run_case_1():
    t_step = 0.005
    x_step = 0.01
    print('{a}{b}{a}'.format(a='-' * 30,
                             b='t_step = {}, x_step = {}'.format(t_step, x_step)))
    res = crank_nicolson(lambda t, x: np.sin(np.pi * x) if t == 0 else Exception('t不等于0, 函数未定义'), t_step, x_step)
    tem_col = np.zeros((len(res), 1))
    res = np.c_[tem_col, res]
    res = np.c_[res, tem_col]
    show_res(res, t_step, x_step)

    t_step = 0.005
    x_step = 0.001
    print('{a}{b}{a}'.format(a='-' * 30,
                             b='t_step = {}, x_step = {}'.format(t_step, x_step)))
    res = crank_nicolson(lambda t, x: np.sin(np.pi * x) if t == 0 else Exception('t不等于0, 函数未定义'), t_step, x_step)
    tem_col = np.zeros((len(res), 1))
    res = np.c_[tem_col, res]
    res = np.c_[res, tem_col]
    show_res(res, t_step, x_step)


if __name__ == '__main__':
    run_case_1()
