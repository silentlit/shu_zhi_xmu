import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


def tem_func(t, y):
    y_0, y_1 = y
    return [y_1, 2.0 * t + 3.0 * y_0]


def target_method(func, init_value_list, t_span, y_upper, z_1=0.0, z_2=2.0, N=10, method='RK45', eps=1e-6):
    y_list = init_value_list.copy()
    y_list[1] = z_1
    res = solve_ivp(func, t_span, y_list, method=method)
    y_res = res.y
    f_z_1 = y_res[0][-1]
    infeasible = abs(f_z_1 - y_upper)
    if infeasible < eps:
        return res
    N -= 1

    y_list[1] = z_2
    res = solve_ivp(func, t_span, y_list, method=method)
    y_res = res.y
    f_z_2 = y_res[0][-1]
    infeasible = abs(f_z_2 - y_upper)
    if infeasible < eps:
        return res
    N -= 1

    while infeasible > eps and N > 0:
        z_3 = z_2 + (y_upper - f_z_2) * (z_2 - z_1) / ((f_z_2 - f_z_1) + 1e-9)
        y_list[1] = z_3
        res = solve_ivp(func, t_span, y_list, method=method)
        y_res = res.y
        f_z_3 = y_res[0][-1]
        infeasible = abs(f_z_3 - y_upper)
        N -= 1
        z_1 = z_2
        z_2 = z_3
        f_z_1 = f_z_2
        f_z_2 = f_z_3
    return res


def limit_diff_method(N, x_0, x_n, u_func, v_func, w_func, alpha, beta):
    h = 1.0 * (x_n - x_0) / N
    x_list = [x_0 + h * x for x in range(0, N + 1)]
    u_list = list(map(lambda x: u_func(x), x_list))
    v_list = list(map(lambda x: v_func(x), x_list))
    w_list = list(map(lambda x: w_func(x), x_list))
    cof_matrix = np.zeros(shape=(N + 1, N + 1))
    for i in range(1, N):
        cof_matrix[i][i - 1] = 2.0 + h * w_list[i]
        cof_matrix[i][i] = -(4.0 + 2.0 * (h ** 2) * v_list[i])
        cof_matrix[i][i + 1] = 2.0 - h * w_list[i]
    cof_matrix[0][0] = 1.0
    cof_matrix[-1][-1] = 1.0
    b_list = list(map(lambda idx: 2.0 * (h ** 2) * u_list[i], u_list))
    b_list[0] = 1.0 * alpha
    b_list[-1] = 1.0 * beta
    b_matrix = np.array([b_list]).T
    return x_list, np.reshape(np.dot(np.linalg.inv(cof_matrix), b_matrix), -1).tolist()


def run_case_1():
    init_value_list = [0.0, 0.0]
    t_span = [0.0, 1.0]
    y_upper = 1.0
    z_1 = 0.0
    z_2 = 2.0
    res = target_method(tem_func, init_value_list, t_span, y_upper, z_1, z_2)
    plt.plot(res.t, res.y[0], 'b', label='y_target_method')

    x_list, y_list = limit_diff_method(10, 0, 1, lambda x: 2.0 * x, lambda _: 3.0, lambda _: 0.0, 0.0, 1.0)
    plt.plot(x_list, y_list, 'r', label='y_limit_diff_method')

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close()


if __name__ == '__main__':
    run_case_1()
