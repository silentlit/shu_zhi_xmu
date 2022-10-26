import numpy as np
import sympy


def thomas(l_list, d_list, u_list, b_list):
    u_list[0] /= d_list[0]
    b_list[0] /= d_list[0]

    for i in range(1, len(l_list)):
        tem = d_list[i] - l_list[i] * u_list[i - 1]
        u_list[i] /= tem
        b_list[i] = (b_list[i] - l_list[i] * b_list[i - 1]) / tem

    x_list = [b_list[-1]]
    for i in range(len(b_list) - 2, -1, -1):
        x_list.insert(0, b_list[i] - u_list[i] * x_list[0])

    return x_list


def run_case_1():
    print('{a}{b}{a}'.format(a='*' * 30, b='解下列三角方程组开始'))
    l_list = [0.05, 0.075, 0.083, 0.0875, 0.2]
    d_list = [1.0, -1.0, -1.0, -1.0, -1.0]
    u_list = [-0.333, 0.15, 0.125, 0.1167, 0.1125]
    b_list = [0.0296, -0.0356, -0.0356, -0.0356, -0.0356, -0.0472]
    print(thomas(l_list, d_list, u_list, b_list))
    print('{a}{b}{a}'.format(a='*' * 30, b='解下列三角方程组结束'))


def seidel_sor(cof_matrix: np.ndarray, b_array: np.ndarray, eps=1e-4, x_array=None, omega=1.0):
    d_array = 1.0 * cof_matrix.diagonal()
    mask_matrix = np.multiply(-1 * np.eye(cof_matrix.shape[0]), cof_matrix)
    mask_matrix += cof_matrix
    mask_matrix *= -1.0
    x_array = 1.0 * (np.array(x_array) if x_array is not None else np.zeros_like(d_array))

    count = 0
    infeasibility = 1e10
    while infeasibility > eps:
        tem_infeasibility = 0
        for i in range(len(d_array)):
            x_old = x_array[i]
            x_new = (np.dot(mask_matrix[i], x_array) + b_array[i]) / d_array[i]
            x_array[i] = (1 - omega) * x_old + omega * x_new
            tem_infeasibility = max(tem_infeasibility, abs(x_old - x_array[i]))
        infeasibility = tem_infeasibility
        count += 1
    return x_array, count


def run_case_2():
    print('{a}{b}{a}'.format(a='*' * 30, b='用松弛法解下面的方程组开始'))
    cof_matrix = np.array([[56, 22, 11, -18],
                           [17, 66, -12, 7],
                           [3, -5, 47, 20],
                           [11, 16, 17, 10]])
    b_array = np.array([34, 82, 18, 26])
    eps = 1e-4
    omega = 1.0
    x_array, count = seidel_sor(cof_matrix, b_array, eps, None, omega)
    print('omega: {}, 次数: {}, 结果: {}'.format(omega, count, x_array))

    omega_list = list(map(lambda x: round(0.5 + x * 0.05, 2), range(int((1.6 - 0.5) / 0.05 + 1))))
    count_min = 1e10
    omega_opt = None
    for omega in omega_list:
        x_array, count = seidel_sor(cof_matrix, b_array, eps, None, omega)
        res_str = 'omega: {}, 次数: {}, 结果: {}'.format(omega, count, x_array)
        print(res_str)
        if count_min > count:
            count_min = count
            omega_opt = omega
    print('最小次数: {}, omega: {}'.format(count_min, omega_opt))
    print('{a}{b}{a}'.format(a='*' * 30, b='用松弛法解下面的方程组结束'))


def calcDFx(fx, x0, delta=1e-6, use_sp=False, subs: dict=None, var_name=None):
    try:
        if not use_sp:
            expr = sympy.sympify(fx)
            y0 = expr.evalf(subs={'x': x0})
            x1 = x0 + delta
            y1 = expr.evalf(subs={'x', x1})
            return (y1 - y0) / (x1 - x0)
        else:
            if subs is None or var_name is None:
                expr = sympy.sympify(fx)
                return expr.diff('x').evalf(subs={'x': x0})
            else:
                expr = sympy.sympify(fx)
                return expr.diff(var_name).evalf(subs=subs)
    except Exception as err:
        print('微分问题: {}, fx: {}, x0: {}'.format(err, fx, x0))
    return None


def nt_lfs(func_list, var_list, ite_dict=None, eps=1e-4):
    expr_list = list(map(lambda x: sympy.sympify(x), func_list))
    if ite_dict is None:
        ite_dict = {}
        for var in var_list:
            ite_dict[var] = 1.0

    infeasibility = 1e10
    while infeasibility > eps:
        cof_matrix = 1.0 * np.zeros(shape=(len(expr_list), len(var_list)))
        for i in range(cof_matrix.shape[0]):
            for j in range(cof_matrix.shape[1]):
                var_name = var_list[j]
                cof_matrix[i][j] = calcDFx(expr_list[i],
                                           ite_dict[var_name],
                                           use_sp=True,
                                           subs=ite_dict,
                                           var_name=var_name)
        b_array = -1.0 * np.array(list(map(lambda expr: expr.evalf(subs=ite_dict), expr_list)))
        delta_x_array, count = seidel_sor(cof_matrix, b_array,
                                          x_array=list(map(lambda x: ite_dict[x], var_list)),
                                          omega=1.2)
        infeasibility = np.max(delta_x_array)
        for i, var in enumerate(var_list):
            ite_dict[var] += delta_x_array[i]


def run_case_3():
    print('{a}{b}{a}'.format(a='*' * 30, b='用牛顿-拉夫森法解如下非线性方程组开始'))
    func_list = [
        '400 - 0.0075 * (300 - T1) ** 2 - T2',
        '400 - 0.02 * (400 - T2) ** 2 - T1'
    ]
    var_list = ['T1', 'T2']
    ite_dict = {
        'T1': 180.0,
        'T2': 292.0
    }
    nt_lfs(func_list, var_list, ite_dict, 1e-4)
    print('求解结果: {}'.format(ite_dict))
    print('{a}{b}{a}'.format(a='*' * 30, b='用牛顿-拉夫森法解如下非线性方程组结束'))


if __name__ == '__main__':
    run_case_1()
    run_case_2()
    run_case_3()
