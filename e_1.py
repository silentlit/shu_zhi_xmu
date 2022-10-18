def get_nt_res(x_list: list, y_list: list, x_new):
    res_list = y_list.copy()

    for i in range(len(x_list) - 1):
        for j in range(len(res_list) - i - 1):
            res_list[j] = (res_list[j] - res_list[j + 1]) / (x_list[j] - x_list[j + 1 + i])

    res = res_list[0]
    for i in range(1, len(res_list)):
        res = res * (x_new - x_list[i]) + res_list[i]

    return res


def get_lgr_res(x_list: list, y_list: list, x_new):
    res = 0.0
    for i, y in enumerate(y_list):
        lgr_cof = 1.0
        for j, x in enumerate(x_list):
            lgr_cof *= (x - x_new) / (x - x_list[i]) if i != j else 1
        res += y * lgr_cof
    return res


def case_1():
    x_list = [37.8, 21.1, 4.4, -12.2, -28.9]
    y_list = [15.4, 10.3, 6.6, 3.9, 2.2]
    # x_list.reverse()
    # y_list.reverse()

    print('f(-20) res: {:.6f}, f(20) res: {:.6f}'.format(get_nt_res(x_list[1:], y_list[1:], -20),
                                                         get_nt_res(x_list[:-1], y_list[:-1], 20)))
    print('f(-20) res: {:.6f}, f(20) res: {:.6f}'.format(get_lgr_res(x_list[1:], y_list[1:], -20),
                                                         get_lgr_res(x_list[:-1], y_list[:-1], 20)))


if __name__ == '__main__':
    case_1()
