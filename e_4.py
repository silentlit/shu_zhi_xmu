import numpy as np
from matplotlib import pyplot as plt


def tri_sample_get_m(x_list, y_list, mode=2, x_0_pos_value=0.0, x_n_pos_value=0.0):
    h_list = list(map(lambda idx: 1.0 * x_list[idx + 1] - x_list[idx], range(len(x_list) - 1)))
    u_list = list(map(lambda idx: 2.0 * (h_list[idx - 1] + h_list[idx]), range(1, len(h_list))))
    b_list = list(map(lambda idx: 1.0 * (y_list[idx + 1] - y_list[idx]) / h_list[idx], range(len(y_list) - 1)))
    v_list = list(map(lambda idx: 6.0 * (b_list[idx] - b_list[idx - 1]), range(1, len(b_list))))

    cof_matrix = 1.0 * np.zeros(shape=(len(x_list), len(y_list)))
    for i in range(1, cof_matrix.shape[0] - 1):
        cof_matrix[i][i - 1] = h_list[i - 1]
        cof_matrix[i][i] = u_list[i - 1]
        cof_matrix[i][i + 1] = h_list[i]

    b_matrix = 1.0 * np.zeros(shape=(len(y_list), 1))
    for i in range(1, len(y_list) - 1):
        b_matrix[i] = v_list[i - 1]

    if mode == 2:
        cof_matrix[0][0] = 1.0
        cof_matrix[cof_matrix.shape[0] - 1][cof_matrix.shape[1] - 1] = 1.0
        b_matrix[0][0] = x_0_pos_value
        b_matrix[b_matrix.shape[0] - 1][0] = x_n_pos_value
    elif mode == 1:
        cof_matrix[0][0] = 2.0 * h_list[0]
        cof_matrix[0][1] = 1.0 * h_list[0]
        cof_matrix[cof_matrix.shape[0] - 1][cof_matrix.shape[1] - 2] = 1.0 * h_list[-1]
        cof_matrix[cof_matrix.shape[0] - 1][cof_matrix.shape[1] - 1] = 2.0 * h_list[-1]
        b_matrix[0][0] = 6.0 * (b_list[0] - x_0_pos_value)
        b_matrix[b_matrix.shape[0] - 1][0] = 6.0 * (x_n_pos_value - b_list[-1])
    else:
        raise Exception('mode需要指定为1或2, 表示使用1阶导数或2阶导数')

    return np.reshape(np.dot(np.linalg.inv(cof_matrix), b_matrix), -1).tolist()


def tri_sample(x_pos_value, x_list, y_list, mode=2, x_0_pos_value=0.0, x_n_pos_value=0.0):
    m_list = tri_sample_get_m(x_list, y_list, mode, x_0_pos_value, x_n_pos_value)
    x_pos_low_idx = 0
    for i in range(len(x_list) - 1):
        if x_pos_value < x_list[i + 1]:
            x_pos_low_idx = i
            break
        x_pos_low_idx = i
    x_j_1 = x_list[x_pos_low_idx + 1]
    x_j = x_list[x_pos_low_idx]
    h_j = x_j_1 - x_j
    m_j = m_list[x_pos_low_idx]
    m_j_1 = m_list[x_pos_low_idx + 1]
    y_j_1 = y_list[x_pos_low_idx + 1]
    y_j = y_list[x_pos_low_idx]
    s_x = ((((x_j_1 - x_pos_value) ** 3) * m_j + ((x_pos_value - x_j) ** 3) * m_j_1) / 6.0 / h_j +
           (y_j_1 / h_j - h_j * m_j_1 / 6.0) * (x_pos_value - x_j) +
           (y_j / h_j - h_j * m_j / 6.0) * (x_j_1 - x_pos_value))
    return s_x


def run_case_1():
    x_list = [0.0, 0.2, 0.6, 1.0, 2.0, 5.0, 10.0]
    y_list = [5.19, 3.77, 2.3, 1.57, 0.8, 0.25, 0.094]
    x_pos_list = [0.1, 0.4, 1.2, 5.8]
    print('{a}{b}{a}'.format(a='-' * 30, b='试利用样条函数插值求Ct开始'))
    for x in x_pos_list:
        print('s({}) = {}'.format(x, tri_sample(x, x_list, y_list, mode=1, x_0_pos_value=-9.45, x_n_pos_value=0.0)))
    print('{a}{b}{a}'.format(a='-' * 30, b='试利用样条函数插值求Ct结束'))

    try:
        plt.rcParams['font.sans-serif'] = ['Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(e)
    step = 0.05
    # length = 20
    # width = 25
    # figsize=(length, width)
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.35, hspace=0.42)

    print('{a}{b}{a}'.format(a='-' * 30, b='利用自由边界三次样条函数拟合开始'))
    x_list = [0.0, 0.6, 1.5, 1.7, 1.9]
    y_list = [-0.8, -0.34, 0.59, 0.59, 0.23]
    x_sample_list = [x_list[0]]
    for i in range(1, len(x_list)):
        while x_sample_list[-1] + step < x_list[i]:
            x_sample_list.append(x_sample_list[-1] + step)
        x_sample_list.append(x_list[i])
    y_sample_list = list(map(lambda x_: tri_sample(x_, x_list, y_list, 2, 0.0, 0.0), x_sample_list))
    ax = plt.subplot(2, 2, 1)
    plt.plot(x_sample_list, y_sample_list)
    plt.scatter(x_list, y_list)
    plt.title('case 1', fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.5)

    x_list = [2.1, 2.3, 2.6, 2.8, 3.0]
    y_list = [0.1, 0.28, 1.03, 1.5, 1.44]
    x_sample_list = [x_list[0]]
    for i in range(1, len(x_list)):
        while x_sample_list[-1] + step < x_list[i]:
            x_sample_list.append(x_sample_list[-1] + step)
        x_sample_list.append(x_list[i])
    y_sample_list = list(map(lambda x_: tri_sample(x_, x_list, y_list, 2, 0.0, 0.0), x_sample_list))
    ax = plt.subplot(2, 2, 2)
    plt.plot(x_sample_list, y_sample_list)
    plt.scatter(x_list, y_list)
    plt.title('case 2', fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.5)

    x_list = [3.6, 4.7, 5.2, 5.7, 5.8]
    y_list = [0.74, -0.82, -1.27, -0.92, -0.92]
    x_sample_list = [x_list[0]]
    for i in range(1, len(x_list)):
        while x_sample_list[-1] + step < x_list[i]:
            x_sample_list.append(x_sample_list[-1] + step)
        x_sample_list.append(x_list[i])
    y_sample_list = list(map(lambda x_: tri_sample(x_, x_list, y_list, 2, 0.0, 0.0), x_sample_list))
    ax = plt.subplot(2, 2, 3)
    plt.plot(x_sample_list, y_sample_list)
    plt.scatter(x_list, y_list)
    plt.title('case 3', fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.5)

    x_list = [6.0, 6.4, 6.9, 7.6, 8.0]
    y_list = [-1.04, -0.79, -0.06, 1.0, 0.0]
    x_sample_list = [x_list[0]]
    for i in range(1, len(x_list)):
        while x_sample_list[-1] + step < x_list[i]:
            x_sample_list.append(x_sample_list[-1] + step)
        x_sample_list.append(x_list[i])
    y_sample_list = list(map(lambda x_: tri_sample(x_, x_list, y_list, 2, 0.0, 0.0), x_sample_list))
    ax = plt.subplot(2, 2, 4)
    plt.plot(x_sample_list, y_sample_list)
    plt.scatter(x_list, y_list)
    plt.title('case 4', fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()
    plt.close()
    print('{a}{b}{a}'.format(a='-' * 30, b='利用自由边界三次样条函数拟合结束'))


if __name__ == '__main__':
    run_case_1()
