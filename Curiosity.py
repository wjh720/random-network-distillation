import os
import math
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import tensorflow_probability as tfp


class Count:

    def __init__(self, n, shape, len=2):
        self.n = n
        self.shape = shape
        self.f = np.zeros(shape + [1], dtype='int32')
        self.len = len

    def output(self, num):
        f = self.f
        for i in range(self.n):
            if isinstance(num[i], np.ndarray):
                for j in range(self.len):
                    f = f[num[i][j]]
            else:
                f = f[num[i]]
        return np.squeeze(f)

    def add(self, num):
        f = self.f
        for i in range(self.n):
            if isinstance(num[i], np.ndarray):
                for j in range(self.len):
                    f = f[num[i][j]]
            else:
                f = f[num[i]]

        f[0] += 1


class Count_p:

    def __init__(self, n, shape, n_1, m_1, len=2):
        self.n_1 = n_1
        self.count_1 = Count(n, shape, len=len)
        self.count_2 = Count(n - n_1, shape[m_1:], len=len)

    def output(self, num):
        ans = self.count_1.output(num)
        if ans == 0:
            return 0
        else:
            res = self.count_2.output(num[self.n_1:])
            if res == 0:
                print(num)
                print(ans)
            ans = 1. * ans / res
            return ans

    def add(self, num):
        self.count_1.add(num)
        self.count_2.add(num[self.n_1:])


class Count_log_p:

    def __init__(self, n, shape, n_1, n_2, m_1, m_2):
        self.n_1 = n_1
        self.count_1 = Count_p(n, shape, n_2, m_2)
        self.count_2 = Count_p(n - n_1, shape[:-m_1], n_2, m_2)

    def output(self, num):
        ans = self.count_1.output(num)
        if ans == 0:
            return 0
        else:
            ans = 1. * ans / self.count_2.output(num[:-self.n_1])
            return ans

    def add(self, num):
        self.count_1.add(num)
        self.count_2.add(num[:-self.n_1])


class Count_div_p:

    def __init__(self, n, shape, n_1, n_2, m_1, m_2):
        self.n_1 = n_1
        self.count_1 = Count_p(n - n_1, shape[:-m_1], n_2, m_2)
        self.count_2 = Count_p(n, shape, n_2, m_2)

    def output(self, num):
        ans = self.count_2.output(num)
        if (ans == 0):
            return 1
        ans = 1. * self.count_1.output(num[:-self.n_1]) / ans
        return ans

    def add(self, num):
        self.count_1.add(num[:-self.n_1])
        self.count_2.add(num)


class Test_Count_div_p:

    def __init__(self, n, shape, n_1, n_2, m_1, m_2):
        self.n_1 = n_1
        self.count_1 = Count_p(n - n_1, shape[:-m_1], n_2, m_2)
        self.count_2 = Count_p(n, shape, n_2, m_2)

    def output(self, num):
        ans_1 = self.count_1.output(num[:-self.n_1])
        ans_2 = self.count_2.output(num)
        if ans_2 == 0:
            return 1, ans_1, ans_2
        ans = 1. * ans_1 / ans_2
        return ans, ans_1, ans_2

    def add(self, num):
        self.count_1.add(num[:-self.n_1])
        self.count_2.add(num)


class Hash_core:

    def __init__(self):
        self.mo1 = 1007
        self.mo2 = 1000000000000007
        self.buffer_size = 100000007
        self.buffer = {}

    def change(self, data):
        ans = 0
        for x in data:
            ans = (ans * self.mo1 + x + self.mo2) % self.mo2
        res = ans % self.buffer_size
        return res, ans

    def update(self, data):
        index, data = self.change(data)
        if data not in self.buffer:
            self.buffer[data] = {index: 0}
        else:
            self.buffer[data][index] += 1

    def output(self, data):
        index, data = self.change(data)
        if data not in self.buffer:
            return 0
        return self.buffer[data][index]


class Hash:

    def __init__(self, n, shape, len=2):
        self.n = n
        self.shape = shape
        self.f = Hash_core()
        self.len = len

    def get(self, num):
        index = []
        for i in range(self.n):
            if isinstance(num[i], np.ndarray):
                for j in range(self.len):
                    index.append(num[i][j])
            else:
                index.append(num[i])
        return index

    def output(self, num):
        index = self.get(num)
        return self.f.output(index)

    def add(self, num):
        index = self.get(num)
        self.f.update(index)


class Hash_p:

    def __init__(self, n, shape, n_1, m_1, len=2):
        self.n_1 = n_1
        self.count_1 = Hash(n, shape, len=len)
        self.count_2 = Hash(n - n_1, shape[m_1:], len=len)

    def output(self, num):
        ans = self.count_1.output(num)
        if ans == 0:
            return 0
        else:
            res = self.count_2.output(num[self.n_1:])
            if res == 0:
                print(num)
                print(ans)
            ans = 1. * ans / res
            return ans

    def add(self, num):
        self.count_1.add(num)
        self.count_2.add(num[self.n_1:])


class Hash_div_p:

    def __init__(self, n, shape, n_1, n_2, m_1, m_2, len=2):
        self.n_1 = n_1
        self.count_1 = Hash_p(n - n_1, shape[:-m_1], n_2, m_2, len=len)
        self.count_2 = Hash_p(n, shape, n_2, m_2, len=len)

    def output(self, num):
        ans = self.count_2.output(num)
        if ans == 0:
            return 1
        ans = 1. * self.count_1.output(num[:-self.n_1]) / ans
        return ans

    def add(self, num):
        self.count_1.add(num[:-self.n_1])
        self.count_2.add(num)


def add_shape(n, size):
    shape = []
    for i in range(n):
        shape += [size]
    return shape


def mk(x, y):
    return [np.array([x, y])]


def mk1(x):
    return [x]


class Key_points:

    def __init__(self, size, a_size, arg, is_print):

        self.arg = arg
        self.is_print = is_print

        if (self.is_print):
            self.figure_path = self.arg.save_path + 'sub-goals/'
            if os.path.exists(self.figure_path):
                shutil.rmtree(self.figure_path)
            os.makedirs(self.figure_path)
        # print(self.figure_path)

        self.size = size
        self.a_size = a_size
        self.range = 3

        log_shape = \
            add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size) + add_shape(2, size) + add_shape(1,
                                                                                                                  a_size)

        self.log_1 = Count_log_p(5, log_shape, 2, 1, 3, 2)
        self.log_2 = Count_log_p(5, log_shape, 2, 1, 3, 2)

        p_shape = \
            add_shape(2, size) + add_shape(1, a_size) + add_shape(2, self.range) + add_shape(2, size) + add_shape(1,
                                                                                                                  a_size)

        self.p_1 = Count_p(5, p_shape, 1, 2)
        self.p_2 = Count_p(5, p_shape, 1, 2)

        other_p_shape = \
            add_shape(2, size) + add_shape(1, a_size) + add_shape(2, self.range) + add_shape(2, size) + add_shape(1,
                                                                                                                  a_size)

        self.other_p_1 = Count_p(5, other_p_shape, 1, 2)
        self.other_p_2 = Count_p(5, other_p_shape, 1, 2)

        check_p_shape = add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

        self.check_p_1 = Count(3, check_p_shape)
        self.check_p_2 = Count(3, check_p_shape)

    def calc_del(self, s_t, next_s_t):
        return next_s_t - s_t + 1

    def update(self, state, next_state):

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        num_1 = [del_s_t_2, s_t_2, a_t_2, s_t_1, a_t_1]
        num_2 = [del_s_t_1, s_t_1, a_t_1, s_t_2, a_t_2]

        self.log_1.add(num_1)
        self.log_2.add(num_2)

        p_num_1 = [s_t_1, a_t_1, del_s_t_2, s_t_2, a_t_2]
        p_num_2 = [s_t_2, a_t_2, del_s_t_1, s_t_1, a_t_1]

        self.p_1.add(p_num_1)
        self.p_2.add(p_num_2)

        other_p_num_1 = [s_t_2, a_t_2, del_s_t_2, s_t_1, a_t_1]
        other_p_num_2 = [s_t_1, a_t_1, del_s_t_1, s_t_2, a_t_2]

        self.other_p_1.add(other_p_num_1)
        self.other_p_2.add(other_p_num_2)

        check_p_num_1 = [del_s_t_1, s_t_1, a_t_1]
        check_p_num_2 = [del_s_t_2, s_t_2, a_t_2]

        self.check_p_1.add(check_p_num_1)
        self.check_p_2.add(check_p_num_2)

    def output(self, state, next_state, agent_index):

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        num_1 = [del_s_t_2, s_t_2, a_t_2, s_t_1, a_t_1]
        num_2 = [del_s_t_1, s_t_1, a_t_1, s_t_2, a_t_2]

        if (self.arg.symmetry):
            p_log = self.log_1.output(num_1)

            if p_log != 0:
                p_log = math.log(p_log)
            # if (p_log != 0):
            #	print(p_log)

            p_log_2 = self.log_2.output(num_2)

            if p_log_2 != 0:
                p_log_2 = math.log(p_log_2)
            # if (p_log_2 != 0):
            #	print(p_log_2)

            return p_log + p_log_2

        if (agent_index == 0):
            p_log = self.log_1.output(num_1)

            if p_log != 0:
                p_log = math.log(p_log)

            return p_log
        else:
            p_log_2 = self.log_2.output(num_2)

            if p_log_2 != 0:
                p_log_2 = math.log(p_log_2)

            return p_log_2

    # return p_log + p_log_2

    def show(self, e):

        if (self.is_print == 0):
            return
        print("start showing, round %d: " % e)

        ans_1 = np.zeros([self.size, self.size])
        ans_2 = np.zeros([self.size, self.size])
        res_1 = np.zeros([self.size, self.size])
        res_2 = np.zeros([self.size, self.size])

        for i1 in range(self.size):
            for j1 in range(self.size):
                for k1 in range(self.a_size):
                    for i3 in range(i1 - 1, i1 + 2):
                        for j3 in range(j1 - 1, j1 + 2):
                            check_p_num_2 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
                            if (np.squeeze(self.check_p_2.output(check_p_num_2)) > 0):
                                for i2 in range(self.size):
                                    for j2 in range(self.size):
                                        for k2 in range(self.a_size):
                                            p_num_1 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i1, j1) + mk1(k1)
                                            prob = self.p_1.output(p_num_1)

                                            num_1 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1) + \
                                                    mk(i2, j2) + mk1(k2)
                                            p_log = self.log_1.output(num_1)

                                            p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i2, j2) + mk1(k2)
                                            prob_2 = self.other_p_1.output(p_num_2)

                                            if (p_log != 0 and p_log != 1):
                                                ans_1[i2][j2] += prob * math.log(p_log)
                                                res_2[i1][j1] += prob_2 * math.log(p_log)

                            check_p_num_1 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
                            if (np.squeeze(self.check_p_1.output(check_p_num_1)) > 0):
                                for i2 in range(self.size):
                                    for j2 in range(self.size):
                                        for k2 in range(self.a_size):
                                            p_num_2 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i1, j1) + mk1(k1)
                                            prob = self.p_2.output(p_num_2)

                                            num_2 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1) + \
                                                    mk(i2, j2) + mk1(k2)
                                            p_log = self.log_2.output(num_2)

                                            p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i2, j2) + mk1(k2)
                                            prob_2 = self.other_p_2.output(p_num_2)

                                            if p_log != 0 and p_log != 1:
                                                ans_2[i2][j2] += prob * math.log(p_log)
                                                res_1[i1][j1] += prob_2 * math.log(p_log)

        print("start drawing round %d" % e)

        figure = plt.figure(figsize=(16, 10))
        ax1 = figure.add_subplot(2, 2, 1)
        ax2 = figure.add_subplot(2, 2, 3)
        ax3 = figure.add_subplot(2, 2, 2)
        ax4 = figure.add_subplot(2, 2, 4)

        ax1.imshow(np.log(ans_1 + 1))
        ax2.imshow(np.log(ans_2 + 1))
        ax3.imshow(np.log(res_1 + 1))
        ax4.imshow(np.log(res_2 + 1))

        figure.savefig('%s/%i.png' % (self.figure_path, e))
        plt.close(figure)


class Dec:
    def __init__(self, size, n_agent, args):
        self.env_n_dim = args.env_n_dim
        self.size = size
        self.n_agent = n_agent
        self.args = args
        self.visited = [np.zeros([args.size, args.size, 2]) for _ in range(self.n_agent)]
        self.visited_old = [np.zeros([args.size, args.size, 2]) for _ in range(self.n_agent)]

        if self.env_n_dim == 1:
            self.show_visited = np.zeros([args.size, args.size])
            self.show_visited_old = np.zeros([args.size, args.size])

    def update(self, state, is_door_open):
        # For heat map
        for i in range(self.n_agent):
            self.visited[i][state[i][0]][state[i][1]][int(is_door_open)] += 1

        if self.env_n_dim == 1:
            self.show_visited[state[0][0]][state[1][0]] += 1

    def output(self, state, i):
        return 1. / np.sqrt(np.sum(self.visited[i][state[i][0]][state[i][1]]) + 1)

    def show(self, path, e):
        if self.env_n_dim == 2:
            figure = plt.figure(figsize=(16, 10))

            ax1 = figure.add_subplot(2, 6, 1)
            ax2 = figure.add_subplot(2, 6, 2)
            ax3 = figure.add_subplot(2, 6, 3)
            ax4 = figure.add_subplot(2, 6, 4)
            ax5 = figure.add_subplot(2, 6, 5)
            ax6 = figure.add_subplot(2, 6, 6)
            ax7 = figure.add_subplot(2, 6, 7)
            ax8 = figure.add_subplot(2, 6, 8)
            ax9 = figure.add_subplot(2, 6, 9)
            ax10 = figure.add_subplot(2, 6, 10)
            ax11 = figure.add_subplot(2, 6, 11)
            ax12 = figure.add_subplot(2, 6, 12)

            ax1.imshow(np.log(self.visited[0][:, :, 0] + 1))
            ax2.imshow(np.log(self.visited[0][:, :, 0] - self.visited_old[0][:, :, 0] + 1))
            ax3.imshow(np.log(self.visited[0][:, :, 1] + 1))
            ax4.imshow(np.log((self.visited[0][:, :, 1] - self.visited_old[0][:, :, 1] + 1)))
            ax5.imshow(np.log(np.sum(self.visited[0], axis=2) + 1))
            ax6.imshow(np.log(np.sum(self.visited[0], axis=2) - np.sum(self.visited_old[0], axis=2) + 1))

            ax7.imshow(np.log(self.visited[1][:, :, 0] + 1))
            ax8.imshow(np.log(self.visited[1][:, :, 0] - self.visited_old[1][:, :, 0] + 1))
            ax9.imshow(np.log(self.visited[1][:, :, 1] + 1))
            ax10.imshow(np.log((self.visited[1][:, :, 1] - self.visited_old[1][:, :, 1] + 1)))
            ax11.imshow(np.log(np.sum(self.visited[1], axis=2) + 1))
            ax12.imshow(np.log(np.sum(self.visited[1], axis=2) - np.sum(self.visited_old[1], axis=2) + 1))

            figure.savefig('%s/%i.png' % (path, e))
            plt.close(figure)

            self.visited_old = [v.copy() for v in self.visited]
        elif self.env_n_dim == 1:
            figure = plt.figure(figsize=(16, 10))

            ax1 = figure.add_subplot(2, 1, 1)
            ax2 = figure.add_subplot(2, 1, 2)

            ax1.imshow(np.log(self.show_visited + 1))
            ax2.imshow(np.log(self.show_visited - self.show_visited_old + 1))

            figure.savefig('%s/%i.png' % (path, e))
            plt.close(figure)

            self.visited_old = [v.copy() for v in self.visited]
            self.show_visited_old = self.show_visited.copy()


class Cen:
    def __init__(self, size, n_agent, args):
        self.env_n_dim = args.env_n_dim
        self.size = size
        self.n_agent = n_agent
        self.args = args
        self.visited = np.zeros([args.size, args.size, args.size, args.size])

    def update(self, state):
        self.visited[state[0][0]][state[0][1]][state[1][0]][state[1][1]] += 1

    def output(self, state):
        return 1. / np.sqrt(self.visited[state[0][0]][state[0][1]][state[1][0]][state[1][1]] + 1)

    def show(self, path, e):
        pass


class C_points:

    def __init__(self, size, a_size, arg, is_print):

        self.arg = arg
        self.is_print = is_print

        if (self.is_print):
            self.figure_path = self.arg.save_path + 'sub-goals/'
            if os.path.exists(self.figure_path):
                shutil.rmtree(self.figure_path)
            os.makedirs(self.figure_path)
        # print(self.figure_path)

        self.size = size
        self.a_size = a_size
        self.range = 3

        p_c_shape = add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

        self.p_c_1 = Count_p(3, p_c_shape, 1, 2)
        self.p_c_2 = Count_p(3, p_c_shape, 1, 2)

        p_shape = add_shape(2, size) + add_shape(1, a_size) + add_shape(2, self.range) + \
                  add_shape(2, size) + add_shape(1, a_size)

        self.p_1 = Count_p(5, p_shape, 1, 2)
        self.p_2 = Count_p(5, p_shape, 1, 2)

        other_p_shape = add_shape(2, size) + add_shape(1, a_size) + add_shape(2, self.range) + \
                        add_shape(2, size) + add_shape(1, a_size)

        self.other_p_1 = Count_p(5, other_p_shape, 1, 2)
        self.other_p_2 = Count_p(5, other_p_shape, 1, 2)

    def calc_del(self, s_t, next_s_t):
        return next_s_t - s_t + 1

    def update(self, state, next_state):

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2, s_t_2, a_t_2]
        c_num_2 = [del_s_t_1, s_t_1, a_t_1]

        self.p_c_1.add(c_num_1)
        self.p_c_2.add(c_num_2)

        p_num_1 = [s_t_1, a_t_1, del_s_t_2, s_t_2, a_t_2]
        p_num_2 = [s_t_2, a_t_2, del_s_t_1, s_t_1, a_t_1]

        self.p_1.add(p_num_1)
        self.p_2.add(p_num_2)

        other_p_num_1 = [s_t_2, a_t_2, del_s_t_2, s_t_1, a_t_1]
        other_p_num_2 = [s_t_1, a_t_1, del_s_t_1, s_t_2, a_t_2]

        self.other_p_1.add(other_p_num_1)
        self.other_p_2.add(other_p_num_2)

    def output(self, state, next_state, agent_index):

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2, s_t_2, a_t_2]
        c_num_2 = [del_s_t_1, s_t_1, a_t_1]

        if (self.arg.symmetry):
            p_1 = -self.p_c_1.output(c_num_1)
            p_2 = -self.p_c_2.output(c_num_2)
            return p_1 + p_2

        if (agent_index == 0):
            p_1 = self.p_c_1.output(c_num_1)
            return p_1
        else:
            p_2 = self.p_c_2.output(c_num_2)
            return p_2

    def show(self, e):

        if (self.is_print == 0):
            return
        print("start showing, round %d: " % e)

        ans_1 = np.zeros([self.size, self.size])
        ans_2 = np.zeros([self.size, self.size])
        res_1 = np.zeros([self.size, self.size])
        res_2 = np.zeros([self.size, self.size])

        for i1 in range(self.size):
            for j1 in range(self.size):
                for k1 in range(self.a_size):
                    for i3 in range(i1 - 1, i1 + 2):
                        for j3 in range(j1 - 1, j1 + 2):
                            c_num = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
                            p_1 = 1 - self.p_c_1.output(c_num)
                            p_2 = 1 - self.p_c_2.output(c_num)
                            if (p_1 < 1):
                                for i2 in range(self.size):
                                    for j2 in range(self.size):
                                        for k2 in range(self.a_size):
                                            p_num_1 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i1, j1) + mk1(k1)
                                            prob = self.p_1.output(p_num_1)

                                            p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i2, j2) + mk1(k2)
                                            prob_2 = self.other_p_1.output(p_num_2)

                                            ans_1[i2][j2] += prob * p_1
                                            res_2[i1][j1] += prob_2 * p_1

                            if (p_2 < 1):
                                for i2 in range(self.size):
                                    for j2 in range(self.size):
                                        for k2 in range(self.a_size):
                                            p_num_1 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i1, j1) + mk1(k1)
                                            prob = self.p_2.output(p_num_1)

                                            p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i2, j2) + mk1(k2)
                                            prob_2 = self.other_p_2.output(p_num_2)

                                            ans_2[i2][j2] += prob * p_2
                                            res_1[i1][j1] += prob_2 * p_2

        print("start drawing round %d" % e)

        figure = plt.figure(figsize=(16, 10))
        ax1 = figure.add_subplot(2, 2, 1)
        ax2 = figure.add_subplot(2, 2, 3)
        ax3 = figure.add_subplot(2, 2, 2)
        ax4 = figure.add_subplot(2, 2, 4)

        ax1.imshow(np.log(ans_1 + 1))
        ax2.imshow(np.log(ans_2 + 1))
        ax3.imshow(np.log(res_1 + 1))
        ax4.imshow(np.log(res_2 + 1))

        figure.savefig('%s/%i.png' % (self.figure_path, e))
        plt.close(figure)


class Island_C_points:

    def __init__(self, size, a_size, arg, is_print):

        self.arg = arg
        self.is_print = is_print

        if (self.is_print):
            self.figure_path = self.arg.save_path + 'sub-goals/'
            if os.path.exists(self.figure_path):
                shutil.rmtree(self.figure_path)
            os.makedirs(self.figure_path)
        # print(self.figure_path)

        self.size = size
        self.a_size = a_size
        self.range = 5

        p_c_shape = add_shape(4, self.range) + add_shape(4, size) + add_shape(1, a_size)

        self.p_c_1 = Count_p(3, p_c_shape, 1, 4, len=4)
        self.p_c_2 = Count_p(3, p_c_shape, 1, 4, len=4)

    def calc_del(self, s_t, next_s_t):
        return np.minimum(np.maximum(next_s_t - s_t, -2), 2) + 2

    def update(self, state, next_state):

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2, s_t_2, a_t_2]
        c_num_2 = [del_s_t_1, s_t_1, a_t_1]

        self.p_c_1.add(c_num_1)
        self.p_c_2.add(c_num_2)

    def output(self, state, next_state, agent_index):

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2, s_t_2, a_t_2]
        c_num_2 = [del_s_t_1, s_t_1, a_t_1]

        if (self.arg.symmetry):
            p_1 = -self.p_c_1.output(c_num_1)
            p_2 = -self.p_c_2.output(c_num_2)
            return p_1 + p_2

        if (agent_index == 0):
            p_1 = self.p_c_1.output(c_num_1)
            return p_1
        else:
            p_2 = self.p_c_2.output(c_num_2)
            return p_2

    def show(self, e):

        pass


class Island_Dec:

    def __init__(self, size, size_2, n_agent, args):
        self.env_n_dim = args.env_n_dim
        self.size = size
        self.n_agent = n_agent
        self.args = args

        self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'cen'

        if self.not_run:
            return

        self.visited = [np.zeros([size, size, 2]) for _ in range(self.n_agent)]
        self.visited_old = [np.zeros([size, size, 2]) for _ in range(self.n_agent)]
        self.visited_dec = [np.zeros([size, size, size_2, size, size]) for _ in range(self.n_agent)]

    def is_under_attack(self, state):
        if state[3] == 0:
            return 0
        s_t = state[:2]
        s_e = state[3: 5]
        del_s_t = s_t - s_e
        if -1 <= del_s_t[0] <= 1 and -1 <= del_s_t[1] <= 1:
            return 1
        else:
            return 0

    def update(self, state):

        if self.not_run:
            return

        # For heat map
        for i in range(self.n_agent):
            t_attack = self.is_under_attack(state[i])
            self.visited[i][state[i][0]][state[i][1]][t_attack] += 1
            self.visited_dec[i][state[i][0]][state[i][1]][state[i][2]][state[i][3]][state[i][4]] += 1

    def output(self, state, i):

        if self.not_run:
            return 0

        return 1. / np.sqrt(self.visited_dec[i][state[i][0]][state[i][1]][state[i][2]]
                            [state[i][3]][state[i][4]] + 1)

    def show(self, path, e):

        if self.not_run:
            return

        figure = plt.figure(figsize=(16, 10))

        ax1 = figure.add_subplot(2, 6, 1)
        ax2 = figure.add_subplot(2, 6, 2)
        ax3 = figure.add_subplot(2, 6, 3)
        ax4 = figure.add_subplot(2, 6, 4)
        ax5 = figure.add_subplot(2, 6, 5)
        ax6 = figure.add_subplot(2, 6, 6)
        ax7 = figure.add_subplot(2, 6, 7)
        ax8 = figure.add_subplot(2, 6, 8)
        ax9 = figure.add_subplot(2, 6, 9)
        ax10 = figure.add_subplot(2, 6, 10)
        ax11 = figure.add_subplot(2, 6, 11)
        ax12 = figure.add_subplot(2, 6, 12)

        ax1.imshow(np.log(self.visited[0][:, :, 0] + 1))
        ax2.imshow(np.log(self.visited[0][:, :, 0] - self.visited_old[0][:, :, 0] + 1))
        ax3.imshow(np.log(self.visited[0][:, :, 1] + 1))
        ax4.imshow(np.log((self.visited[0][:, :, 1] - self.visited_old[0][:, :, 1] + 1)))
        ax5.imshow(np.log(np.sum(self.visited[0], axis=2) + 1))
        ax6.imshow(np.log(np.sum(self.visited[0], axis=2) - np.sum(self.visited_old[0], axis=2) + 1))

        ax7.imshow(np.log(self.visited[1][:, :, 0] + 1))
        ax8.imshow(np.log(self.visited[1][:, :, 0] - self.visited_old[1][:, :, 0] + 1))
        ax9.imshow(np.log(self.visited[1][:, :, 1] + 1))
        ax10.imshow(np.log((self.visited[1][:, :, 1] - self.visited_old[1][:, :, 1] + 1)))
        ax11.imshow(np.log(np.sum(self.visited[1], axis=2) + 1))
        ax12.imshow(np.log(np.sum(self.visited[1], axis=2) - np.sum(self.visited_old[1], axis=2) + 1))

        figure.savefig('%s/%i.png' % (path, e))
        plt.close(figure)

        self.visited_old = [v.copy() for v in self.visited]


class Island_Cen:
    def __init__(self, size, size_2, n_agent, args):
        self.env_n_dim = args.env_n_dim
        self.size = size
        self.n_agent = n_agent
        self.args = args
        self.visited = np.zeros([size, size, size, size, size, size])

    def update(self, state):
        self.visited[state[0][0]][state[0][1]][state[0][3]][state[0][4]][state[1][0]][state[1][1]] += 1

    def output(self, state):
        return 1. / np.sqrt(self.visited[state[0][0]][state[0][1]][state[0][3]][state[0][4]]
                            [state[1][0]][state[1][1]] + 1)

    def show(self, path, e):
        pass


class x_Island_Cen:
    def __init__(self, size, size_2, n_agent, args):

        self.env_n_dim = args.env_n_dim
        self.size = size
        self.n_agent = n_agent
        self.args = args

        self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'coor_t' or \
                       self.args.s_alg_name == 'coor_r_tv' or self.args.s_alg_name == 'dec'

        if self.not_run:
            return

        self.visited = [[np.zeros([size, size, size, size, size, size]) for i in range(self.n_agent)]
                        for j in range(self.n_agent)]

    def update(self, state):

        if self.not_run:
            return

        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i != j:
                    self.visited[i][j][state[i][0]][state[i][1]][state[i][3]][state[i][4]][state[j][0]][state[j][1]] \
                        += 1

    def output(self, state, i):

        if self.not_run:
            return 0

        res = 0
        for j in range(self.n_agent):
            if i != j:
                res += 1. / np.sqrt(self.visited[i][j][state[i][0]][state[i][1]][state[i][3]][state[i][4]]
                                    [state[j][0]][state[j][1]] + 1)
        res /= self.n_agent - 1
        return res

    def show(self, path, e):
        pass


class Pushball_Dec:

    def __init__(self, size, n_agent, args):
        self.env_n_dim = args.env_n_dim
        self.size = size
        self.n_agent = n_agent
        self.args = args
        self.visited = [np.zeros([size, size]) for _ in range(self.n_agent)]
        self.visited_old = [np.zeros([size, size]) for _ in range(self.n_agent)]
        self.visited_dec = [np.zeros([size, size, size, size]) for _ in range(self.n_agent)]

    def update(self, state):
        # For heat map
        for i in range(self.n_agent):
            self.visited[i][state[i][0]][state[i][1]] += 1
            self.visited_dec[i][state[i][0]][state[i][1]][state[i][2]][state[i][3]] += 1

    def output(self, state, i):
        return 1. / np.sqrt(self.visited_dec[i][state[i][0]][state[i][1]][state[i][2]][state[i][3]] + 1)

    def show(self, path, e):
        if self.env_n_dim == 2:
            figure = plt.figure(figsize=(16, 10))

            ax1 = figure.add_subplot(2, 2, 1)
            ax2 = figure.add_subplot(2, 2, 2)
            ax3 = figure.add_subplot(2, 2, 3)
            ax4 = figure.add_subplot(2, 2, 4)

            ax1.imshow(np.log(self.visited[0] + 1))
            ax2.imshow(np.log(self.visited[0] - self.visited_old[0] + 1))

            ax3.imshow(np.log(self.visited[1] + 1))
            ax4.imshow(np.log(self.visited[1] - self.visited_old[1] + 1))

            figure.savefig('%s/%i.png' % (path, e))
            plt.close(figure)

            self.visited_old = [v.copy() for v in self.visited]


class Pushball_Cen:
    def __init__(self, size, n_agent, args):
        self.env_n_dim = args.env_n_dim
        self.size = size
        self.n_agent = n_agent
        self.args = args
        self.visited = np.zeros([size, size, size, size, size, size])

    def update(self, state):
        self.visited[state[0][0]][state[0][1]][state[0][2]][state[0][3]][state[1][0]][state[1][1]] += 1

    def output(self, state):
        return 1. / np.sqrt(self.visited[state[0][0]][state[0][1]][state[0][2]][state[0][3]]
                            [state[1][0]][state[1][1]] + 1)

    def show(self, path, e):
        pass


class Appro_C_points:

    def __init__(self, size, a_size, arg, is_print):

        self.arg = arg
        self.is_print = is_print and not self.arg.s_data_gather

        self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'dec' or \
                       self.arg.s_alg_name == 'cen'
        if self.not_run:
            return

        if (self.is_print):
            self.figure_path = self.arg.save_path + 'sub-goals/'
            if os.path.exists(self.figure_path):
                shutil.rmtree(self.figure_path)
            os.makedirs(self.figure_path)
        # print(self.figure_path)

        self.size = size
        self.a_size = a_size
        self.range = 3

        p_c_shape = \
            add_shape(2, self.range) + add_shape(2, size) + \
            add_shape(1, a_size) + add_shape(2, size) + add_shape(1, a_size)

        self.p_c_1 = Count_div_p(5, p_c_shape, 2, 1, 3, 2)
        self.p_c_2 = Count_div_p(5, p_c_shape, 2, 1, 3, 2)

        if self.is_print:
            p_shape = \
                add_shape(2, size) + add_shape(1, a_size) + \
                add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

            self.p_1 = Count_p(5, p_shape, 1, 2)
            self.p_2 = Count_p(5, p_shape, 1, 2)

            other_p_shape = \
                add_shape(2, size) + add_shape(1, a_size) + \
                add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

            self.other_p_1 = Count_p(5, other_p_shape, 1, 2)
            self.other_p_2 = Count_p(5, other_p_shape, 1, 2)

            check_p_shape = add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

            self.check_p_1 = Count(3, check_p_shape)
            self.check_p_2 = Count(3, check_p_shape)

    def calc_del(self, s_t, next_s_t):
        return next_s_t - s_t + 1

    def update(self, state, next_state):

        if self.not_run:
            return

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2, s_t_2, a_t_2, s_t_1, a_t_1]
        c_num_2 = [del_s_t_1, s_t_1, a_t_1, s_t_2, a_t_2]

        self.p_c_1.add(c_num_1)
        self.p_c_2.add(c_num_2)

        if self.is_print:
            p_num_1 = [s_t_1, a_t_1, del_s_t_2, s_t_2, a_t_2]
            p_num_2 = [s_t_2, a_t_2, del_s_t_1, s_t_1, a_t_1]

            self.p_1.add(p_num_1)
            self.p_2.add(p_num_2)

            other_p_num_1 = [s_t_2, a_t_2, del_s_t_2, s_t_1, a_t_1]
            other_p_num_2 = [s_t_1, a_t_1, del_s_t_1, s_t_2, a_t_2]

            self.other_p_1.add(other_p_num_1)
            self.other_p_2.add(other_p_num_2)

            check_p_num_1 = [del_s_t_1, s_t_1, a_t_1]
            check_p_num_2 = [del_s_t_2, s_t_2, a_t_2]

            self.check_p_1.add(check_p_num_1)
            self.check_p_2.add(check_p_num_2)

    def output(self, state, next_state, agent_index):

        if self.not_run:
            return 0

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2, s_t_2, a_t_2, s_t_1, a_t_1]
        c_num_2 = [del_s_t_1, s_t_1, a_t_1, s_t_2, a_t_2]

        if (self.arg.symmetry):
            p_1 = 1 - self.p_c_1.output(c_num_1)
            p_2 = 1 - self.p_c_2.output(c_num_2)
            return p_1 + p_2

        if (agent_index == 0):
            p_1 = 1 - self.p_c_1.output(c_num_1)
            return p_1
        else:
            p_2 = 1 - self.p_c_2.output(c_num_2)
            return p_2

    def show(self, e):

        if not self.is_print or self.not_run:
            return

        print("start showing, round %d: " % e)

        ans_1 = np.zeros([self.size, self.size])
        ans_2 = np.zeros([self.size, self.size])
        res_1 = np.zeros([self.size, self.size])
        res_2 = np.zeros([self.size, self.size])

        for i1 in range(self.size):
            for j1 in range(self.size):
                for k1 in range(self.a_size):
                    for i3 in range(i1 - 1, i1 + 2):
                        for j3 in range(j1 - 1, j1 + 2):
                            check_p_num_2 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
                            if self.check_p_2.output(check_p_num_2) > 0:
                                for i2 in range(self.size):
                                    for j2 in range(self.size):
                                        for k2 in range(self.a_size):
                                            p_num_1 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i1, j1) + mk1(k1)
                                            prob = self.p_1.output(p_num_1)

                                            num_1 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1) + \
                                                    mk(i2, j2) + mk1(k2)
                                            p_1 = 1 - self.p_c_1.output(num_1)

                                            p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i2, j2) + mk1(k2)
                                            prob_2 = self.other_p_1.output(p_num_2)

                                            ans_1[i2][j2] += prob * p_1
                                            res_2[i1][j1] += prob_2 * p_1

                            check_p_num_1 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
                            if self.check_p_1.output(check_p_num_1) > 0:
                                for i2 in range(self.size):
                                    for j2 in range(self.size):
                                        for k2 in range(self.a_size):
                                            p_num_2 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i1, j1) + mk1(k1)
                                            prob = self.p_2.output(p_num_2)

                                            num_2 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1) + \
                                                    mk(i2, j2) + mk1(k2)
                                            p_2 = 1 - self.p_c_2.output(num_2)

                                            p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
                                                      mk(i2, j2) + mk1(k2)
                                            prob_2 = self.other_p_2.output(p_num_2)

                                            ans_2[i2][j2] += prob * p_2
                                            res_1[i1][j1] += prob_2 * p_2

        print("start drawing round %d" % e)

        figure = plt.figure(figsize=(16, 10))
        ax1 = figure.add_subplot(2, 2, 1)
        ax2 = figure.add_subplot(2, 2, 3)
        ax3 = figure.add_subplot(2, 2, 2)
        ax4 = figure.add_subplot(2, 2, 4)

        ax1.imshow(np.log(ans_1 + 1))
        ax2.imshow(np.log(ans_2 + 1))
        ax3.imshow(np.log(res_1 + 1))
        ax4.imshow(np.log(res_2 + 1))

        figure.savefig('%s/%i.png' % (self.figure_path, e))
        plt.close(figure)


class Pushball_Appro_C_points:

    def __init__(self, size, a_size, arg, is_print):

        self.arg = arg
        self.is_print = is_print

        self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'dec' or \
                       self.arg.s_alg_name == 'cen'
        if self.not_run:
            return

        if (self.is_print):
            self.figure_path = self.arg.save_path + 'sub-goals/'
            if os.path.exists(self.figure_path):
                shutil.rmtree(self.figure_path)
            os.makedirs(self.figure_path)
        # print(self.figure_path)

        self.size = size
        self.a_size = a_size
        self.range = 3

        p_c_shape = \
            add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size) + add_shape(2, size) + \
            add_shape(2, size) + add_shape(1, a_size)

        self.p_c_1 = Count_div_p(6, p_c_shape, 2, 1, 3, 2)
        self.p_c_2 = Count_div_p(6, p_c_shape, 2, 1, 3, 2)

    def calc_del(self, s_t, next_s_t):
        return next_s_t - s_t + 1

    def update(self, state, next_state):

        if self.not_run:
            return

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        s_e = s_t_1[2:]

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2, s_t_2, a_t_2, s_e, s_t_1, a_t_1]
        c_num_2 = [del_s_t_1, s_t_1, a_t_1, s_e, s_t_2, a_t_2]

        self.p_c_1.add(c_num_1)
        self.p_c_2.add(c_num_2)

    def output(self, state, next_state, agent_index):

        if self.not_run:
            return 0

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        s_e = s_t_1[2:]

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2, s_t_2, a_t_2, s_e, s_t_1, a_t_1]
        c_num_2 = [del_s_t_1, s_t_1, a_t_1, s_e, s_t_2, a_t_2]

        if (self.arg.symmetry):
            p_1 = 1 - self.p_c_1.output(c_num_1)
            p_2 = 1 - self.p_c_2.output(c_num_2)
            return p_1 + p_2

        if (agent_index == 0):
            p_1 = 1 - self.p_c_1.output(c_num_1)
            return p_1
        else:
            p_2 = 1 - self.p_c_2.output(c_num_2)
            return p_2

    def show(self, e):
        pass


class Island_Appro_C_points:

    def __init__(self, size, a_size, agent_power_size, wolf_power_size, arg, is_print):

        self.arg = arg
        self.is_print = is_print

        self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'cen' or self.arg.s_alg_name == 'dec'

        if self.not_run:
            return

        if (self.is_print):
            self.figure_path = self.arg.save_path + 'sub-goals/'
            if os.path.exists(self.figure_path):
                shutil.rmtree(self.figure_path)
            os.makedirs(self.figure_path)
        # print(self.figure_path)

        self.size = size
        self.a_size = a_size
        self.agent_power_size = agent_power_size
        self.wolf_power_size = wolf_power_size
        self.range = 3

        p_c_shape = \
            add_shape(2, self.range) + add_shape(1, 3) + \
            add_shape(2, size) + add_shape(1, agent_power_size) + add_shape(1, a_size) + \
            add_shape(2, size) + \
            add_shape(2, size)

        self.p_c_1 = Count_div_p(7, p_c_shape, 1, 2, 2, 3)
        self.p_c_2 = Count_div_p(7, p_c_shape, 1, 2, 2, 3)

    def calc_del(self, s_t, next_s_t):
        del_s_t = next_s_t - s_t
        del_s_t[:2] += 1
        del_s_t[2] += 2
        return del_s_t

    def update(self, state, next_state):
        if self.not_run:
            return

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        s_wolf = s_t_1[3:]

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2[:2], del_s_t_2[2], s_t_2[:2], s_t_2[2], a_t_2, s_wolf, s_t_1]
        c_num_2 = [del_s_t_1[:2], del_s_t_1[2], s_t_1[:2], s_t_1[2], a_t_1, s_wolf, s_t_2]

        self.p_c_1.add(c_num_1)
        self.p_c_2.add(c_num_2)

    def output(self, state, next_state, agent_index):
        if self.not_run:
            return 0

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        s_wolf = s_t_1[3:]

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2[:2], del_s_t_2[2], s_t_2[:2], s_t_2[2], a_t_2, s_wolf, s_t_1]
        c_num_2 = [del_s_t_1[:2], del_s_t_1[2], s_t_1[:2], s_t_1[2], a_t_1, s_wolf, s_t_2]

        if (self.arg.symmetry):
            p_1 = 1 - self.p_c_1.output(c_num_1)
            p_2 = 1 - self.p_c_2.output(c_num_2)
            return p_1 + p_2

        if (agent_index == 0):
            p_1 = 1 - self.p_c_1.output(c_num_1)
            return p_1
        else:
            p_2 = 1 - self.p_c_2.output(c_num_2)
            return p_2

    def show(self, e):
        pass


class Test_Island_Appro_C_points:

    def __init__(self, size, a_size, agent_power_size, wolf_power_size, arg, is_print):

        self.arg = arg
        self.is_print = is_print

        self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'cen'

        if self.not_run:
            return

        if (self.is_print):
            self.figure_path = self.arg.save_path + 'sub-goals-test/'
            if os.path.exists(self.figure_path):
                shutil.rmtree(self.figure_path)
            os.makedirs(self.figure_path)
        # print(self.figure_path)

        self.size = size
        self.a_size = a_size
        self.agent_power_size = agent_power_size
        self.wolf_power_size = wolf_power_size
        self.range = 3

        p_c_shape = \
            add_shape(2, self.range) + add_shape(1, 3) + \
            add_shape(2, size) + add_shape(1, agent_power_size) + add_shape(1, a_size) + \
            add_shape(2, size) + \
            add_shape(2, size)

        self.p_c_1 = Test_Count_div_p(7, p_c_shape, 1, 2, 2, 3)
        self.p_c_2 = Test_Count_div_p(7, p_c_shape, 1, 2, 2, 3)

    def calc_del(self, s_t, next_s_t):
        del_s_t = next_s_t - s_t
        del_s_t[:2] += 1
        del_s_t[2] += 2
        return del_s_t

    def update(self, state, next_state):
        if self.not_run:
            return

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        s_wolf = s_t_1[3:]

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2[:2], del_s_t_2[2], s_t_2[:2], s_t_2[2], a_t_2, s_wolf, s_t_1]
        c_num_2 = [del_s_t_1[:2], del_s_t_1[2], s_t_1[:2], s_t_1[2], a_t_1, s_wolf, s_t_2]

        self.p_c_1.add(c_num_1)
        self.p_c_2.add(c_num_2)

    def output(self, state, next_state, agent_index):
        if self.not_run:
            return 0

        s_t, a_t = state
        s_t_1, s_t_2 = s_t
        a_t_1, a_t_2 = a_t

        s_wolf = s_t_1[3:]

        next_s_t = next_state[0]
        next_s_t_1, next_s_t_2 = next_s_t

        del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
        del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

        c_num_1 = [del_s_t_2[:2], del_s_t_2[2], s_t_2[:2], s_t_2[2], a_t_2, s_wolf, s_t_1]
        c_num_2 = [del_s_t_1[:2], del_s_t_1[2], s_t_1[:2], s_t_1[2], a_t_1, s_wolf, s_t_2]

        if agent_index == 0:
            ans, ans_1, ans_2 = self.p_c_1.output(c_num_1)
            p_1 = 1 - ans
            return p_1, ans_1, ans_2
        else:
            ans, ans_1, ans_2 = self.p_c_2.output(c_num_2)
            p_2 = 1 - ans
            return p_2, ans_1, ans_2

    def show(self, e):
        pass


class VI:

    def __init__(self, goal_range, input_range, s_1, s_2, name):
        self.s_1 = s_1
        self.s_2 = s_2
        self.name = name

        self.input_range = input_range
        self.goal_range = goal_range

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_network()

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

    def create_network(self):
        self.data_input = tf.placeholder(tf.float32, shape=[None, self.input_range])
        self.labels = tf.placeholder(tf.int32, shape=[None, ])
        self.num_data = tf.placeholder(tf.float32, shape=[1])

        with tf.device('gpu:0/'):
            self.model = tf.keras.Sequential([
                tfp.layers.DenseReparameterization(64, activation=tf.nn.relu),
                tfp.layers.DenseReparameterization(64, activation=tf.nn.relu),
                tfp.layers.DenseReparameterization(self.goal_range),
            ])
            self.logits = self.model(self.data_input)
            self.output = tf.nn.softmax(self.logits)

            neg_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.logits)
            self.log_loss = tf.reduce_mean(neg_log_likelihood)
            self.kl = 1. * sum(self.model.losses) / self.num_data
            self.loss = self.log_loss + self.kl
            self.train_op = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)

    def run(self, labels, feature):
        with self.graph.as_default():
            res = self.sess.run(self.output, feed_dict={self.data_input: feature})
            res = np.take(res, labels)
        return res

    def update(self, labels, feature):
        num_data = labels.shape[0]
        with self.graph.as_default():
            _, loss = \
                self.sess.run([self.train_op, self.log_loss],
                              feed_dict={self.data_input: feature,
                                         self.labels: labels,
                                         self.num_data: np.array([num_data])})
    # if self.name == 'p' and self.s_2 == 1:
    #    print(loss)


class Island_VI_Appro_C_points:

    def __init__(self, size, a_size, agent_power_size, wolf_power_size, args, is_print):

        self.args = args
        self.is_print = is_print

        self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'cen' or self.args.s_alg_name == 'dec'

        if self.not_run:
            return

        if self.is_print:
            self.figure_path = self.args.save_path + 'sub-goals/'
            if os.path.exists(self.figure_path):
                shutil.rmtree(self.figure_path)
            os.makedirs(self.figure_path)
        # print(self.figure_path)

        self.n_agent = self.args.n_agent
        self.size = size
        self.a_size = a_size
        self.agent_power_size = agent_power_size
        self.wolf_power_size = wolf_power_size

        self.xy_range = 5
        self.harm_range = self.args.x_island_harm_range
        # self.goal_range = 55
        self.goal_range = self.xy_range * self.harm_range
        # self.input_range_p = 349
        self.input_range_p = (self.size * 2 + self.agent_power_size + self.a_size) * (self.n_agent - 1) + \
                             self.size * 2
        # self.input_range_t = 426
        self.input_range_t = self.input_range_p + self.size * 2 + self.agent_power_size + self.a_size

        print('goal_range', self.goal_range, 'input_range_p', self.input_range_p, 'input_range_t', self.input_range_t)

        self.eye_size = np.eye(self.size)
        self.eye_action = np.eye(self.a_size)
        self.eye_agent_power = np.eye(self.agent_power_size)
        self.eye_wolf_power = np.eye(self.args.x_island_wolf_max_power)

        self.appro_T = self.args.appro_T

        self.batch_size = 2048
        self.collect = 0
        self.collect_label = []
        self.collect_p = []
        self.collect_t = []

        self.model_p = []
        self.model_t = []
        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i != j:
                    self.model_p.append(VI(self.goal_range, self.input_range_p, j, i, 'p'))
                    self.model_t.append(VI(self.goal_range, self.input_range_t, j, i, 't'))

    def calc_del(self, s_t, next_s_t):
        del_s_t = next_s_t - s_t
        next_s = 0
        if del_s_t[0] == 0 and del_s_t[0] == 0:
            next_s = 0
        elif del_s_t[0] == 0 and del_s_t[0] == -1:
            next_s = 1
        elif del_s_t[0] == 0 and del_s_t[0] == 1:
            next_s = 2
        elif del_s_t[0] == -1 and del_s_t[0] == 0:
            next_s = 3
        elif del_s_t[0] == 1 and del_s_t[0] == 0:
            next_s = 4
        del_s_t[2] += self.harm_range - 1
        next_s += del_s_t[2] * self.xy_range
        return next_s

    def make(self, state, next_state):
        s_t, a_t = state
        next_s_t = next_state[0]

        s_e = s_t[0][3:]

        v_del_i = []
        v_s_t_i = []
        v_a_t_i = []
        for i in range(self.n_agent):
            v_del_i.append(self.calc_del(s_t[i], next_s_t[i]))
            v_state = np.concatenate([self.eye_size[s_t[i][0]], self.eye_size[s_t[i][1]],
                                      self.eye_agent_power[s_t[i][2]]], axis=0)
            v_s_t_i.append(v_state)
            v_a_t_i.append(self.eye_action[a_t[i]])

        v_s_e = np.concatenate([self.eye_size[s_e[0]], self.eye_size[s_e[1]]], axis=0)

        p_num_label = []
        p_num_x_p = []
        p_num_x_t = []
        for i in range(self.n_agent):
            y = v_del_i[i]
            for j in range(self.n_agent):
                if i != j:
                    x_p = []
                    for k in range(self.n_agent):
                        if j != k:
                            x_p.append(np.concatenate([v_s_t_i[k], v_a_t_i[k]], axis=0))
                    x_p = np.concatenate(x_p + [v_s_e], axis=0)
                    x_t = np.concatenate([x_p, v_s_t_i[j], v_a_t_i[j]], axis=0)
                    p_num_label.append(y)
                    p_num_x_p.append(x_p)
                    p_num_x_t.append(x_t)

        return p_num_label, p_num_x_p, p_num_x_t

    def update(self):

        labels = np.concatenate(self.collect_label, axis=1)
        feature_p = np.concatenate(self.collect_p, axis=1)
        feature_t = np.concatenate(self.collect_t, axis=1)

        t_stamp = 0
        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i != j:
                    self.model_p[t_stamp].update(labels[t_stamp], feature_p[t_stamp])
                    self.model_t[t_stamp].update(labels[t_stamp], feature_t[t_stamp])
                    t_stamp += 1

        self.collect = 0
        self.collect_label = []
        self.collect_p = []
        self.collect_t = []

    def output(self, label, x_p, x_t):

        self.collect_label.append(label)
        self.collect_p.append(x_p)
        self.collect_t.append(x_t)

        num_data = label.shape[1]
        self.collect += num_data
        if self.collect >= self.batch_size:
            self.update()

        coor_rewards = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
        coor_p = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
        coor_t = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

        t_stamp = 0
        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i != j:
                    prob_p = self.model_p[t_stamp].run(label[t_stamp], x_p[t_stamp])
                    prob_t = self.model_t[t_stamp].run(label[t_stamp], x_t[t_stamp])
                    t_stamp += 1
                    coor_p[j][i] = prob_p
                    coor_t[j][i] = prob_t
                    coor_rewards[j][i] = 1. - prob_p / np.maximum(prob_t, self.appro_T)

        return coor_rewards, coor_p, coor_t

    def show(self, e):
        pass


class Visual_coor_rew:

    def __init__(self, size, n_agent, args, num_envs, name, is_print=True):
        self.env_n_dim = args.env_n_dim
        self.size = size
        self.n_agent = n_agent
        self.args = args
        self.num_envs = num_envs

        self.is_print = is_print

        self.visited = [np.zeros([2, args.size, args.size]) for _ in range(self.n_agent)]
        self.visited_old = [np.zeros([2, args.size, args.size]) for _ in range(self.n_agent)]
        self.value = [np.zeros([2, args.size, args.size]) for _ in range(self.n_agent)]
        self.value_old = [np.zeros([2, args.size, args.size]) for _ in range(self.n_agent)]

        self.figure_path = self.args.save_path + 'sub-goals-%s/' % name
        if os.path.exists(self.figure_path):
            shutil.rmtree(self.figure_path)
        os.makedirs(self.figure_path)

        self.e_step = 0

    def update(self, infos, dones, coor_rewards):

        if self.args.s_data_gather or not self.is_print:
            return

        for i in range(self.num_envs):
            pre_state = infos[i]['pre_state']
            if (pre_state != None):
                for j in range(self.n_agent):
                    self.value[j][0][pre_state[j][0]][pre_state[j][1]] += coor_rewards[j][i]
                    self.visited[j][0][pre_state[j][0]][pre_state[j][1]] += 1
                    self.value[1 - j][1][pre_state[1 - j][0]][pre_state[1 - j][1]] += coor_rewards[j][i]
                    self.visited[1 - j][1][pre_state[1 - j][0]][pre_state[1 - j][1]] += 1
            if dones[i]:
                self.e_step += 1
                if self.e_step % (10000 * self.args.t_save_rate) == 0:
                    self.show(self.e_step)

    def normal(self, data, total):
        t_data = data.copy()
        t_total = total.copy()
        zero = total == 0
        t_total[zero] = 1
        t_data = t_data / t_total
        t_min = np.min(t_data)
        t_max = np.max(t_data)
        t_data[zero] = t_min
        t_data = (t_data - t_min) / (t_max - t_min + 1)
        return np.log(t_data + 1)

    def show(self, e):

        if self.args.s_data_gather or not self.is_print:
            return

        figure = plt.figure(figsize=(16, 10))

        ax1 = figure.add_subplot(2, 4, 1)
        ax2 = figure.add_subplot(2, 4, 2)
        ax3 = figure.add_subplot(2, 4, 3)
        ax4 = figure.add_subplot(2, 4, 4)
        ax5 = figure.add_subplot(2, 4, 5)
        ax6 = figure.add_subplot(2, 4, 6)
        ax7 = figure.add_subplot(2, 4, 7)
        ax8 = figure.add_subplot(2, 4, 8)

        ax1.imshow(self.normal(self.value[0][0], self.visited[0][0]))
        ax2.imshow(self.normal(self.value[0][0] - self.value_old[0][0],
                               self.visited[0][0] - self.visited_old[0][0]))
        ax3.imshow(self.normal(self.value[0][1], self.visited[0][1]))
        ax4.imshow(self.normal(self.value[0][1] - self.value_old[0][1],
                               self.visited[0][1] - self.visited_old[0][1]))

        ax5.imshow(self.normal(self.value[1][0], self.visited[1][0]))
        ax6.imshow(self.normal(self.value[1][0] - self.value_old[1][0],
                               self.visited[1][0] - self.visited_old[1][0]))
        ax7.imshow(self.normal(self.value[1][1], self.visited[1][1]))
        ax8.imshow(self.normal(self.value[1][1] - self.value_old[1][1],
                               self.visited[1][1] - self.visited_old[1][1]))

        figure.savefig('%s/%i.png' % (self.figure_path, e))
        plt.close(figure)

        self.visited_old = [v.copy() for v in self.visited]
        self.value_old = [v.copy() for v in self.value]
