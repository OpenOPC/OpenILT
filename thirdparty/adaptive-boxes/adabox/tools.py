import numpy as np
from scipy import stats
import json
import pandas as pd

# from ard_lib.two_dim.ard_2d_interface import InterfaceUnit


def is_broken(vector_to_test, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    it_is = False

    for i in range(len(vector_to_test) - 1):
        diff_val = abs(vector_to_test[i] - vector_to_test[i + 1])
        if diff_val <= error_ratio_sup:
            if diff_val >= error_ratio_inf:
                it_is = False
            else:
                it_is = True
                break
        else:
            # print('less than')
            it_is = True
            break

    return it_is


def get_dist_left(all_x_points_arg, init_x_index_arg, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    # Left
    l_lim = 0
    index_val = init_x_index_arg
    while index_val > l_lim:
        # print(index)
        diff_bound_val = abs(all_x_points_arg[index_val, 0] - all_x_points_arg[index_val - 1, 0])
        # print(diff_bound)
        if diff_bound_val >= error_ratio_sup or diff_bound_val <= error_ratio_inf:
            break
        index_val = index_val - 1

    f_index_l_val = index_val
    dist_l_val = init_x_index_arg - f_index_l_val
    return dist_l_val


def get_dist_right(all_x_points_arg, init_x_index_arg, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    # Right
    r_lim = len(all_x_points_arg) - 1
    index_val = init_x_index_arg
    while index_val < r_lim:
        # print(index)
        diff_bound = abs(all_x_points_arg[index_val, 0] - all_x_points_arg[index_val + 1, 0])
        # print(diff_bound)
        if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
            break
        index_val = index_val + 1

    f_index_r_val = index_val + 1
    dist_r_val = f_index_r_val - init_x_index_arg
    return dist_r_val


def get_dist_down(all_y_points_arg, init_y_index_arg, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    # Left
    d_lim = 0
    index_val = init_y_index_arg
    while index_val > d_lim:
        # print(index_val)
        diff_bound = abs(all_y_points_arg[index_val, 1] - all_y_points_arg[index_val - 1, 1])
        # print(diff_bound)
        if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
            break
        index_val = index_val - 1

    f_index_d_val = index_val
    dist_d_val = init_y_index_arg - f_index_d_val
    return dist_d_val


def get_dist_up(all_y_points_arg, init_y_index_arg, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    # Right
    u_lim = len(all_y_points_arg) - 1
    index_val = init_y_index_arg
    while index_val < u_lim:
        # print(index_val)
        diff_bound = abs(all_y_points_arg[index_val, 1] - all_y_points_arg[index_val + 1, 1])
        # print(diff_bound)
        if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
            break
        index_val = index_val + 1

    f_index_u_val = index_val + 1
    dist_u_val = f_index_u_val - init_y_index_arg
    return dist_u_val


def get_final_index_down(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
    # Down
    down_lim = 0
    index = init_y_index_arg
    while index >= down_lim:
        # print(index)
        temp_y = all_y_points_arg[index, 1]
        all_x_points_arg = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

        temp_x = all_y_points_arg[index, 0]
        temp_x_index = np.where(all_x_points_arg[:, 0] == temp_x)[0][0]

        index_lim_sup = temp_x_index + dist_r_arg
        index_lim_inf = temp_x_index - dist_l_arg

        if index_lim_inf < 0:
            index_lim_inf = 0

        if index_lim_sup > len(all_x_points_arg):
            index_lim_sup = len(all_x_points_arg)

        temp_range_lr = range(index_lim_inf, index_lim_sup)

        just_x = all_x_points_arg[temp_range_lr, 0]
        if is_broken(just_x, sep_value):
            break
        index = index - 1

    final_index_val = index + 1
    return final_index_val


def get_final_index_up(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
    # Up
    up_lim = len(all_y_points_arg) - 1
    index = init_y_index_arg
    while index <= up_lim:
        # print(index)
        temp_y = all_y_points_arg[index, 1]
        all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

        temp_x = all_y_points_arg[index, 0]
        temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

        index_lim_sup = temp_x_index + dist_r_arg
        index_lim_inf = temp_x_index - dist_l_arg

        if index_lim_inf < 0:
            index_lim_inf = 0

        if index_lim_sup > len(all_x_points):
            index_lim_sup = len(all_x_points)

        temp_range_lr = range(index_lim_inf, index_lim_sup)

        just_x = all_x_points[temp_range_lr, 0]
        if is_broken(just_x, sep_value):
            break
        index = index + 1

    final_index_val = index - 1
    return final_index_val


def get_final_xy_index_down(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
    # Down
    final_index = get_final_index_down(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value)

    # ---- last step
    temp_y = all_y_points_arg[final_index, 1]
    all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

    # ---- plot
    temp_x = all_y_points_arg[final_index, 0]
    temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

    index_lim_sup = temp_x_index + dist_r_arg
    index_lim_inf = temp_x_index - dist_l_arg

    if index_lim_inf < 0:
        index_lim_inf = 0

    if index_lim_sup > len(all_x_points):
        index_lim_sup = len(all_x_points)

    temp_range_lr = range(index_lim_inf, index_lim_sup)

    final_x_min = all_x_points[temp_range_lr, 0].min()
    final_x_max = all_x_points[temp_range_lr, 0].max()
    final_y_down = temp_y
    return final_x_min, final_x_max, final_y_down


def get_final_xy_index_up(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
    # Up
    final_index = get_final_index_up(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value)
    # ---- last step
    temp_y = all_y_points_arg[final_index, 1]
    all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

    # ---- plot
    temp_x = all_y_points_arg[final_index, 0]
    temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

    index_lim_sup = temp_x_index + dist_r_arg
    index_lim_inf = temp_x_index - dist_l_arg

    if index_lim_inf < 0:
        index_lim_inf = 0

    if index_lim_sup > len(all_x_points):
        index_lim_sup = len(all_x_points)

    temp_range_lr = range(index_lim_inf, index_lim_sup)

    final_x_min = all_x_points[temp_range_lr, 0].min()
    final_x_max = all_x_points[temp_range_lr, 0].max()
    final_y_up = temp_y
    return final_x_min, final_x_max, final_y_up


def get_separation_value(data_2d_global_arg):
    n_sample = 100
    x_data = np.unique(np.sort(data_2d_global_arg[:, 0]))
    y_data = np.unique(np.sort(data_2d_global_arg[:, 1]))

    diffs_x = np.zeros(shape=[n_sample])
    diffs_y = np.zeros(shape=[n_sample])

    for p in range(n_sample):
        x_rand_num = int(np.random.rand() * (len(x_data) - 1))
        y_rand_num = int(np.random.rand() * (len(y_data) - 1))
        # print(str(x_rand_num) + '  ' + str(y_rand_num))
        diffs_x[p] = np.abs(x_data[x_rand_num] - x_data[x_rand_num + 1])
        diffs_y[p] = np.abs(y_data[y_rand_num] - y_data[y_rand_num + 1])

    sep_value_val = (stats.mode(diffs_x, keepdims=True).mode[0] + stats.mode(diffs_y, keepdims=True).mode[0]) / 2
    # print(sep_value_val)
    return sep_value_val


def create_2d_data_from_vertex(vertex_2d_data):
    shape_vertex_data = vertex_2d_data.shape
    data_2d_global_val = np.zeros(shape=[shape_vertex_data[0], (shape_vertex_data[1] - 1) + 1])
    data_2d_global_val[:, [0, 1]] = np.array(vertex_2d_data.loc[:, ['x', 'y']])
    data_2d_global_val = np.unique(data_2d_global_val, axis=0)
    return data_2d_global_val


class Rectangle:

    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.a = abs(x2 - x1)
        self.b = abs(y2 - y1)

        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x1, y2])
        self.p3 = np.array([x2, y1])
        self.p4 = np.array([x2, y2])
    #
    # def get_area(self):
    #     return abs(self.x2 - self.x1) * abs(self.y2 - self.y1)

    def get_area(self):
        return self.a * self.b

    def get_side_ratio(self):
        if self.b == 0:
            return 0
        else:
            return self.a / self.b


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# array order: [x1, x2, y1, y2]
def save_to_json(path, array_to_save, sep_value):
    json_dump = json.dumps({'data': array_to_save, 'sep_value': sep_value}, cls=NumpyEncoder)
    outfile = open(path, 'w', encoding='utf-8')
    json.dump(json_dump, outfile, ensure_ascii=False, indent=2)


def load_from_json(path):
    file = open(path)
    json_str = json.load(file)
    json_format = json.loads(json_str)
    return json_format


# Returns array -> x1 x2 y1 y2 is_checked? gi gj (g:groups) and Summary [group_id, n_elements, diff_y, diff_x]
def create_groups(json_data_arg, sep_value_arg):
    data_shape_val = json_data_arg.shape
    data_prepros_val = np.zeros(shape=[data_shape_val[0], data_shape_val[1] + 5])

    # data_prepros: 0-3(x,y,z) 4(is checked?) 5(area) 6(ratio) 7(g_i) 8(g_j)

    sep = sep_value_arg / 2
    for i_d in range(len(json_data_arg)):
        data_prepros_val[i_d][0] = json_data_arg[i_d][0] - sep
        data_prepros_val[i_d][1] = json_data_arg[i_d][1] + sep
        data_prepros_val[i_d][2] = json_data_arg[i_d][2] - sep
        data_prepros_val[i_d][3] = json_data_arg[i_d][3] + sep

        data_prepros_val[i_d][4] = 0  # (is checked?) init in False

        # area (x2-x1) * (y2-y1)
        diff_x = abs(data_prepros_val[i_d][1] - data_prepros_val[i_d][0])
        diff_y = abs(data_prepros_val[i_d][3] - data_prepros_val[i_d][2])

        area = diff_x * diff_y

        # ratio (x2-x1) / (y2-y1)
        ratio = diff_x / diff_y

        data_prepros_val[i_d][5] = np.round(area, decimals=4)  # area
        data_prepros_val[i_d][6] = np.round(ratio, decimals=4)  # ratio

    #   Init groups
    data_prepros_pd = pd.DataFrame(data_prepros_val)
    data_prepros_pd.sort_values(by=5)
    data_groups = data_prepros_pd.groupby(by=5)

    gi_counter = 0
    summary_val = []
    for g in data_groups:
        # print('-> ' + str(g[0]))
        g_data = g[1]
        g_data_groups = g_data.groupby(by=6)
        for g_d in g_data_groups:
            # print('----> ' + str(g_d[0]))
            # print('--------------> ' + str(gi_counter))
            g_data_data = g_d[1]

            indexes = np.array(g_data_data.index)
            data_prepros_val[indexes, 7] = gi_counter
            data_prepros_val[indexes, 8] = list(range(len(indexes)))

            diff_x = abs(data_prepros_val[indexes[0], 1] - data_prepros_val[indexes[0], 0])
            diff_y = abs(data_prepros_val[indexes[0], 3] - data_prepros_val[indexes[0], 2])

            summary_val.append([gi_counter, len(indexes), diff_y, diff_x])

            gi_counter = gi_counter + 1

    result_data = data_prepros_val[:, [0, 1, 2, 3, 4, 7, 8]]
    return result_data, summary_val

#
# def search_lr_interfaces(data_prepros_arg, n_partition_arg, sep_value_arg, n_split_sep_value_arg, error_val_arg):
#
#     sep_value_split_arg = sep_value_arg / n_split_sep_value_arg  # greater than 2
#
#     test_partition_arg = data_prepros_arg[n_partition_arg]
#
#     error_sep_value = error_val_arg * sep_value_arg
#     error_sep_value_sup = 0 + error_sep_value
#     error_sep_value_inf = 0 - error_sep_value
#
#     x_units_val = []
#
#     # Iter 2 times because of x1 and x2 to find en left and right side of the partition
#     for it in range(2):
#
#         if it == 0:
#             side_temp = 'Left'
#         else:
#             side_temp = 'Right'
#
#         data_prepros_x = data_prepros_arg[:, 1 - it]
#         common_x_pos_arg = test_partition_arg[0 + it]
#
#         diffs = data_prepros_x - common_x_pos_arg
#         diffs_condition = (abs(diffs) <= error_sep_value_sup) & (abs(diffs) >= error_sep_value_inf)
#         location = np.where(diffs_condition)
#         partition_pos = location[0]
#
#         for p in partition_pos:
#             # print(p)
#             temp_partition = data_prepros_arg[p]
#
#             is_partition_checked = temp_partition[4]
#
#             if is_partition_checked == 0:
#                 y1_temp = np.array([test_partition_arg[2], temp_partition[2]]).max()
#                 y2_temp = np.array([test_partition_arg[3], temp_partition[3]]).min()
#
#                 # plt.scatter(common_x_pos_arg, y1_temp, marker='x')
#                 # plt.scatter(common_x_pos_arg, y2_temp, marker='x')
#
#                 # test if range is inside both partitions
#                 condition_1 = (y1_temp >= temp_partition[2]) & (y2_temp <= temp_partition[3])
#                 condition_2 = (y1_temp >= test_partition_arg[2]) & (y2_temp <= test_partition_arg[3])
#
#                 if condition_1 & condition_2:
#                     # index_distance = int(abs(y1_temp - y2_temp)/sep_value_split)
#
#                     # temp
#                     temp_list = []
#                     index_count_aux = int(np.round(abs(temp_partition[3] - temp_partition[2]) / sep_value_split_arg))
#                     for s in range(index_count_aux):
#                         # print(s)
#                         val = temp_partition[3] - sep_value_split_arg / 2 - s * sep_value_split_arg
#                         # plt.scatter(test_partition[1] + sep_value_split, val, marker='x', c='b')
#                         if (val > y1_temp) & (val < y2_temp):
#                             # plt.scatter(common_x_pos_arg + sep_value_split_arg / 2, val, marker='_', c='b')
#                             # print('Partition ' + str(temp_partition[5]) + ' ' + str(temp_partition[6])
#                             #       + ' index: ' + str(s))
#                             temp_list.append([temp_partition[5], temp_partition[6], s])
#
#                     # test
#                     test_list = []
#                     index_count_aux = int(np.round(abs(test_partition_arg[3] - test_partition_arg[2]) / sep_value_split_arg))
#                     for s in range(index_count_aux):
#                         val = test_partition_arg[3] - sep_value_split_arg / 2 - s * sep_value_split_arg
#                         # plt.scatter(test_partition[1] - sep_value_split, val, marker='x', c='r')
#
#                         if (val > y1_temp) & (val < y2_temp):
#                             # plt.scatter(common_x_pos_arg - sep_value_split_arg / 2, val, marker='_', c='r')
#                             # print('Partition ' + str(test_partition_arg[5]) + ' ' + str(test_partition_arg[6])
#                             #       + ' index: ' + str(s))
#                             test_list.append([test_partition_arg[5], test_partition_arg[6], s])
#
#                     for l in range(len(temp_list)):
#                         if it == 0:
#                             # Left
#                             l_part = temp_list[l]
#                             r_part = test_list[l]
#                         else:
#                             # Right
#                             l_part = test_list[l]
#                             r_part = temp_list[l]
#
#                         gi_l = int(l_part[0])
#                         gj_l = int(l_part[1])
#                         gl_index = int(l_part[2])
#
#                         gi_r = int(r_part[0])
#                         gj_r = int(r_part[1])
#                         gr_index = int(r_part[2])
#
#                         x_units_val.append(InterfaceUnit(((gi_l, gj_l), (gi_r, gj_r)), (gl_index, gr_index)))
#     return x_units_val

#
# def search_ud_interfaces(data_prepros_arg, n_partition_arg, sep_value_arg, n_split_sep_value_arg, error_val_arg):
#
#     sep_value_split_arg = sep_value_arg / n_split_sep_value_arg  # greater than 2
#
#     test_partition_arg = data_prepros_arg[n_partition_arg]
#
#     error_sep_value = error_val_arg * sep_value_arg
#     error_sep_value_sup = 0 + error_sep_value
#     error_sep_value_inf = 0 - error_sep_value
#
#     y_units_val = []
#
#     # Iter 2 times because of y1 and y2 to find in up and down side of the partition
#     for it in range(2):
#
#         if it == 0:
#             side_temp = 'Up'
#         else:
#             side_temp = 'Down'
#
#         data_prepros_y = data_prepros_arg[:, 3 - it]
#         common_y_pos_arg = test_partition_arg[2 + it]
#
#         diffs = data_prepros_y - common_y_pos_arg
#         diffs_condition = (abs(diffs) <= error_sep_value_sup) & (abs(diffs) >= error_sep_value_inf)
#         location = np.where(diffs_condition)
#         partition_pos = location[0]
#
#         for p in partition_pos:
#             # print(p)
#             temp_partition = data_prepros_arg[p]
#
#             is_partition_checked = temp_partition[4]
#
#             if is_partition_checked == 0:
#                 x1_temp = np.array([test_partition_arg[0], temp_partition[0]]).max()
#                 x2_temp = np.array([test_partition_arg[1], temp_partition[1]]).min()
#
#                 # plt.scatter(x1_temp, common_y_pos_arg, marker='x')
#                 # plt.scatter(x2_temp, common_y_pos_arg, marker='x')
#
#                 # test if range is inside both partitions
#                 condition_1 = (x1_temp >= temp_partition[0]) & (x2_temp <= temp_partition[1])
#                 condition_2 = (x1_temp >= test_partition_arg[0]) & (x2_temp <= test_partition_arg[1])
#
#                 if condition_1 & condition_2:
#                     # index_distance = int(abs(y1_temp - y2_temp)/sep_value_split)
#
#                     # temp
#                     temp_list = []
#                     index_count_aux = int(np.round(abs(temp_partition[1] - temp_partition[0]) / sep_value_split_arg))
#                     for s in range(index_count_aux):
#                         # print(s)
#                         val = temp_partition[0] + sep_value_split_arg / 2 + s * sep_value_split_arg
#                         # plt.scatter(test_partition[1] + sep_value_split, val, marker='x', c='b')
#                         if (val > x1_temp) & (val < x2_temp):
#                             # plt.scatter(val, common_y_pos_arg + sep_value_split_arg / 2, marker='|', c='y')
#                             # print('Partition ' + str(temp_partition[5]) + ' ' + str(temp_partition[6])
#                             #       + ' index: ' + str(s))
#                             temp_list.append([temp_partition[5], temp_partition[6], s])
#
#                     # test
#                     test_list = []
#                     index_count_aux = int(np.round(abs(test_partition_arg[1] - test_partition_arg[0]) / sep_value_split_arg))
#                     for s in range(index_count_aux):
#                         val = test_partition_arg[0] + sep_value_split_arg / 2 + s * sep_value_split_arg
#                         # plt.scatter(test_partition[1] - sep_value_split, val, marker='x', c='r')
#
#                         if (val > x1_temp) & (val < x2_temp):
#                             # plt.scatter(val, common_y_pos_arg - sep_value_split_arg / 2, marker='|', c='g')
#                             # print('Partition ' + str(test_partition_arg[5]) + ' ' + str(test_partition_arg[6])
#                             #       + ' index: ' + str(s))
#                             test_list.append([test_partition_arg[5], test_partition_arg[6], s])
#
#                     for l in range(len(temp_list)):
#                         if it == 1:
#                             # Up
#                             u_part = temp_list[l]
#                             d_part = test_list[l]
#                         else:
#                             # Down
#                             u_part = test_list[l]
#                             d_part = temp_list[l]
#
#                         gi_u = int(u_part[0])
#                         gj_u = int(u_part[1])
#                         gu_index = int(u_part[2])
#
#                         gi_d = int(d_part[0])
#                         gj_d = int(d_part[1])
#                         gd_index = int(d_part[2])
#
#                         y_units_val.append(InterfaceUnit(((gi_u, gj_u), (gi_d, gj_d)), (gu_index, gd_index)))
#
#     return y_units_val
#
#
# def get_xy_units(data_prepros_arg, sep_value_arg, n_split_sep_value_arg, error_val_arg):
#
#     # n_split_sep_value = 10
#     # error_val = 0.05
#     # init algorithm
#     x_units_val = []
#     y_units_val = []
#     for n_partition in range(len(data_prepros_arg)):
#         x_units_temp = search_lr_interfaces(data_prepros_arg, n_partition, sep_value_arg, n_split_sep_value_arg, error_val_arg)
#         y_units_temp = search_ud_interfaces(data_prepros_arg, n_partition, sep_value_arg, n_split_sep_value_arg, error_val_arg)
#
#         x_units_val.extend(x_units_temp)
#         y_units_val.extend(y_units_temp)
#
#         data_prepros_arg[n_partition][4] = 1  # Partition Complete
#
#     return y_units_val, x_units_val
