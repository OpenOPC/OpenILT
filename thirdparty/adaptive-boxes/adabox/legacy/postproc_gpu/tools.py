
import numpy as np
import pandas as pd


class InterfaceUnit:
    """group1[0][0]   ----   group2[0][0]"""
    """ each group has interfaces"""
    group = ((0, 0), (0, 0))
    position = (0, 0)

    def __init__(self, group, position):
        self.group = group
        self.position = position


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


def search_lr_interfaces(data_prepros_arg, n_partition_arg, sep_value_arg, n_split_sep_value_arg, error_val_arg):

    sep_value_split_arg = sep_value_arg / n_split_sep_value_arg  # greater than 2

    test_partition_arg = data_prepros_arg[n_partition_arg]

    error_sep_value = error_val_arg * sep_value_arg
    error_sep_value_sup = 0 + error_sep_value
    error_sep_value_inf = 0 - error_sep_value

    x_units_val = []

    # Iter 2 times because of x1 and x2 to find en left and right side of the partition
    for it in range(2):

        if it == 0:
            side_temp = 'Left'
        else:
            side_temp = 'Right'

        data_prepros_x = data_prepros_arg[:, 1 - it]
        common_x_pos_arg = test_partition_arg[0 + it]

        diffs = data_prepros_x - common_x_pos_arg
        diffs_condition = (abs(diffs) <= error_sep_value_sup) & (abs(diffs) >= error_sep_value_inf)
        location = np.where(diffs_condition)
        partition_pos = location[0]

        for p in partition_pos:
            # print(p)
            temp_partition = data_prepros_arg[p]

            is_partition_checked = temp_partition[4]

            if is_partition_checked == 0:
                y1_temp = np.array([test_partition_arg[2], temp_partition[2]]).max()
                y2_temp = np.array([test_partition_arg[3], temp_partition[3]]).min()

                # plt.scatter(common_x_pos_arg, y1_temp, marker='x')
                # plt.scatter(common_x_pos_arg, y2_temp, marker='x')

                # test if range is inside both partitions
                condition_1 = (y1_temp >= temp_partition[2]) & (y2_temp <= temp_partition[3])
                condition_2 = (y1_temp >= test_partition_arg[2]) & (y2_temp <= test_partition_arg[3])

                if condition_1 & condition_2:
                    # index_distance = int(abs(y1_temp - y2_temp)/sep_value_split)

                    # temp
                    temp_list = []
                    index_count_aux = int(np.round(abs(temp_partition[3] - temp_partition[2]) / sep_value_split_arg))
                    for s in range(index_count_aux):
                        # print(s)
                        val = temp_partition[3] - sep_value_split_arg / 2 - s * sep_value_split_arg
                        # plt.scatter(test_partition[1] + sep_value_split, val, marker='x', c='b')
                        if (val > y1_temp) & (val < y2_temp):
                            # plt.scatter(common_x_pos_arg + sep_value_split_arg / 2, val, marker='_', c='b')
                            # print('Partition ' + str(temp_partition[5]) + ' ' + str(temp_partition[6])
                            #       + ' index: ' + str(s))
                            temp_list.append([temp_partition[5], temp_partition[6], s])

                    # test
                    test_list = []
                    index_count_aux = int(np.round(abs(test_partition_arg[3] - test_partition_arg[2]) / sep_value_split_arg))
                    for s in range(index_count_aux):
                        val = test_partition_arg[3] - sep_value_split_arg / 2 - s * sep_value_split_arg
                        # plt.scatter(test_partition[1] - sep_value_split, val, marker='x', c='r')

                        if (val > y1_temp) & (val < y2_temp):
                            # plt.scatter(common_x_pos_arg - sep_value_split_arg / 2, val, marker='_', c='r')
                            # print('Partition ' + str(test_partition_arg[5]) + ' ' + str(test_partition_arg[6])
                            #       + ' index: ' + str(s))
                            test_list.append([test_partition_arg[5], test_partition_arg[6], s])

                    for l in range(len(temp_list)):
                        if it == 0:
                            # Left
                            l_part = temp_list[l]
                            r_part = test_list[l]
                        else:
                            # Right
                            l_part = test_list[l]
                            r_part = temp_list[l]

                        gi_l = int(l_part[0])
                        gj_l = int(l_part[1])
                        gl_index = int(l_part[2])

                        gi_r = int(r_part[0])
                        gj_r = int(r_part[1])
                        gr_index = int(r_part[2])

                        x_units_val.append(InterfaceUnit(((gi_l, gj_l), (gi_r, gj_r)), (gl_index, gr_index)))
    return x_units_val


def search_ud_interfaces(data_prepros_arg, n_partition_arg, sep_value_arg, n_split_sep_value_arg, error_val_arg):

    sep_value_split_arg = sep_value_arg / n_split_sep_value_arg  # greater than 2

    test_partition_arg = data_prepros_arg[n_partition_arg]

    error_sep_value = error_val_arg * sep_value_arg
    error_sep_value_sup = 0 + error_sep_value
    error_sep_value_inf = 0 - error_sep_value

    y_units_val = []

    # Iter 2 times because of y1 and y2 to find in up and down side of the partition
    for it in range(2):

        if it == 0:
            side_temp = 'Up'
        else:
            side_temp = 'Down'

        data_prepros_y = data_prepros_arg[:, 3 - it]
        common_y_pos_arg = test_partition_arg[2 + it]

        diffs = data_prepros_y - common_y_pos_arg
        diffs_condition = (abs(diffs) <= error_sep_value_sup) & (abs(diffs) >= error_sep_value_inf)
        location = np.where(diffs_condition)
        partition_pos = location[0]

        for p in partition_pos:
            # print(p)
            temp_partition = data_prepros_arg[p]

            is_partition_checked = temp_partition[4]

            if is_partition_checked == 0:
                x1_temp = np.array([test_partition_arg[0], temp_partition[0]]).max()
                x2_temp = np.array([test_partition_arg[1], temp_partition[1]]).min()

                # plt.scatter(x1_temp, common_y_pos_arg, marker='x')
                # plt.scatter(x2_temp, common_y_pos_arg, marker='x')

                # test if range is inside both partitions
                condition_1 = (x1_temp >= temp_partition[0]) & (x2_temp <= temp_partition[1])
                condition_2 = (x1_temp >= test_partition_arg[0]) & (x2_temp <= test_partition_arg[1])

                if condition_1 & condition_2:
                    # index_distance = int(abs(y1_temp - y2_temp)/sep_value_split)

                    # temp
                    temp_list = []
                    index_count_aux = int(np.round(abs(temp_partition[1] - temp_partition[0]) / sep_value_split_arg))
                    for s in range(index_count_aux):
                        # print(s)
                        val = temp_partition[0] + sep_value_split_arg / 2 + s * sep_value_split_arg
                        # plt.scatter(test_partition[1] + sep_value_split, val, marker='x', c='b')
                        if (val > x1_temp) & (val < x2_temp):
                            # plt.scatter(val, common_y_pos_arg + sep_value_split_arg / 2, marker='|', c='y')
                            # print('Partition ' + str(temp_partition[5]) + ' ' + str(temp_partition[6])
                            #       + ' index: ' + str(s))
                            temp_list.append([temp_partition[5], temp_partition[6], s])

                    # test
                    test_list = []
                    index_count_aux = int(np.round(abs(test_partition_arg[1] - test_partition_arg[0]) / sep_value_split_arg))
                    for s in range(index_count_aux):
                        val = test_partition_arg[0] + sep_value_split_arg / 2 + s * sep_value_split_arg
                        # plt.scatter(test_partition[1] - sep_value_split, val, marker='x', c='r')

                        if (val > x1_temp) & (val < x2_temp):
                            # plt.scatter(val, common_y_pos_arg - sep_value_split_arg / 2, marker='|', c='g')
                            # print('Partition ' + str(test_partition_arg[5]) + ' ' + str(test_partition_arg[6])
                            #       + ' index: ' + str(s))
                            test_list.append([test_partition_arg[5], test_partition_arg[6], s])

                    for l in range(len(temp_list)):
                        if it == 1:
                            # Up
                            u_part = temp_list[l]
                            d_part = test_list[l]
                        else:
                            # Down
                            u_part = test_list[l]
                            d_part = temp_list[l]

                        gi_u = int(u_part[0])
                        gj_u = int(u_part[1])
                        gu_index = int(u_part[2])

                        gi_d = int(d_part[0])
                        gj_d = int(d_part[1])
                        gd_index = int(d_part[2])

                        y_units_val.append(InterfaceUnit(((gi_u, gj_u), (gi_d, gj_d)), (gu_index, gd_index)))

    return y_units_val


def get_xy_units(data_prepros_arg, sep_value_arg, n_split_sep_value_arg, error_val_arg):

    # n_split_sep_value = 10
    # error_val = 0.05
    # init algorithm
    x_units_val = []
    y_units_val = []
    for n_partition in range(len(data_prepros_arg)):
        x_units_temp = search_lr_interfaces(data_prepros_arg, n_partition, sep_value_arg, n_split_sep_value_arg, error_val_arg)
        y_units_temp = search_ud_interfaces(data_prepros_arg, n_partition, sep_value_arg, n_split_sep_value_arg, error_val_arg)

        x_units_val.extend(x_units_temp)
        y_units_val.extend(y_units_temp)

        data_prepros_arg[n_partition][4] = 1  # Partition Complete

    return y_units_val, x_units_val
