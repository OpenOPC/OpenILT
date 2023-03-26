
from multiprocessing import Pool
from adabox.tools import *


class FindRectangleArgs:
    def __init__(self, data_2d_arg, sep_value_arg, init_y_arg, init_x_arg):
        self.data_2d_arg = data_2d_arg
        self.sep_value_arg = sep_value_arg
        self.init_y_arg = init_y_arg
        self.init_x_arg = init_x_arg


def find_rectangle(find_rectangle_args: FindRectangleArgs):
    # args:
    data_2d_arg = find_rectangle_args.data_2d_arg
    sep_value_arg = find_rectangle_args.sep_value_arg
    init_y_arg = find_rectangle_args.init_y_arg
    init_x_arg = find_rectangle_args.init_x_arg

    # work:
    all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == init_y_arg], axis=0)
    all_y_points = np.sort(data_2d_arg[data_2d_arg[:, 0] == init_x_arg], axis=0)

    # plt.scatter(all_x_points[:, 0], all_x_points[:, 1], 0.5, marker='.')
    # plt.scatter(all_y_points[:, 0], all_y_points[:, 1], 0.5, marker='.')

    init_x_index = np.where(all_x_points[:, 0] == init_x_arg)[0][0]
    init_y_index = np.where(all_y_points[:, 1] == init_y_arg)[0][0]

    # if is_broken(all_x_points[:, 0]):
    #     # print('is broken in x: true')

    dist_l = get_dist_left(all_x_points, init_x_index, sep_value_arg)
    dist_r = get_dist_right(all_x_points, init_x_index, sep_value_arg)

    f_index_l = init_x_index - dist_l
    f_index_r = dist_r + init_x_index

    lr_range = range(f_index_l, f_index_r)
    all_x_points = all_x_points[lr_range, :]

    # plt.scatter(all_x_points[:, 0], all_x_points[:, 1], 0.5, marker='.')

    # if is_broken(all_y_points[:, 1]):
    #     # print('is broken in y: true')

    dist_d = get_dist_down(all_y_points, init_y_index, sep_value_arg)
    dist_u = get_dist_up(all_y_points, init_y_index, sep_value_arg)

    f_index_d = init_y_index - dist_d
    f_index_u = dist_u + init_y_index

    du_range = range(f_index_d, f_index_u)
    all_y_points = all_y_points[du_range, :]

    # plt.scatter(all_y_points[:, 0], all_y_points[:, 1], 0.5, marker='.')

    # Re calc indexes
    init_x_index = np.where(all_x_points[:, 0] == init_x_arg)[0][0]
    init_y_index = np.where(all_y_points[:, 1] == init_y_arg)[0][0]

    # Has a hole? for each x vector > > >
    final_x_min = np.zeros(shape=[2])
    final_x_max = np.zeros(shape=[2])
    # # Down
    final_x_min[0], final_x_max[0], final_y_down = get_final_xy_index_down(data_2d_arg, all_y_points, init_y_index,
                                                                           dist_l,
                                                                           dist_r, sep_value_arg)
    # # Up
    final_x_min[1], final_x_max[1], final_y_up = get_final_xy_index_up(data_2d_arg, all_y_points, init_y_index, dist_l,
                                                                       dist_r,
                                                                       sep_value_arg)
    # Square/Rectangle Data
    x1_out = final_x_min.max()
    x2_out = final_x_max.min()
    y1_out = final_y_down
    y2_out = final_y_up

    return Rectangle(x1_out, x2_out, y1_out, y2_out)


def save_rectangle(data_2d_global_arg, rectangle: Rectangle, rectangle_id):
    # Write Condition
    condition = ((data_2d_global_arg[:, 0] >= rectangle.x1) & (data_2d_global_arg[:, 0] <= rectangle.x2)) & (
            (data_2d_global_arg[:, 1] >= rectangle.y1) & (data_2d_global_arg[:, 1] <= rectangle.y2))
    data_2d_global_arg[condition, 2] = rectangle_id



def decompose(data_2d_global_arg, n_searches_per_step):

    data_2d_global = data_2d_global_arg.copy()
    pool = Pool()
    # print('Using ' + str(pool._processes) + ' process(threads) and '
    #       + str(n_searches_per_step) + ' searches per step')
    # Heuristic parameters

    sep_value = get_separation_value(data_2d_global)
    # print(sep_value)

    n_sqr = 0
    n_sqr_empty = 0
    recs = []

    #   Loop
    # start = time.time()
    while True:
        # Select Data which is empty
        condition_sqr = data_2d_global[:, 2] == n_sqr_empty
        data_2d = data_2d_global[condition_sqr, :]

        # Break condition
        if len(data_2d) == 0:
            break

        # n_searches = 200

        # Create args (SERIAL)
        r_args = []
        for i in range(n_searches_per_step):
            rand_point = int(np.random.rand() * len(data_2d))
            init_x = data_2d[rand_point][0]
            init_y = data_2d[rand_point][1]
            r_args.append(FindRectangleArgs(data_2d, sep_value, init_y, init_x))
        # end_args = time.time()

        recs_temp = pool.map(find_rectangle, r_args)

        # features array = [index, area, side_ratio]
        features = np.zeros(shape=[n_searches_per_step, 3])

        for i in range(n_searches_per_step):
            features[i, 0] = i
            features[i, 1] = recs_temp[i].get_area()
            features[i, 2] = recs_temp[i].get_side_ratio()

        # Max
        max_sqr_index = np.where(features[:, 1] == features[:, 1].max())[0][0]

        n_sqr += 1
        save_rectangle(data_2d_global, recs_temp[max_sqr_index], n_sqr)

        recs.append(recs_temp[max_sqr_index])

    return recs, sep_value
