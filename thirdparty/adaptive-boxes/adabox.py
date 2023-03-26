
import sys
import time
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


if len(sys.argv) < 3:
    print('ERROR Args number. Needed: \n[1]In Path(with file.npy) -- prepros file \n[2]Out Path(with .json)')
    sys.exit()


in_path = str(sys.argv[1])
out_path = str(sys.argv[2])

np_data = np.load(in_path)
vertex_bottom_set = pd.DataFrame(np_data)
vertex_bottom_set.columns = ['x', 'y', 'z']

rectangles_list = []
n_tests = 10

# Heuristic parameters
max_steps = 5000
percent_max_steps = 0.9

init_ab_ratio = (0.25, 6)
false_counter_max_steps = int(0.2 * max_steps)

# Create Global Matrix of points: []
data_2d_global = create_2d_data_from_vertex(vertex_bottom_set)
# Get separation Value
sep_value = get_separation_value(data_2d_global)
print(sep_value)


# Init Parallel
pool = []
if __name__ == '__main__':
    pool = Pool()
    print('Processes in Pool: ' + str(pool._processes))

# Create Global Matrix of points: []
data_2d_global = create_2d_data_from_vertex(vertex_bottom_set)
# Get separation Value
sep_value = get_separation_value(data_2d_global)
print(sep_value)

n_sqr = 0
n_sqr_empty = 0
recs = []

#   Loop
start = time.time()
while True:
    # Select Data which is empty
    condition_sqr = data_2d_global[:, 2] == n_sqr_empty
    data_2d = data_2d_global[condition_sqr, :]

    # Break condition
    if len(data_2d) == 0:
        break

    n_searches = 200

    # Create args (SERIAL)
    r_args = []
    for i in range(n_searches):
        rand_point = int(np.random.rand() * len(data_2d))
        init_x = data_2d[rand_point][0]
        init_y = data_2d[rand_point][1]
        r_args.append(FindRectangleArgs(data_2d, sep_value, init_y, init_x))
    # end_args = time.time()

    recs_temp = pool.map(find_rectangle, r_args)

    # features array = [index, area, side_ratio]
    features = np.zeros(shape=[n_searches, 3])

    for i in range(n_searches):
        features[i, 0] = i
        features[i, 1] = recs_temp[i].get_area()
        features[i, 2] = recs_temp[i].get_side_ratio()

    # Max
    max_sqr_index = np.where(features[:, 1] == features[:, 1].max())[0][0]

    n_sqr += 1
    save_rectangle(data_2d_global, recs_temp[max_sqr_index], n_sqr)

    recs.append(recs_temp[max_sqr_index])


end = time.time()
print('Work Finished!!!')
print('Elapsed time: ' + str(end - start))

# Save best data set
best_set = recs
array_to_save = np.zeros(shape=[len(best_set), 4])

for x in range(len(best_set)):
    array_to_save[x, 0] = best_set[x].x1
    array_to_save[x, 1] = best_set[x].x2
    array_to_save[x, 2] = best_set[x].y1
    array_to_save[x, 3] = best_set[x].y2

save_to_json(out_path, array_to_save, sep_value)
