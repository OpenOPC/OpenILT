
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
from plyfile import PlyData

from adabox.tools import *

plt.ioff()


def plot_vertex_3d(vertex_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertex_data.x, vertex_data.y, vertex_data.z, c='b', marker='.')
    plt.show()


def plot_vertex_2d(vertex_data):
    plt.figure()
    plt.scatter(vertex_data.x, vertex_data.y, 1, c='b', marker='.')


# Plot the rectangles
# recs_arg: List of Rectangles[Rectangle class]
def plot_rectangles(recs_arg, sep_value_arg):
    plt.figure()
    sep_to_plot = sep_value_arg / 2
    for rec_val in recs_arg:
        plot_rectangle(rec_val, sep_to_plot)


def plot_rectangle(rec_arg, sep_to_plot_arg):
    p1 = np.array([rec_arg.x1 - sep_to_plot_arg, rec_arg.y1 - sep_to_plot_arg])
    p2 = np.array([rec_arg.x1 - sep_to_plot_arg, rec_arg.y2 + sep_to_plot_arg])
    p3 = np.array([rec_arg.x2 + sep_to_plot_arg, rec_arg.y1 - sep_to_plot_arg])
    p4 = np.array([rec_arg.x2 + sep_to_plot_arg, rec_arg.y2 + sep_to_plot_arg])

    ps = np.array([p1, p2, p4, p3, p1])
    plt.plot(ps[:, 0], ps[:, 1])


class Rectangle:

    def __init__(self, x1_arg, x2_arg, y1_arg, y2_arg):
        self.x1 = x1_arg
        self.x2 = x2_arg
        self.y1 = y1_arg
        self.y2 = y2_arg
        self.a = abs(x2_arg - x1_arg)
        self.b = abs(y2_arg - y1_arg)

    def get_area(self):
        return self.a * self.b

    def get_side_ratio(self):
        if self.b == 0:
            return 0
        else:
            return self.a / self.b


class FindRectangleArgs:
    def __init__(self, data_2d_arg, sep_value_arg, init_y_arg, init_x_arg):
        self.data_2d_arg = data_2d_arg
        self.sep_value_arg = sep_value_arg
        self.init_y_arg = init_y_arg
        self.init_x_arg = init_x_arg


def find_rectangle(find_rectangle_args: FindRectangleArgs):
    data_2d_arg = find_rectangle_args.data_2d_arg
    sep_value_arg = find_rectangle_args.sep_value_arg
    init_y_arg = find_rectangle_args.init_y_arg
    init_x_arg = find_rectangle_args.init_x_arg

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


path = '/Users/Juan/django_projects/adaptive-boxes/data/voxel_complex_shape.ply'
ply_data = PlyData.read(path)

# vertex
vertex = ply_data.elements[0].data
vertex_new = pd.DataFrame(vertex)
vertex_pos = vertex_new.loc[:, ['x', 'y', 'z']]

# plot_vertex_3d(vertex_pos)

# Handle z-data
z_min = vertex_pos.z.min()
z_max = vertex_pos.z.max()

# z-scale
z_scale = vertex_pos.z.drop_duplicates()
z_scale.reset_index(drop=True, inplace=True)
z_scale = np.sort(z_scale)

# Get Bottom
z_level = 0
vertex_bottom_set = vertex_pos[vertex_pos.z == z_scale[z_level]]

# plot_vertex_2d(vertex_bottom_set)


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




# Init
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

    n_searches = 100

    # Find Rectangles -- Loop Serial ---
    # start = time.time()

    # start_args = time.time()
    # Create args
    r_args = []
    for i in range(n_searches):
        rand_point = int(np.random.rand() * len(data_2d))
        init_x = data_2d[rand_point][0]
        init_y = data_2d[rand_point][1]
        r_args.append(FindRectangleArgs(data_2d, sep_value, init_y, init_x))
    # end_args = time.time()

    recs_temp = []
    for i in range(n_searches):
        # Find the Rectangle
        rec = find_rectangle(r_args[i])
        recs_temp.append(rec)

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
    # sep_to_plot = sep_value / 2
    # plot_rectangle(recs_temp[max_sqr_index], sep_to_plot)


end = time.time()
print('Work Finished!!!')
print('Elapsed time: ' + str(end - start))

plot_rectangles(recs, sep_value)


# plot_rectangles(recs_temp)






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

    n_searches = 100

    # Find Rectangles -- Loop Serial ---
    # start = time.time()

    # start_args = time.time()
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
    # sep_to_plot = sep_value / 2
    # plot_rectangle(recs_temp[max_sqr_index], sep_to_plot)


end = time.time()
print('Work Finished!!!')
print('Elapsed time: ' + str(end - start))

plot_rectangles(recs, sep_value)
