
import matplotlib.pyplot as plt
import pandas as pd
from plyfile import PlyData
from lib.tools import *


def plot_vertex_3d(vertex_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertex_data.x, vertex_data.y, vertex_data.z, c='r', marker='o')
    plt.show()


def plot_vertex_2d(vertex_data):
    plt.figure()
    plt.scatter(vertex_data.x, vertex_data.y, vertex_data.z, c='r', marker='o')


path = '/Users/Juan/django_projects/adaptive-boxes/data_raw/mera.ply'
ply_data = PlyData.read(path)

# vertex
vertex = ply_data.elements[0].data
vertex_new = pd.DataFrame(vertex)
vertex_pos = vertex_new.loc[:, ['x', 'y', 'z']]

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

plot_vertex_2d(vertex_bottom_set)
plot_vertex_3d(vertex_bottom_set)

rectangles_list = []
n_tests = 10

# Heuristic parameters
max_steps = 5000
percent_max_steps = 0.9

init_ab_ratio = (0.25, 6)
false_counter_max_steps = int(0.2 * max_steps)

# Get Sep Value
data_2d_global = create_2d_data_from_vertex(vertex_bottom_set)
# Get separation Value
sep_value = get_separation_value(data_2d_global)
print(sep_value)


for n_test in range(n_tests):
    print('test#' + str(n_test))
    # Init --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Create Data
    data_2d_global = create_2d_data_from_vertex(vertex_bottom_set)

    # Heuristic
    # Init
    n_sqr = 0
    n_sqr_empty = 0
    # max_steps = 10000
    # percent_max_steps = 0.9
    rectangles = []

    # Adaptive process parameters
    init_ab_ratio = (0.2, 6)
    init_area_limit = (data_2d_global[:, 0].max() - data_2d_global[:, 0].min())/2
    sc_false_counter = 0
    false_counter_max_steps = int(0.1 * max_steps)

    # plt.figure()
    # plt.scatter(data_2d_global[:, 0], data_2d_global[:, 1], 0.1, c=data_2d_global[:, 2], marker='.')

    counter = 0
    while counter < max_steps:
        # print('iteration #: ' + str(counter))
        # Loop
        condition_sqr = data_2d_global[:, 2] == n_sqr_empty
        data_2d = data_2d_global[condition_sqr, :]

        # Break condition
        if len(data_2d) == 0:
            break

        # Random
        rand_point = int(np.random.rand() * len(data_2d))
        init_x = data_2d[rand_point][0]
        init_y = data_2d[rand_point][1]

        # plt.scatter(init_x, init_y, 15, marker='.') #Point to test

        # Order and get only x
        # All values which has init_y
        all_x_points = np.sort(data_2d[data_2d[:, 1] == init_y], axis=0)
        all_y_points = np.sort(data_2d[data_2d[:, 0] == init_x], axis=0)

        # plt.scatter(all_x_points[:, 0], all_x_points[:, 1], 0.5, marker='.')
        # plt.scatter(all_y_points[:, 0], all_y_points[:, 1], 0.5, marker='.')

        init_x_index = np.where(all_x_points[:, 0] == init_x)[0][0]
        init_y_index = np.where(all_y_points[:, 1] == init_y)[0][0]

        # if is_broken(all_x_points[:, 0]):
        #     # print('is broken in x: true')

        dist_l = get_dist_left(all_x_points, init_x_index, sep_value)
        dist_r = get_dist_right(all_x_points, init_x_index, sep_value)

        f_index_l = init_x_index - dist_l
        f_index_r = dist_r + init_x_index

        lr_range = range(f_index_l, f_index_r)
        all_x_points = all_x_points[lr_range, :]

        # plt.scatter(all_x_points[:, 0], all_x_points[:, 1], 0.5, marker='.')

        # if is_broken(all_y_points[:, 1]):
        #     # print('is broken in y: true')

        dist_d = get_dist_down(all_y_points, init_y_index, sep_value)
        dist_u = get_dist_up(all_y_points, init_y_index, sep_value)

        f_index_d = init_y_index - dist_d
        f_index_u = dist_u + init_y_index

        du_range = range(f_index_d, f_index_u)
        all_y_points = all_y_points[du_range, :]

        # plt.scatter(all_y_points[:, 0], all_y_points[:, 1], 0.5, marker='.')

        # Re calc indexes
        init_x_index = np.where(all_x_points[:, 0] == init_x)[0][0]
        init_y_index = np.where(all_y_points[:, 1] == init_y)[0][0]

        # Has a hole? for each x vector > > >
        final_x_min = np.zeros(shape=[2])
        final_x_max = np.zeros(shape=[2])
        # # Down
        final_x_min[0], final_x_max[0], final_y_down = get_final_xy_index_down(data_2d ,all_y_points, init_y_index, dist_l, dist_r, sep_value)
        # # Up
        final_x_min[1], final_x_max[1], final_y_up = get_final_xy_index_up(data_2d, all_y_points, init_y_index, dist_l, dist_r, sep_value)

        # Square/Rectangle Data
        x1 = final_x_min.max()
        x2 = final_x_max.min()
        y1 = final_y_down
        y2 = final_y_up

        a_side = x2 - x1
        b_side = y2 - y1

        # Conditions
        if counter < int(percent_max_steps * max_steps):
            # side_condition
            if (a_side == 0) or (b_side ==0):
                ab_ratio = -1
            else:
                ab_ratio = a_side/b_side

            # print('     ab_ratio: ' + str(ab_ratio))
            side_condition = (ab_ratio >= init_ab_ratio[0]) & (ab_ratio <= init_ab_ratio[1])
            # print('     side condition: ' + str(side_condition))

            # area condition
            ab_area = a_side * b_side
            if ab_area > init_area_limit:
                area_condition = True
            else:
                area_condition = False

        else:
            side_condition = True
            area_condition = True

        if side_condition and area_condition:
            n_sqr = n_sqr + 1
            condition = ((data_2d_global[:, 0] >= x1) & (data_2d_global[:,0] <= x2)) & ((data_2d_global[:,1] >= y1) & (data_2d_global[:,1] <= y2))
            data_2d_global[condition, 2] = n_sqr + 1

            r = Rectangle(x1, x2, y1, y2)
            rectangles.append(r)

            sc_false_counter = 0
        else:
            sc_false_counter = sc_false_counter + 1

        # print('     False counter: ' + str(sc_false_counter))

        if sc_false_counter > false_counter_max_steps:
            # print('---------------------->Dividing area!')
            init_area_limit = init_area_limit/2
            sc_false_counter = 0

        counter = counter + 1

    rectangles_list.append(rectangles)


# Areas and Ab ratios
areas_means = []
ab_ratio_means = []
zero_areas = []

for recs in rectangles_list:
    recs_len = len(recs)
    areas = []
    ab_ratios = []
    zero_area_count = 0
    for i in range(recs_len):

        area_temp = recs[i].get_area()
        if area_temp == 0:
            zero_area_count = zero_area_count + 1
        else:
            areas.append(area_temp)

        a_side = recs[i].x2 - recs[i].x1
        b_side = recs[i].y2 - recs[i].y1

        if(a_side != 0) and (b_side != 0):
            ab_ratios.append(a_side/b_side)

    areas_means.append(np.array(areas).mean())
    ab_ratio_means.append(np.array(ab_ratios).mean())
    zero_areas.append(zero_area_count)


areas_means = np.array(areas_means)
ab_ratio_means = np.array(ab_ratio_means)
zero_areas = np.array(zero_areas)

guides = np.zeros(shape=[len(areas_means), 3])

guides[:, 0] = areas_means
guides[:, 1] = ab_ratio_means
guides[:, 2] = zero_areas

guides_pd = pd.DataFrame(guides).sort_values(by=0)

guides_pd_last = guides_pd.iloc[-7:-1, :]

guides_pd_last_sorted = guides_pd_last.sort_values(by=1)

# n_worst = np.where(areas_means == areas_means.min())[0][0]
n_best = guides_pd_last_sorted.iloc[0, :].name


# n_worst = np.where(ab_ratio_means == ab_ratio_means.max())[0][0]
# n_best = np.where(ab_ratio_means == ab_ratio_means.min())[0][0]
#
# n_worst = np.where(zero_areas == zero_areas.min())[0][0]
# n_best = np.where(zero_areas == zero_areas.max())[0][0]



# Plot Rectangles
index_recs = n_best
rectangles_to_plot = rectangles_list[index_recs]
plt.figure()
sep = sep_value/2
for rec in rectangles_to_plot:
    p1 = np.array([rec.x1 - sep, rec.y1 - sep])
    p2 = np.array([rec.x1 - sep, rec.y2 + sep])
    p3 = np.array([rec.x2 + sep, rec.y1 - sep])
    p4 = np.array([rec.x2 + sep, rec.y2 + sep])

    ps = np.array([p1, p2, p4, p3, p1])
    plt.plot(ps[:, 0], ps[:, 1])



# Plot All
plt.figure()
plt.scatter(data_2d_global[:, 0], data_2d_global[:, 1], 0.1, marker='.')


#
#
# np.median(areas)
# np.std(areas)
# skew(areas)
# kurtosis(areas)
# print(areas.mean())



# ps = np.array([p1, p2, p4, p3, p1])
# plt.plot(ps[:, 0], ps[:, 1], 0.05)




#
# plt.figure()
# plt.scatter(data_2d_global[:, 0], data_2d_global[:, 1], 0.5, c=data_2d_global[:, 2], marker='.')