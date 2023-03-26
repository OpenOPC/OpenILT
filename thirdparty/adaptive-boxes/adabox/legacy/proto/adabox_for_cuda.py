
import time
import numpy as np
from adabox.tools import Rectangle, save_to_json


def get_right_bottom_rectangle(idx_i_arg, idx_j_arg, n_arg, m_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i = idx_i_arg
        j = idx_j_arg + step_j

        if j == n_arg:
            break

        temp_val = data_matrix[i, j]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i = idx_i_arg + step_i

            if i == m_arg:
                break

            temp_val = data_matrix[i, j]

            if temp_val == 0:
                break

            step_i += 1

        if step_j == 0:
            first_step_i = step_i
        else:
            if step_i < first_step_i:
                break

        step_j += 1

    x1_val = idx_j_arg
    y1_val = idx_i_arg
    x2_val = idx_j_arg + step_j - 1
    y2_val = idx_i_arg + first_step_i - 1

    return x1_val, x2_val, y1_val, y2_val


def get_left_bottom_rectangle(idx_i_arg, idx_j_arg, m_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i = idx_i_arg
        j = idx_j_arg - step_j

        if j == -1:
            break

        temp_val = data_matrix[i, j]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i = idx_i_arg + step_i

            if i == m_arg:
                break

            temp_val = data_matrix[i, j]

            if temp_val == 0:
                break

            step_i += 1

        if step_j == 0:
            first_step_i = step_i
        else:
            if step_i < first_step_i:
                break

        step_j += 1

    x1_val = idx_j_arg
    y1_val = idx_i_arg
    x2_val = idx_j_arg - step_j + 1
    y2_val = idx_i_arg + first_step_i - 1

    return x1_val, x2_val, y1_val, y2_val


def get_left_top_rectangle(idx_i_arg, idx_j_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i = idx_i_arg
        j = idx_j_arg - step_j

        if j == -1:
            break

        temp_val = data_matrix[i, j]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i = idx_i_arg - step_i

            if i == -1:
                break

            temp_val = data_matrix[i, j]

            if temp_val == 0:
                break

            step_i += 1

        if step_j == 0:
            first_step_i = step_i
        else:
            if step_i < first_step_i:
                break

        step_j += 1

    x1_val = idx_j_arg
    y1_val = idx_i_arg
    x2_val = idx_j_arg - step_j + 1
    y2_val = idx_i_arg - first_step_i + 1

    return x1_val, x2_val, y1_val, y2_val


def get_right_top_rectangle(idx_i_arg, idx_j_arg, n_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i = idx_i_arg
        j = idx_j_arg + step_j

        if j == n_arg:
            break

        temp_val = data_matrix[i, j]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i = idx_i_arg - step_i

            if i == -1:
                break

            temp_val = data_matrix[i, j]

            if temp_val == 0:
                break

            step_i += 1

        if step_j == 0:
            first_step_i = step_i
        else:
            if step_i < first_step_i:
                break

        step_j += 1

    x1_val = idx_j_arg
    y1_val = idx_i_arg
    x2_val = idx_j_arg + step_j - 1
    y2_val = idx_i_arg - first_step_i + 1

    return x1_val, x2_val, y1_val, y2_val


in_path = '/Users/Juan/django_projects/adaptive-boxes/data_prepros/boston12.binary'
out_path = '/Users/Juan/django_projects/adaptive-boxes/results/boston12.json'

start = time.time()

data_matrix = np.loadtxt(in_path, delimiter=",")


end = time.time()
print('Loaded Data!')
print('Elapsed time: ' + str(end - start))

# Kernel Data
dim3_block_x = 1
dim3_block_y = 1

block_dim_y = 1
block_dim_x = 1

# KERNEL
# Kernel non-editable - they go in for-loop
block_idx_x = 0
block_idx_y = 0

thread_idx_x = 0
thread_idx_y = 0

# Kernel editable
# Params
#       4 threads: [right-bottom right_top , left-bt, left-tp], 4 coords: [x1 x2 y1 y2]
coords = np.zeros(shape=[4, 4])    # Could be stored in Cache- Shared Memory
idx_i = 1   # y rand point
idx_j = 1   # x rand point


n = data_matrix.shape[1]    # for j
m = data_matrix.shape[0]    # for i

recs = []
stop_flag = False
print('Doing the Decomposition')
start = time.time()
while not stop_flag:

    ones_counter = (data_matrix == 1).sum()
    # print(ones_counter)
    if ones_counter == 0:
        print("End!")
        break

    # get random Point
    whs_one = np.where(data_matrix == 1)
    whs_one_len = whs_one[0].shape[0]
    rand_num = int(np.random.rand()*whs_one_len)

    idx_i = whs_one[0][rand_num]  # y rand point
    idx_j = whs_one[1][rand_num]  # x rand point

    # Decompositions
    x1, x2, y1, y2 = get_right_bottom_rectangle(idx_i, idx_j, n, m)
    coords[0, :] = np.array([x1, x2, y1, y2])

    x1, x2, y1, y2 = get_right_top_rectangle(idx_i, idx_j, n)
    coords[1, :] = np.array([x1, x2, y1, y2])

    x1, x2, y1, y2 = get_left_bottom_rectangle(idx_i, idx_j, m)
    coords[2, :] = np.array([x1, x2, y1, y2])

    x1, x2, y1, y2 = get_left_top_rectangle(idx_i, idx_j)
    coords[3, :] = np.array([x1, x2, y1, y2])

    # coords[]
    pr = coords[[0, 1], 1].min()
    pl = coords[[2, 3], 1].max()

    pb = coords[[0, 2], 3].min()
    pt = coords[[1, 3], 3].max()

    # final x1x2 and y1y2
    x1 = int(pl)
    x2 = int(pr)
    y1 = int(pt)
    y2 = int(pb)

    # write data
    recs.append(Rectangle(x1, x2, y1, y2))
    data_matrix[y1:y2+1, x1:x2+1] = 0

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

save_to_json(out_path, array_to_save, 1)

#
# # Plot
# plot_rectangles(recs, 1)
# plt.show()

#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.imshow(data_matrix)
# ax.set_aspect('equal')
