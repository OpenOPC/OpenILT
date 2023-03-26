
import time

import matplotlib.pyplot as plt
import numpy as np


def get_right_bottom_rectangle(idx_i_arg, idx_j_arg, m_arg, n_arg, data_matrix_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i_val = idx_i_arg
        j_val = idx_j_arg + step_j

        if j_val == n_arg:
            break

        temp_val = data_matrix_arg[i_val * n_arg + j_val]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i_val = idx_i_arg + step_i

            if i_val == m_arg:
                break

            temp_val = data_matrix_arg[i_val * n_arg + j_val]

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


def get_left_bottom_rectangle(idx_i_arg, idx_j_arg, m_arg, n_arg, data_matrix_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i_val = idx_i_arg
        j_val = idx_j_arg - step_j

        if j_val == -1:
            break

        temp_val = data_matrix_arg[i_val * n_arg + j_val]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i_val = idx_i_arg + step_i

            if i_val == m_arg:
                break

            temp_val = data_matrix_arg[i_val * n_arg + j_val]

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


def get_left_top_rectangle(idx_i_arg, idx_j_arg, n_arg, data_matrix_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i_val = idx_i_arg
        j_val = idx_j_arg - step_j

        if j_val == -1:
            break

        temp_val = data_matrix_arg[i_val * n_arg + j_val]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i_val = idx_i_arg - step_i

            if i_val == -1:
                break

            temp_val = data_matrix_arg[i_val * n_arg + j_val]

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


def get_right_top_rectangle(idx_i_arg, idx_j_arg, n_arg, data_matrix_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i_val = idx_i_arg
        j_val = idx_j_arg + step_j

        if j_val == n_arg:
            break

        temp_val = data_matrix_arg[i_val * n_arg + j_val]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i_val = idx_i_arg - step_i

            if i_val == -1:
                break

            temp_val = data_matrix_arg[i_val * n_arg + j_val]

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


in_path = '/Users/Juan/django_projects/adaptive-boxes/data_binary/boston12.binary'
out_path = ''

start = time.time()

data_matrix = np.loadtxt(in_path, delimiter=",")
# Flatten Matrix
data_matrix_f = data_matrix.flatten()

# Kernel Data
dim3_grid_x = 1
dim3_grid_y = 1

dim3_block_x = 1    # fixed
dim3_block_y = 4    # fixed

block_dim_y = dim3_block_y
block_dim_x = dim3_block_x

grid_dim_y = dim3_grid_y
grid_dim_x = dim3_grid_x


# KERNEL
# Kernel editable
# Params
#       4 threads: [right-bottom right_top , left-bt, left-tp], 4 coords: [x1 x2 y1 y2]
coords_m = 5
coords_n = 4
coords = np.zeros(shape=[dim3_grid_y, dim3_grid_x, (coords_m * coords_n)])    # Could be stored in Shared Memory
# idx_i = 1   # y rand point
# idx_j = 1   # x rand point

m = data_matrix.shape[0]    # for i
n = data_matrix.shape[1]    # for j


# get random Point
whs_one = np.where(data_matrix == 1)
whs_one_len = whs_one[0].shape[0]
rand_num = int(np.random.rand() * whs_one_len)

idx_i = whs_one[0][rand_num]  # y rand point
idx_j = whs_one[1][rand_num]  # x rand point


# Kernel non-editable - they go in for-loop
block_idx_x = 0
block_idx_y = 0

thread_idx_x = 0
thread_idx_y = 0
# Run Kernel

for block_idx_y in range(grid_dim_y):
    for block_idx_x in range(grid_dim_x):
        print(' ---> running blockId.x: ' + str(block_idx_x) + ' threadId.y: ' + str(block_idx_y))
        #
        # idx_i = int(np.random.rand() * m)  # y-i rand point
        # idx_j = int(np.random.rand() * n)  # x-j rand point

        idx_i = 11
        idx_j = 10

        for thread_idx_y in range(block_dim_y):
            for thread_idx_x in range(block_dim_x):
                print('     ---> running threadId.x: ' + str(thread_idx_x) + ' threadId.y: ' + str(thread_idx_y))
                i = thread_idx_y
                j = thread_idx_x

                g_i = block_dim_y * block_idx_y + i
                g_j = block_dim_x * block_idx_x + j

                if data_matrix_f[idx_i*n + idx_j] == 1:
                    x1 = 0
                    x2 = 0
                    y1 = 0
                    y2 = 0
                    if i == 0:
                        x1, x2, y1, y2 = get_right_bottom_rectangle(idx_i, idx_j, m, n, data_matrix_f)
                    if i == 1:
                        x1, x2, y1, y2 = get_right_top_rectangle(idx_i, idx_j, n, data_matrix_f)
                    if i == 2:
                        x1, x2, y1, y2 = get_left_bottom_rectangle(idx_i, idx_j, m, n, data_matrix_f)
                    if i == 3:
                        x1, x2, y1, y2 = get_left_top_rectangle(idx_i, idx_j, n, data_matrix_f)

                    coords[block_idx_y][block_idx_x][i * coords_n + 0] = x1
                    coords[block_idx_y][block_idx_x][i * coords_n + 1] = x2
                    coords[block_idx_y][block_idx_x][i * coords_n + 2] = y1
                    coords[block_idx_y][block_idx_x][i * coords_n + 3] = y2
                else:
                    print('             disabled thread - rand value is zero')


# max and min in coords[], last row is the final x1 x2 y1 y2
for block_idx_y in range(grid_dim_y):
    for block_idx_x in range(grid_dim_x):
        print(' ---> running blockId.x: ' + str(block_idx_x) + ' threadId.y: ' + str(block_idx_y))
        for thread_idx_y in range(block_dim_y):
            for thread_idx_x in range(block_dim_x):
                print('     ---> running threadId.x: ' + str(thread_idx_x) + ' threadId.y: ' + str(thread_idx_y))
                i = thread_idx_y
                j = thread_idx_x

                g_i = block_dim_y * block_idx_y + i
                g_j = block_dim_x * block_idx_x + j

                x1 = 0
                x2 = 0
                y1 = 0
                y2 = 0

                if i == 0:
                    # pl = coords[[2, 3], 1].max()
                    a = coords[block_idx_y][block_idx_x][coords_n * 2 + 1]
                    b = coords[block_idx_y][block_idx_x][coords_n * 3 + 1]
                    pl = a
                    if b > a:
                        pl = b
                    coords[block_idx_y][block_idx_x][coords_n*4 + i] = pl

                if i == 1:
                    # pr = coords[[0, 1], 1].min()
                    a = coords[block_idx_y][block_idx_x][coords_n * 0 + 1]
                    b = coords[block_idx_y][block_idx_x][coords_n * 1 + 1]
                    pr = a
                    if b < a:
                        pr = b
                    coords[block_idx_y][block_idx_x][coords_n * 4 + i] = pr

                if i == 2:
                    # pt = coords[[1, 3], 3].max()
                    a = coords[block_idx_y][block_idx_x][block_dim_y * 1 + 3]
                    b = coords[block_idx_y][block_idx_x][block_dim_y * 3 + 3]
                    pt = a
                    if b > a:
                        pt = b
                    coords[block_idx_y][block_idx_x][coords_n * 4 + i] = pt

                if i == 3:
                    # pb = coords[[0, 2], 3].min()
                    a = coords[block_idx_y][block_idx_x][coords_n * 0 + 3]
                    b = coords[block_idx_y][block_idx_x][coords_n * 2 + 3]
                    pb = a
                    if b < a:
                        pb = b
                    coords[block_idx_y][block_idx_x][coords_n * 4 + i] = pb


# get area, area value of each block in coord[0][0]
for block_idx_y in range(grid_dim_y):
    for block_idx_x in range(grid_dim_x):
        print(' ---> running blockId.x: ' + str(block_idx_x) + ' threadId.y: ' + str(block_idx_y))
        for thread_idx_y in range(block_dim_y):
            for thread_idx_x in range(block_dim_x):
                print('     ---> running threadId.x: ' + str(thread_idx_x) + ' threadId.y: ' + str(thread_idx_y))
                i = thread_idx_y
                j = thread_idx_x

                g_i = block_dim_y * block_idx_y + i
                g_j = block_dim_x * block_idx_x + j

                x1 = 0
                x2 = 0
                y1 = 0
                y2 = 0

                if i == 0:
                    # a*b
                    a = abs(coords[block_idx_y][block_idx_x][coords_n * 4 + 0] - coords[block_idx_y][block_idx_x][coords_n * 4 + 1])
                    b = abs(coords[block_idx_y][block_idx_x][coords_n * 4 + 2] - coords[block_idx_y][block_idx_x][coords_n * 4 + 3])
                    area = int(a*b)
                    coords[block_idx_y][block_idx_x][coords_n * 0 + 0] = area   # write area in coord[0][0]
                    print('area  ' + str(area))


# get the max area - should exist communication between blocks
for block_idx_y in range(grid_dim_y):
    for block_idx_x in range(grid_dim_x):
        print(' ---> running blockId.x: ' + str(block_idx_x) + ' threadId.y: ' + str(block_idx_y))
        for thread_idx_y in range(block_dim_y):
            for thread_idx_x in range(block_dim_x):
                print('     ---> running threadId.x: ' + str(thread_idx_x) + ' threadId.y: ' + str(thread_idx_y))
                i = thread_idx_y
                j = thread_idx_x

                g_i = block_dim_y * block_idx_y + i
                g_j = block_dim_x * block_idx_x + j

                x1 = 0
                x2 = 0
                y1 = 0
                y2 = 0

                if i == 0:
                    # a*b
                    area = coords[block_idx_y][block_idx_x][coords_n * 0 + 0]
                    print('area  ' + str(area))




# recs = []
# # write data
# recs.append(Rectangle(x1, x2, y1, y2))
# data_matrix[y1:y2+1, x1:x2+1] = 0





pr = coords[[0, 1], 1].min()
pl = coords[[2, 3], 1].max()
pb = coords[[0, 2], 3].min()
pt = coords[[1, 3], 3].max()

# final x1x2 and y1y2


# Plot
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
plt.imshow(data_matrix)
ax.set_aspect('equal')

x1 = int(coords[0][0][coords_n * 4 + 0])
x2 = int(coords[0][0][coords_n * 4 + 1])
y1 = int(coords[0][0][coords_n * 4 + 2])
y2 = int(coords[0][0][coords_n * 4 + 3])

p1 = np.array([x1, y1])
p2 = np.array([x1, y2])
p3 = np.array([x2, y1])
p4 = np.array([x2, y2])
ps = np.array([p1, p2, p4, p3, p1])
plt.plot(ps[:, 0], ps[:, 1], c='r')





#

for i in range(dim3_block_y):

    x1 = coords[i * block_dim_y + 0]
    x2 = coords[i * block_dim_y + 1]
    y1 = coords[i * block_dim_y + 2]
    y2 = coords[i * block_dim_y + 3]

    p1 = np.array([x1, y1])
    p2 = np.array([x1, y2])
    p3 = np.array([x2, y1])
    p4 = np.array([x2, y2])
    ps = np.array([p1, p2, p4, p3, p1])
    plt.plot(ps[:, 0], ps[:, 1], c='w')




#
#
# n = data_matrix.shape[1]    # for j
# m = data_matrix.shape[0]    # for i
#
# recs = []
# stop_flag = False
# print('Doing the Decomposition')
# while not stop_flag:
#
#     ones_counter = (data_matrix == 1).sum()
#     print(ones_counter)
#     if ones_counter == 0:
#         print("End!")
#         break
#
#     search_end_flag = False
#     while not search_end_flag:
#         idx_i = int(np.random.rand()*m)   # y rand point
#         idx_j = int(np.random.rand()*n)   # x rand point
#         if data_matrix[idx_i, idx_j] == 1:
#             break
#
#     x1, x2, y1, y2 = get_right_bottom_rectangle(idx_i, idx_j, n, m)
#     coords[0, :] = np.array([x1, x2, y1, y2])
#
#     x1, x2, y1, y2 = get_right_top_rectangle(idx_i, idx_j, n)
#     coords[1, :] = np.array([x1, x2, y1, y2])
#
#     x1, x2, y1, y2 = get_left_bottom_rectangle(idx_i, idx_j, m)
#     coords[2, :] = np.array([x1, x2, y1, y2])
#
#     x1, x2, y1, y2 = get_left_top_rectangle(idx_i, idx_j)
#     coords[3, :] = np.array([x1, x2, y1, y2])
#
#     # coords[]
#     pr = coords[[0, 1], 1].min()
#     pl = coords[[2, 3], 1].max()
#
#     pb = coords[[0, 2], 3].min()
#     pt = coords[[1, 3], 3].max()
#
#     # final x1x2 and y1y2
#     x1 = int(pl)
#     x2 = int(pr)
#     y1 = int(pt)
#     y2 = int(pb)
#
#     # write data
#     recs.append(Rectangle(x1, x2, y1, y2))
#     data_matrix[y1:y2+1, x1:x2+1] = 0
#
# end = time.time()
# print('Work Finished!!!')
# print('Elapsed time: ' + str(end - start))
#
#
# # Plot
# plot_rectangles(recs, 1)
# plt.show()
#
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # plt.imshow(data_matrix)
# # ax.set_aspect('equal')



# Plot
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
plt.imshow(data_matrix)
ax.set_aspect('equal')

x1 = 40
x2 = 41
y1 = 58
y2 = 254

p1 = np.array([x1, y1])
p2 = np.array([x1, y2])
p3 = np.array([x2, y1])
p4 = np.array([x2, y2])
ps = np.array([p1, p2, p4, p3, p1])
plt.plot(ps[:, 0], ps[:, 1], c='r')
