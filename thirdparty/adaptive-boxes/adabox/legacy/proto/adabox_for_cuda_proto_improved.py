import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# if len(sys.argv) < 3:
#     print('ERROR Args number. Needed: \n[1]In Path(with file.npy) -- prepros file \n[2]Out Path(with .json)')
#     sys.exit()
#
#
# in_path = str(sys.argv[1])
# out_path = str(sys.argv[2])


in_path = '/Users/Juan/django_projects/adaptive-boxes/data_binary/squares.binary'
out_path = ''

data_matrix = np.loadtxt(in_path, delimiter=",")
data_matrix[:,0] = 1


# Plot
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
plt.imshow(data_matrix)
ax.set_aspect('equal')

# Flatten Matrix
data_matrix_f = data_matrix.flatten()

# Kernel Data

dim3_block_x = data_matrix.shape[1]
dim3_block_y = data_matrix.shape[0]

block_dim_y = dim3_block_y
block_dim_x = dim3_block_x

# KERNEL
# Kernel non-editable - they go in for-loop
block_idx_x = 0
block_idx_y = 0

thread_idx_x = 0
thread_idx_y = 0

# Kernel editable
# Params
distances = np.zeros(shape=[data_matrix_f.shape[0]])  # Could be stored in Cache- Shared Memory
idx_i = 7  # y rand point
idx_j = 13  # x rand point

plt.scatter(idx_j, idx_i, c='r')

m = data_matrix.shape[0]
n = data_matrix.shape[1]

# br ----
for i in range(idx_i, m):
    temp_value = data_matrix_f[i * n + idx_j]

    if temp_value == 0:
        i = i - 1
        break
    else:
        plt.scatter(idx_j, i, c='g', marker='x')

d0 = i

for j in range(idx_j + 1, n):
    for i in range(idx_i, d0 + 1):
        # print(str(j) + ' ' + str(i))
        temp_value = data_matrix_f[i * n + j]

        if temp_value == 0:
            i = i - 1
            break
        else:
            plt.scatter(j, i, c='b', marker='x')

    if i < d0:
        j = j - 1
        break

# bl ----
for i in range(idx_i, m):
    temp_value = data_matrix_f[i * n + idx_j]

    if temp_value == 0:
        i = i - 1
        break
    else:
        plt.scatter(idx_j, i, c='g', marker='x')

d0 = i

for j in range(idx_j - 1, -1, -1):
    for i in range(idx_i, d0 + 1):
        # print(str(j) + ' ' + str(i))
        temp_value = data_matrix_f[i * n + j]

        if temp_value == 0:
            i = i - 1
            break
        else:
            plt.scatter(j, i, c='b', marker='x')

    if i < d0:
        j = j + 1
        break


# tl ----
for i in range(idx_i, -1, -1):
    temp_value = data_matrix_f[i * n + idx_j]

    if temp_value == 0:
        i = i + 1
        break
    else:
        plt.scatter(idx_j, i, c='g', marker='x')

d0 = i

for j in range(idx_j - 1, -1, -1):
    for i in range(idx_i, d0 - 1, -1):
        # print(str(j) + ' ' + str(i))
        temp_value = data_matrix_f[i * n + j]

        if temp_value == 0:
            i = i + 1
            break
        else:
            plt.scatter(j, i, c='b', marker='x')

    if i > d0:
        j = j + 1
        break


# tr ----
for i in range(idx_i, -1, -1):
    temp_value = data_matrix_f[i * n + idx_j]

    if temp_value == 0:
        i = i + 1
        break
    else:
        plt.scatter(idx_j, i, c='g', marker='x')

d0 = i

for j in range(idx_j + 1, n):
    for i in range(idx_i, d0 -1, - 1):
        # print(str(j) + ' ' + str(i))
        temp_value = data_matrix_f[i * n + j]

        if temp_value == 0:
            i = i + 1
            break
        else:
            plt.scatter(j, i, c='b', marker='x')

    if i > d0:
        j = j - 1
        break



# plt.scatter(j, idx_i_arg, c='g', marker='x')
# plt.scatter(j, idx_i_arg + first_step_i - 1, c='g', marker='x')


# Run Kernel
for thread_idx_y in range(block_dim_y):
    for thread_idx_x in range(block_dim_x):
        # print('running threadId.x: ' + str(thread_idx_x) + ' threadId.y: ' + str(thread_idx_y))
        i = thread_idx_y
        j = thread_idx_x

        g_i = block_dim_y * block_idx_y + i
        g_j = block_dim_x * block_idx_x + j

        m = block_dim_y
        n = block_dim_x

        plt.scatter(j, i, c='b', marker='x')

        val_in_b = data_matrix_f[n * i + j]
        val_in_a = data_matrix_f[n * i + idx_j]

        distance_j = (j - idx_j) * val_in_b * val_in_a
        distance_i = (i - idx_i) * val_in_b * val_in_a
        print('i: ' + str(i) + '  j: ' + str(j) + '   distance  ' + str(distance_j))

        # if distance_j > 0:
        distances[i * n + j] = distance_j
        #     distances[i * n + j] = distance_j

        # if j == idx_j:
        #     distances[i * n + j] = distance_j + distance_i

print(distances.reshape([m, n]))
distances_matrix = distances.reshape([m, n])

# Break
# Get min distance in left - Atomic can be used(In this case: min() function)

distances_matrix = distances.reshape([m, n])

idx_d = 1
distances_matrix[idx_d, :].max()
distances_matrix[idx_d, :].min()

for thread_idx_y in range(block_dim_y):
    for thread_idx_x in range(block_dim_x):
        # print('running threadId.x: ' + str(thread_idx_x) + ' threadId.y: ' + str(thread_idx_y))
        i = thread_idx_y
        j = thread_idx_x

        g_i = block_dim_y * block_idx_y + i
        g_j = block_dim_x * block_idx_x + j

        m = block_dim_y
        n = block_dim_x

        if (j == 0):
            distances[i * n + 0: i * n + m]


def get_right_bottom_rectangle(idx_i_arg, idx_j_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i = idx_i_arg
        j = idx_j_arg + step_j

        if j == n:
            break

        temp_val = data_matrix[i, j]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i = idx_i_arg + step_i

            if i == m:
                break

            # print(i)
            temp_val = data_matrix[i, j]
            # print(temp_val)
            # plt.scatter(j, i, c='g', marker='x')

            if temp_val == 0:
                break

            step_i += 1

        if step_j == 0:
            first_step_i = step_i
        else:
            if step_i < first_step_i:
                break

        plt.scatter(j, idx_i_arg, c='g', marker='x')
        plt.scatter(j, idx_i_arg + first_step_i - 1, c='g', marker='x')

    x1_val = idx_j_arg
    y1_val = idx_i_arg
    x2_val = idx_j_arg + step_j - 1
    y2_val = idx_i_arg + first_step_i - 1

    return x1_val, x2_val, y1_val, y2_val


def get_left_bottom_rectangle(idx_i_arg, idx_j_arg):
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

            if i == m:
                break

            # print(i)
            temp_val = data_matrix[i, j]
            # print(temp_val)
            # plt.scatter(j, i, c='g', marker='x')

            if temp_val == 0:
                break

            step_i += 1

        if step_j == 0:
            first_step_i = step_i
        else:
            if step_i < first_step_i:
                break

        plt.scatter(j, idx_i_arg, c='g', marker='x')
        plt.scatter(j, idx_i_arg + first_step_i - 1, c='b', marker='x')

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

            # print(i)
            temp_val = data_matrix[i, j]
            # print(temp_val)
            # plt.scatter(j, i, c='g', marker='x')

            if temp_val == 0:
                break

            step_i += 1

        if step_j == 0:
            first_step_i = step_i
        else:
            if step_i < first_step_i:
                break

        plt.scatter(j, idx_i_arg, c='g', marker='x')
        plt.scatter(j, idx_i_arg - first_step_i + 1, c='b', marker='x')

        step_j += 1

    x1_val = idx_j_arg
    y1_val = idx_i_arg
    x2_val = idx_j_arg - step_j + 1
    y2_val = idx_i_arg - first_step_i + 1

    return x1_val, x2_val, y1_val, y2_val


def get_right_top_rectangle(idx_i_arg, idx_j_arg):
    step_j = 0
    first_step_i = 0

    while True:
        i = idx_i_arg
        j = idx_j_arg + step_j

        if j == n:
            break

        temp_val = data_matrix[i, j]
        if temp_val == 0:
            break

        step_i = 0
        while True:
            i = idx_i_arg - step_i

            if i == -1:
                break

            # print(i)
            temp_val = data_matrix[i, j]
            # print(temp_val)
            # plt.scatter(j, i, c='g', marker='x')

            if temp_val == 0:
                break

            step_i += 1

        if step_j == 0:
            first_step_i = step_i
        else:
            if step_i < first_step_i:
                break

        plt.scatter(j, idx_i_arg, c='g', marker='x')
        plt.scatter(j, idx_i_arg - first_step_i + 1, c='g', marker='x')

        step_j += 1

    x1_val = idx_j_arg
    y1_val = idx_i_arg
    x2_val = idx_j_arg + step_j - 1
    y2_val = idx_i_arg - first_step_i + 1

    return x1_val, x2_val, y1_val, y2_val


# Plot
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
plt.imshow(data_matrix)
ax.set_aspect('equal')

m = data_matrix.shape[0]  # for i
n = data_matrix.shape[1]  # for j

for i_n in range(m):
    for j_n in range(n):
        if data_matrix[i_n, j_n] == 1:
            plt.scatter(j_n, i_n, c='w', marker='.')

idx_i = 10  # y rand point
idx_j = 1  # x rand point

plt.scatter(idx_j, idx_i, c='r')

coords = np.zeros(shape=[4, 4])  # 4 threads: [right-bottom right_top , left-bt, left-tp], 4 coords: [x1 x2 y1 y2]

x1, x2, y1, y2 = get_right_bottom_rectangle(idx_i, idx_j)
coords[0, :] = np.array([x1, x2, y1, y2])

p1 = np.array([x1, y1])
p2 = np.array([x1, y2])
p3 = np.array([x2, y1])
p4 = np.array([x2, y2])
ps = np.array([p1, p2, p4, p3, p1])
plt.plot(ps[:, 0], ps[:, 1], c='w')

x1, x2, y1, y2 = get_right_top_rectangle(idx_i, idx_j)
coords[1, :] = np.array([x1, x2, y1, y2])

p1 = np.array([x1, y1])
p2 = np.array([x1, y2])
p3 = np.array([x2, y1])
p4 = np.array([x2, y2])
ps = np.array([p1, p2, p4, p3, p1])
plt.plot(ps[:, 0], ps[:, 1], c='w')

x1, x2, y1, y2 = get_left_bottom_rectangle(idx_i, idx_j)
coords[2, :] = np.array([x1, x2, y1, y2])

p1 = np.array([x1, y1])
p2 = np.array([x1, y2])
p3 = np.array([x2, y1])
p4 = np.array([x2, y2])
ps = np.array([p1, p2, p4, p3, p1])
plt.plot(ps[:, 0], ps[:, 1], c='w')

x1, x2, y1, y2 = get_left_top_rectangle(idx_i, idx_j)
coords[3, :] = np.array([x1, x2, y1, y2])

p1 = np.array([x1, y1])
p2 = np.array([x1, y2])
p3 = np.array([x2, y1])
p4 = np.array([x2, y2])
ps = np.array([p1, p2, p4, p3, p1])
plt.plot(ps[:, 0], ps[:, 1], c='w')

# coords[]
pr = coords[[0, 1], 1].min()
pl = coords[[2, 3], 1].max()

pb = coords[[0, 2], 3].min()
pt = coords[[1, 3], 3].max()

# final x1x2 and y1y2
x1 = pl
x2 = pr
y1 = pt
y2 = pb

plt.scatter(x1, y1, c='r')
plt.scatter(x2, y2, c='b')

p1 = np.array([x1, y1])
p2 = np.array([x1, y2])
p3 = np.array([x2, y1])
p4 = np.array([x2, y2])
ps = np.array([p1, p2, p4, p3, p1])
plt.plot(ps[:, 0], ps[:, 1], c='r')

data_matrix[y1:y2 + 1, x1:x2 + 1] = 0
