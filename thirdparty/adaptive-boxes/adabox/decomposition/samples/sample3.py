import ctypes
import random
from timeit import default_timer as timer
import numpy as np

from adabox.plot_tools import plot_rectangles
from adabox.tools import Rectangle


def find_a_rectangle(point, data_binary_matrix, so_lib):
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_double_p = ctypes.POINTER(ctypes.c_double)

    data_matrix_ptr = data_binary_matrix.ctypes.data_as(c_int_p)
    out = np.array([0, 0, 0, 0]).astype(np.intc)
    out_ptr = out.ctypes.data_as(c_int_p)

    area = np.array([0]).astype(np.intc)
    area_ptr = area.ctypes.data_as(c_int_p)

    ab_ratio = np.array([0]).astype(np.float64)
    ab_ratio_ptr = ab_ratio.ctypes.data_as(c_double_p)

    idx_var = int(point[0])
    idj_var = int(point[1])
    m = data_binary_matrix.shape[0]
    n = data_binary_matrix.shape[1]
    so_lib.find_largest_rectangle(idx_var, idj_var, m, n, data_matrix_ptr, out_ptr, area_ptr, ab_ratio_ptr)
    return out, area, ab_ratio


def remove_rectangle_from_matrix(rec_to_remove, data_binary_matrix):
    x1 = rec_to_remove[0]
    y1 = rec_to_remove[1]
    x2 = rec_to_remove[2]
    y2 = rec_to_remove[3]
    data_binary_matrix[x2:y2 + 1, x1:y1 + 1] = 0


def find_rectangles_and_filter_the_best(random_points_arg, data_matrix_arg, lib_arg):
    results = []
    for rp in random_points_arg:
        rec, rec_area, ab_ratio = find_a_rectangle(rp, data_matrix_arg, lib_arg)
        results.append([rec, rec_area, ab_ratio])

    # conditions
    results_array_area = np.array(results)[:, 1]
    results_array_ab_side = np.array(results)[:, 2]
    condition1 = np.logical_and(results_array_ab_side >= 0.6, results_array_ab_side <= 1.2)
    condition1_index = np.where(condition1)
    filtered_area = results_array_area[condition1_index]

    if filtered_area.shape[0] != 0:
        max_item_index = condition1_index[0][filtered_area.argmax()]
        result = results[max_item_index]
    else:
        raise Exception("Rectangle not found")

    return result[0], result[1], result[2]


so_file = "/adabox/decomposition/cpp/getters_completed.so"
getters_so_lib = ctypes.CDLL(so_file)

# Input Path
in_path = './sample_data/boston12.csv'

# Load Demo data with columns [x_position y_position flag]
data_matrix = np.loadtxt(in_path, delimiter=",")
data_matrix = data_matrix.astype(np.intc)

# Plot demo data
# plt.imshow(np.flip(data_matrix, axis=0), cmap='magma', interpolation='nearest')


# search rectangle
coords = np.argwhere(data_matrix == 1)
recs = []
areas = []
ab_ratios = []

start = timer()
while coords.shape[0] != 0:
    start2 = timer()
    n_searches = 100
    random_points = random.choices(coords, k=n_searches)
    end2 = timer()
    print("elapsed time random point " + str((end2 - start2) * 1000) + " milli-seconds")

    start2 = timer()
    try:
        rec, rec_area, ab_ratio = find_rectangles_and_filter_the_best(random_points, data_matrix, getters_so_lib)
    except Exception as e:
        print("Error: Rectangle not found, passing")
        print("")
        continue

    remove_rectangle_from_matrix(rec, data_matrix)
    end2 = timer()
    print("elapsed time random find/remove " + str((end2 - start2) * 1000) + " milli-seconds")

    start2 = timer()
    coords = np.argwhere(data_matrix == 1)
    recs.append(rec)
    areas.append(rec_area)
    ab_ratios.append(ab_ratio)
    end2 = timer()
    print("elapsed time random coords and appends " + str((end2 - start2) * 1000) + " milli-seconds")
    print("")
end = timer()
print("elapsed time " + str(end - start) + "seconds")

# Plotting
rectangles_list = list(map(lambda x: Rectangle(x[0], x[1], x[2], x[3]), recs))
plot_rectangles(rectangles_list, 1)
