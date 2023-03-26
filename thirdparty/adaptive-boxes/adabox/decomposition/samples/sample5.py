import ctypes
import random
from timeit import default_timer as timer
import numpy as np

from adabox.plot_tools import plot_rectangles
from adabox.tools import Rectangle


def slice_rectangle(rec_to_slice, slices):
    a = rec_to_slice[1] - rec_to_slice[0]
    b = rec_to_slice[3] - rec_to_slice[2]
    if a > b:
        side = a
        min_coord = 0
        max_coord = 1
    else:
        side = b
        min_coord = 2
        max_coord = 3

    sep = int((side / slices))
    reference = rec_to_slice[min_coord]
    sliced_recs = []
    for i in range(slices - 1):
        nr = rec_to_slice.copy()
        nr[min_coord] = reference
        reference = reference + sep
        nr[max_coord] = reference
        sliced_recs.append(nr)

    nr = rec_to_slice.copy()
    nr[min_coord] = reference
    nr[max_coord] = rec_to_slice[max_coord]
    sliced_recs.append(nr)
    sliced_areas = list(map(lambda r: ((r[1] - r[0]) * (r[3] - r[2])), sliced_recs))
    sliced_ab_ratio = list(map(lambda r: ((r[1] - r[0]) / (r[3] - r[2])), sliced_recs))
    return sliced_recs, sliced_areas, sliced_ab_ratio

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
        rec_out, rec_area_out, ab_ratio_out = find_a_rectangle(rp, data_matrix_arg, lib_arg)
        results.append([rec_out, rec_area_out, ab_ratio_out])

    # conditions
    results_array_area = np.array(results)[:, 1]
    result = results[results_array_area.argmax()]
    return result[0], result[1], result[2]


so_file = "/adabox/decomposition/cpp/getters_completed.so"
getters_so_lib = ctypes.CDLL(so_file)

# Input Path
in_path = './sample_data/boston12.csv'

# Load Demo data with columns [x_position y_position flag]
data_matrix = np.loadtxt(in_path, delimiter=",")
data_matrix = data_matrix.astype(np.intc)

total_area = data_matrix.sum()
n_gpus = 8
max_area = total_area / n_gpus

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
    rec, rec_area, ab_ratio = find_rectangles_and_filter_the_best(random_points, data_matrix, getters_so_lib)
    remove_rectangle_from_matrix(rec, data_matrix)
    end2 = timer()
    print("elapsed time random find/remove " + str((end2 - start2) * 1000) + " milli-seconds")

    start2 = timer()
    coords = np.argwhere(data_matrix == 1)

    if rec_area[0] >= max_area:
        print("NOTE: rec found is gt max area!")
        slices = int(np.ceil(rec_area / max_area)[0])
        s_recs, s_areas, s_ab_ratios = slice_rectangle(rec, slices)

        recs.extend(s_recs)
        areas.extend(s_areas)
        ab_ratios.extend(s_ab_ratios)
    else:
        recs.append(rec)
        areas.append(rec_area)
        ab_ratios.append(ab_ratio)


    end2 = timer()
    print("elapsed time random coords and appends " + str((end2 - start2) * 1000) + " milli-seconds")
    print("")
end = timer()
print("elapsed time " + str(end - start) + "seconds")

# Casting Rectangles
areas_condition = np.array(areas).flatten() >= max_area
not_valid_recs_indexes = np.where(areas_condition)

# largest rectangles
idx = int(not_valid_recs_indexes[0])
r = recs[idx]
slices = int(np.ceil(areas[idx] / max_area)[0])

a = r[1] - r[0]
b = r[3] - r[2]

# for a[x2-x1]
sep = int((a / slices))
reference = r[0]
sliced_recs = []
for i in range(slices - 1):
    nr = r.copy()
    nr[0] = reference
    reference = reference + sep
    nr[1] = reference
    sliced_recs.append(nr)

nr = r.copy()
nr[0] = reference
nr[1] = r[1]
sliced_recs.append(nr)

# for b[y2-xy1]
sep = int((b / slices))
reference = r[2]
sliced_recs = []
for i in range(slices - 1):
    nr = r.copy()
    nr[2] = reference
    reference = reference + sep
    nr[3] = reference
    sliced_recs.append(nr)

nr = r.copy()
nr[2] = reference
nr[3] = r[3]
sliced_recs.append(nr)

# for all ----- ---- ---
idx = 0
r = recs[idx]
slices_out = int(np.ceil(areas[idx] / max_area)[0])


rec_to_slice = r
slices = slices_out

a = rec_to_slice[1] - rec_to_slice[0]
b = rec_to_slice[3] - rec_to_slice[2]
if a > b:
    side = a
    min_coord = 0
    max_coord = 1
else:
    side = b
    min_coord = 2
    max_coord = 3

sep = int((side / slices))
reference = rec_to_slice[min_coord]
sliced_recs = []
for i in range(slices - 1):
    nr = rec_to_slice.copy()
    nr[min_coord] = reference
    reference = reference + sep
    nr[max_coord] = reference
    sliced_recs.append(nr)

nr = rec_to_slice.copy()
nr[min_coord] = reference
nr[max_coord] = rec_to_slice[max_coord]
sliced_recs.append(nr)

sliced_areas = list(map(lambda r: ((r[1] - r[0]) * (r[3] - r[2])), sliced_recs))
sliced_ab_ratio = list(map(lambda r: ((r[1] - r[0]) / (r[3] - r[2])), sliced_recs))


def slice_rectangle(rec_to_slice, slices):
    a = rec_to_slice[1] - rec_to_slice[0]
    b = rec_to_slice[3] - rec_to_slice[2]
    if a > b:
        side = a
        min_coord = 0
        max_coord = 1
    else:
        side = b
        min_coord = 2
        max_coord = 3

    sep = int((side / slices))
    reference = rec_to_slice[min_coord]
    sliced_recs = []
    for i in range(slices - 1):
        nr = rec_to_slice.copy()
        nr[min_coord] = reference
        reference = reference + sep
        nr[max_coord] = reference
        sliced_recs.append(nr)

    nr = rec_to_slice.copy()
    nr[min_coord] = reference
    nr[max_coord] = rec_to_slice[max_coord]
    sliced_recs.append(nr)
    sliced_areas = list(map(lambda r: ((r[1] - r[0]) * (r[3] - r[2])), sliced_recs))
    sliced_ab_ratio = list(map(lambda r: ((r[1] - r[0]) / (r[3] - r[2])), sliced_recs))
    return sliced_recs, sliced_areas, sliced_ab_ratio


# Plotting
rectangles_list = list(map(lambda x: Rectangle(x[0], x[1], x[2], x[3]), [r]))
rectangles_list2 = list(map(lambda x: Rectangle(x[0], x[1], x[2], x[3]), sliced_recs))
plot_rectangles(rectangles_list, 1)
plot_rectangles(rectangles_list2, 1)





# Plotting
rectangles_list = list(map(lambda x: Rectangle(x[0], x[1], x[2], x[3]), recs))
plot_rectangles(rectangles_list, 1)
