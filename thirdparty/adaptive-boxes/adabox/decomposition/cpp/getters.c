
#include "stdio.h"
#include "stdlib.h"

int get_bottom_distance(int idx_i_arg, int idx_j_arg, int n_arg, int lim, int *data_matrix_arg) {
    int di = 0;
    int temp_val = 0;
    for (int i = idx_i_arg; i < lim; i++) {
        temp_val = data_matrix_arg[i * n_arg + idx_j_arg];
        if (temp_val == 0) {
            break;
        }
        di++;
    }
    return di;
}

int get_top_distance(int idx_i_arg, int idx_j_arg, int n_arg, int lim, int *data_matrix_arg) {
    int di = 0;
    int temp_val = 0;
    for (int i = idx_i_arg; i > lim; i--) {
        temp_val = data_matrix_arg[i * n_arg + idx_j_arg];
        if (temp_val == 0) {
            break;
        }
        di++;
    }
    return di;
}

// results matrix: [x1 x2 y1 y2]
void
get_right_bottom_rectangle(int idx_i_arg, int idx_j_arg, long m_arg, long n_arg, int *data_matrix_arg, int *results) {

    int x1_val = 0;
    int x2_val = 0;
    int y1_val = 0;
    int y2_val = 0;

    int d0, dj;

    d0 = get_bottom_distance(idx_i_arg, idx_j_arg, n_arg, m_arg, data_matrix_arg);

    dj = 0;
    for (int j = idx_j_arg + 1; j < n_arg; j++) {
        int di = get_bottom_distance(idx_i_arg, j, n_arg, idx_i_arg + d0, data_matrix_arg);
        if (di < d0) {
            break;
        }
        dj++;
    }

    x1_val = idx_j_arg;
    y1_val = idx_i_arg;
    x2_val = idx_j_arg + dj;
    y2_val = idx_i_arg + d0 - 1;

    results[0] = x1_val;
    results[1] = x2_val;
    results[2] = y1_val;
    results[3] = y2_val;

}


void
get_left_bottom_rectangle(int idx_i_arg, int idx_j_arg, long m_arg, long n_arg, int *data_matrix_arg, int *results) {


    int x1_val = 0;
    int x2_val = 0;
    int y1_val = 0;
    int y2_val = 0;


    int d0, dj;

    d0 = get_bottom_distance(idx_i_arg, idx_j_arg, n_arg, m_arg, data_matrix_arg);
    dj = 0;
    for (int j = idx_j_arg - 1; j >= 0; j--) {

        int di = get_bottom_distance(idx_i_arg, j, n_arg, idx_i_arg + d0, data_matrix_arg);

        if (di < d0) {
            break;
        }
        dj++;
    }

    x1_val = idx_j_arg;
    y1_val = idx_i_arg;
    x2_val = idx_j_arg - dj;
    y2_val = idx_i_arg + d0 - 1;


    results[0] = x1_val;
    results[1] = x2_val;
    results[2] = y1_val;
    results[3] = y2_val;
}


void get_left_top_rectangle(int idx_i_arg, int idx_j_arg, long n_arg, int *data_matrix_arg, int *results) {


    int x1_val = 0;
    int x2_val = 0;
    int y1_val = 0;
    int y2_val = 0;


    int d0, dj;

    d0 = get_top_distance(idx_i_arg, idx_j_arg, n_arg, -1, data_matrix_arg);
    dj = 0;
    for (int j = idx_j_arg - 1; j > -1; j--) {

        int di = get_top_distance(idx_i_arg, j, n_arg, idx_i_arg - d0, data_matrix_arg);
        if (di < d0) {
            break;
        }
        dj++;
    }

    x1_val = idx_j_arg;
    y1_val = idx_i_arg;
    x2_val = idx_j_arg - dj;
    y2_val = idx_i_arg - d0 + 1;


    results[0] = x1_val;
    results[1] = x2_val;
    results[2] = y1_val;
    results[3] = y2_val;
}


void get_right_top_rectangle(int idx_i_arg, int idx_j_arg, long n_arg, int *data_matrix_arg, int *results) {


    int x1_val = 0;
    int x2_val = 0;
    int y1_val = 0;
    int y2_val = 0;


    int d0, dj;

    d0 = get_top_distance(idx_i_arg, idx_j_arg, n_arg, -1, data_matrix_arg);
    dj = 0;
    for (int j = idx_j_arg + 1; j < n_arg; j++) {
        int di = get_top_distance(idx_i_arg, j, n_arg, idx_i_arg - d0, data_matrix_arg);
        if (di < d0) {
            break;
        }
        dj++;
    }

    x1_val = idx_j_arg;
    y1_val = idx_i_arg;
    x2_val = idx_j_arg + dj;
    y2_val = idx_i_arg - d0 + 1;


    results[0] = x1_val;
    results[1] = x2_val;
    results[2] = y1_val;
    results[3] = y2_val;
}

void
find_largest_rectangle(int idx_i, int idx_j, long m, long n, int *data_matrix, int *out, int *area, double *ab_ratio) {

    const int coords_m = 5;
    const int coords_n = 4;
    int j;
    int coords[coords_m * coords_n];

    int results0[4] = {0, 0, 0, 0};
    int results1[4] = {0, 0, 0, 0};
    int results2[4] = {0, 0, 0, 0};
    int results3[4] = {0, 0, 0, 0};

    get_right_bottom_rectangle(idx_i, idx_j, m, n, data_matrix, results0);
    get_right_top_rectangle(idx_i, idx_j, n, data_matrix, results1);
    get_left_bottom_rectangle(idx_i, idx_j, m, n, data_matrix, results2);
    get_left_top_rectangle(idx_i, idx_j, n, data_matrix, results3);

    j = 0;
    coords[j * coords_n + 0] = results0[0];
    coords[j * coords_n + 1] = results0[1];
    coords[j * coords_n + 2] = results0[2];
    coords[j * coords_n + 3] = results0[3];

    j = 1;
    coords[j * coords_n + 0] = results1[0];
    coords[j * coords_n + 1] = results1[1];
    coords[j * coords_n + 2] = results1[2];
    coords[j * coords_n + 3] = results1[3];

    j = 2;
    coords[j * coords_n + 0] = results2[0];
    coords[j * coords_n + 1] = results2[1];
    coords[j * coords_n + 2] = results2[2];
    coords[j * coords_n + 3] = results2[3];

    j = 3;
    coords[j * coords_n + 0] = results3[0];
    coords[j * coords_n + 1] = results3[1];
    coords[j * coords_n + 2] = results3[2];
    coords[j * coords_n + 3] = results3[3];



    // merge last rectangles
    int a;
    int b;

    j = 0;
    a = coords[2 * coords_n + 1];
    b = coords[3 * coords_n + 1];
    int pl = a;
    if (b > a) {
        pl = b;
    }
    coords[4 * coords_n + j] = pl;
    out[0] = pl;

    j = 1;
    a = coords[0 * coords_n + 1];
    b = coords[1 * coords_n + 1];
    int pr = a;
    if (b < a) {
        pr = b;
    }
    coords[4 * coords_n + j] = pr;
    out[1] = pr;


    j = 2;
    a = coords[1 * coords_n + 3];
    b = coords[3 * coords_n + 3];
    int pt = a;
    if (b > a) {
        pt = b;
    }
    coords[4 * coords_n + j] = pt;
    out[2] = pt;


    j = 3;
    a = coords[0 * coords_n + 3];
    b = coords[2 * coords_n + 3];
    int pb = a;
    if (b < a) {
        pb = b;
    }
    coords[4 * coords_n + j] = pb;
    out[3] = pb;

    //  area
    int a_side = abs(coords[coords_n * 4 + 0] - coords[coords_n * 4 + 1]) + 1;
    int b_side = abs(coords[coords_n * 4 + 2] - coords[coords_n * 4 + 3]) + 1;
    area[0] = a_side * b_side;

    // ab_ ratio
    ab_ratio[0] = (double) ((double) a_side) / ((double) b_side);
}