# Prepros
Scripts to deal with large `.ply` files.

## Prepros Ply Data `(prepros_ply_data.py)`
Save `.ply` data file into a numpy file data. It just saves the bottom part of the voxel data reducing the size of the global file.


## Common pipeline

-   Save a `.ply` file, which contains voxelized data(created using FreeCad + Blender).
-   Then, use `prepros_ply_data.py`, to create `.npy` file which contains the bottom part of the voxelized data.
-   (_Optional for adaptive boxes gpu_) Using `csv_matrix_from_prepros.py` transform `.npy` into a binary matrix in `.csv` file.

