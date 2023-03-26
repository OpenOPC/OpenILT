# CHANGELOG

## 19 Feb 2023

Current decomposition process (Check [complete](adabox/complete)):
1. [binary_matrix_to_rec_list.py](complete%2Fbinary_matrix_to_rec_list.py)
2. [rec_list_to_group_intf_details.py](complete%2Frec_list_to_group_intf_details.py)
3. [group_intf_details_to_gexf_partitioned.py](complete%2Fgroup_intf_details_to_gexf_partitioned.py)
4. [gexf_partitioned_to_gpu_device_global_data.py](complete%2Fgexf_partitioned_to_gpu_device_global_data.py)

Summary of the decomposition process (moved to legacy):
- Decompose binary matrix into list of rectangles: [sample9.py](adabox%2Fdecomposition%2Fsamples%2Fsample9.py)
- List of Rectangles to group/interface details(summary_groups, x_units, y_units, group_details): [post_process_csv_gpu.py](legacy%2Fpostproc_gpu%2Fpost_process_csv_gpu.py)
- Group/interface details to GEFX file with partitions using Metis: [create_partitions_with_metis.py](graph%2Fproto%2Fmetis%2Fcreate_partitions_with_metis.py)
- GEFX partitions to GPU Data(global, device 1, device 2, device 3, etc) [post_proc_kl_partitioning.py](graph%2Fkl_bisection%2Fpost_proc%2Fpost_proc_kl_partitioning.py)


## 7 Feb 2023
- Check sample 9 to generate adabox partitions, this implementation is not set in a new file yet.
- [getters_completed.so](adabox%2Fdecomposition%2Fcpp%2Fgetters_completed.so) is required to run adabox, more info check Readme inside.
- [partitions_data](graph%2Fpartitions_data) check scripts for partition analysis using metis.
- Check this file to post proc partitions with adabox: [post_process_csv_gpu.py](legacy%2Fpostproc_gpu%2Fpost_process_csv_gpu.py)
- Last proto file for Metis: [proto_2.py](graph%2Fproto%2Fmetis%2Fproto_2.py)


