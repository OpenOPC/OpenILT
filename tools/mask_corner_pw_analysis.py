import numpy as np
import os
import glob
import csv
from collections import defaultdict
import torch
import matplotlib.pyplot as plt


#################
## selected index
################
import torch

root_dir = "/data/yangluo/MTILT_0.1.0_beta/exps/mask"
files = os.listdir(root_dir)
valid_files = list(filter(lambda f: "mask_ratio_" in f, files))

def group_files_with_different_ratio(files):
    ratio_groups = defaultdict(list)
    for filename in files:
        prefix = filename.split("_")[-2]
        runs = filename.split("_")[-1]
        ratio_groups[prefix].append(runs)

    return ratio_groups


file_prefix_string = "_".join(os.path.basename(valid_files[0]).split("_")[:-2])
mask_ratio_group = group_files_with_different_ratio(valid_files)


#
# for file in valid_files:
#     index_dir = os.path.join(root_dir, file, "mask_index")
#     txt_files = glob.glob(os.path.join(index_dir, "*.txt"))
#     for txt in txt_files:
#         sample_num = txt.split(".")[0][-1]
#         if isinstance(eval(sample_num), int):
#
#
#

def read_pws_csv(csv_file):
    with open(csv_file,"r") as f:
        csv_reader = csv.reader(f)
        tmp = []
        pws = []
        for pw in csv_reader:
             if pw:
                for epe in pw:
                    tmp.append(eval(epe))
             else:
                pws.append(np.array(tmp).reshape(5,2))
                tmp = []
        pws.append(np.array(tmp).reshape(5, 2))

    return pws

################
## pw mean & var
################
pws_dict = {}
for prefix in mask_ratio_group.keys():
    files_per_ratio = os.path.join(root_dir, file_prefix_string+"_"+prefix)
    pws = []
    for run in mask_ratio_group[prefix]:
        file_per_ratio_run = files_per_ratio + "_" + run
        pw_file = file_per_ratio_run + "/" + "pw.csv"
        pw = read_pws_csv(pw_file)
        pws.append(pw)
    pws_dict[prefix] = pws

mean_epe_ratios = {}
var_mean_epe_ratios = {}
mean_nom_epe_ratios = {}
var_nom_epe_ratios = {}
worst_nom_epe_ratios = {}
std_epe_ratios = {}

for prefix in pws_dict.keys():
    pws_total_runs_samples = np.array(pws_dict[prefix])
    average_pws = np.mean(pws_total_runs_samples, axis=1)
    mean_pws_per_run = np.mean(average_pws, axis=0)
    var_pws_per_run = np.std(average_pws, axis=0, ddof=1)
    mean_nom_epe = mean_pws_per_run[2,1]
    var_nom_epe = var_pws_per_run[2,1]
    mean_nom_epe_ratios[prefix] = mean_nom_epe
    var_nom_epe_ratios[prefix] = var_nom_epe

    mean_epe = np.mean(mean_pws_per_run)
    var_mean_epe = np.std(mean_pws_per_run,ddof=1)
    mean_epe_ratios[prefix] = mean_epe
    var_mean_epe_ratios[prefix] = var_mean_epe

    torch_pws = torch.tensor(pws_dict[prefix])
    runs, cases, c,l = torch_pws.shape
    worst_nom_epe_ratios_runs, _ = torch.max(torch_pws.reshape(runs, cases,-1),dim=-1)
    worst_nom_epe_ratios[prefix] = torch.mean(worst_nom_epe_ratios_runs.float())

    std_epe_per_runs = torch.std(torch_pws.float().reshape(runs,cases, -1), dim=-1)
    std_epe_ratios[prefix] = torch.mean(std_epe_per_runs)




sorted_mean_nom_epe_ratios = {k: mean_nom_epe_ratios[k] for k in sorted(mean_nom_epe_ratios, key=lambda x: float(x))}
mean_nom_epe_ratios_list = [value.item() for value in sorted_mean_nom_epe_ratios.values()]
sorted_var_nom_epe_ratios = {k: var_nom_epe_ratios[k] for k in sorted(var_nom_epe_ratios, key=lambda x: float(x))}
var_nom_epe_ratios_list = [value.item() for value in sorted_var_nom_epe_ratios.values()]


sorted_worst_nom_epe_ratios = {k: worst_nom_epe_ratios[k] for k in sorted(worst_nom_epe_ratios, key=lambda x: float(x))}
worst_nom_epe_ratios_list = [value.item() for value in sorted_worst_nom_epe_ratios.values()]
sorted_std_epe_ratios = {k: std_epe_ratios[k] for k in sorted(std_epe_ratios, key=lambda x: float(x))}
std_epe_ratios_list = [value.item() for value in sorted_std_epe_ratios.values()]


sorted_mean_epe_ratios = {k: mean_epe_ratios[k] for k in sorted(mean_epe_ratios, key=lambda x: float(x))}
mean_epe_ratios_list = [value for value in sorted_mean_epe_ratios.values()]
sorted_var_mean_epe_ratios = {k: var_mean_epe_ratios[k] for k in sorted(var_mean_epe_ratios, key=lambda x: float(x))}
var_mean_epe_ratios_list = [value for value in sorted_var_mean_epe_ratios.values()]


fig, axes = plt.subplots()
axes.plot(range(len(mean_nom_epe_ratios_list)), mean_nom_epe_ratios_list, c='red', label='nom_mean_epe')
# axes.fill_between(range(len(mean_nom_epe_ratios_list)), np.array(mean_nom_epe_ratios_list) - np.array(var_nom_epe_ratios_list),
#                   np.array(mean_nom_epe_ratios_list) + np.array(var_nom_epe_ratios_list),
#                   color='red',alpha=0.2)

axes.set_title(f'pcb nom epe')
axes.legend(loc='best')
axes.set_xticks(range(len(mean_nom_epe_ratios_list)), list(sorted_mean_nom_epe_ratios.keys()))
axes.set_xlabel('mask_ratio')
axes.set_ylabel('epe')

fig.subplots_adjust(hspace=0.5)
plt.show()
fig.savefig(f'/data/yangluo/MTILT_0.1.0_beta/mask_retuls_show/{file_prefix_string}_nom_epe.png')
plt.close()

fig, axes = plt.subplots()
axes.plot(range(len(mean_epe_ratios_list)), mean_epe_ratios_list, c='blue', label='mean_epe')
# axes.fill_between(range(len(mean_epe_ratios_list)), np.array(mean_epe_ratios_list) - np.array(var_mean_epe_ratios_list),
#                   np.array(mean_epe_ratios_list) + np.array(var_mean_epe_ratios_list),
#                   color='blue',alpha=0.2)

axes.set_title(f'pcb mean epe')
axes.legend(loc='best')
axes.set_xticks(range(len(mean_epe_ratios_list)), list(sorted_mean_nom_epe_ratios.keys()))
axes.set_xlabel('mask_ratio')
axes.set_ylabel('epe')

fig.subplots_adjust(hspace=0.5)
plt.show()
fig.savefig(f'/data/yangluo/MTILT_0.1.0_beta/mask_retuls_show/{file_prefix_string}_mean_epe.png')
plt.close()



fig, axes = plt.subplots()
axes.plot(range(len(worst_nom_epe_ratios_list)), worst_nom_epe_ratios_list, c='blue', label='mean_epe')
axes.set_title(f'pcb worst epe')
axes.legend(loc='best')
axes.set_xticks(range(len(worst_nom_epe_ratios_list)), list(sorted_mean_nom_epe_ratios.keys()))
axes.set_xlabel('mask_ratio')
axes.set_ylabel('epe')

fig.subplots_adjust(hspace=0.5)
plt.show()
fig.savefig(f'/data/yangluo/MTILT_0.1.0_beta/mask_retuls_show/{file_prefix_string}_worst_epe.png')
plt.close()

fig, axes = plt.subplots()
axes.plot(range(len(std_epe_ratios_list)), std_epe_ratios_list, c='blue', label='mean_epe')
axes.set_title(f'pcb std epe')
axes.legend(loc='best')
axes.set_xticks(range(len(std_epe_ratios_list)), list(sorted_mean_nom_epe_ratios.keys()))
axes.set_xlabel('mask_ratio')
axes.set_ylabel('epe')

fig.subplots_adjust(hspace=0.5)
plt.show()
fig.savefig(f'/data/yangluo/MTILT_0.1.0_beta/mask_retuls_show/{file_prefix_string}_std_epe.png')
plt.close()


fig, axes = plt.subplots()
axes.plot(range(len(std_epe_ratios_list)), mean_epe_ratios_list, c='green', label='mean')
axes.plot(range(len(std_epe_ratios_list)), std_epe_ratios_list, c='yellow', label='std')
axes.plot(range(len(std_epe_ratios_list)), mean_nom_epe_ratios_list, c='blue', label='nominal')
axes.plot(range(len(std_epe_ratios_list)), worst_nom_epe_ratios_list, c='red', label='worst')
axes.set_title(f'epe distibution')
axes.legend(loc='best')
axes.set_xticks(range(len(std_epe_ratios_list)), list(sorted_mean_nom_epe_ratios.keys()))
axes.set_xlabel('mask_ratio')
axes.set_ylabel('epe')

fig.subplots_adjust(hspace=0.5)
plt.show()
fig.savefig(f'/data/yangluo/MTILT_0.1.0_beta/mask_retuls_show/{file_prefix_string}_epe_distribution.png')
plt.close()

print("mean_epe:", mean_epe_ratios_list)
print("std:", std_epe_ratios_list)
print("nom_mean_epe:", mean_nom_epe_ratios_list)
print("worst:", worst_nom_epe_ratios_list)

