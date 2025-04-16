import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import glob
dir = "/data/yangluo/MTILT_0.1.0_beta/figure6/5dose_simpleilt_pcb_2048_exact_corners_l2weight_3.0"
saved_show_pws = os.path.join(dir, 'analysis')
if not os.path.exists(saved_show_pws):
    os.makedirs(saved_show_pws)

files = list(glob.glob(os.path.join(dir,"*.csv")))
pws_dict = {}
for file in files:
    basename = os.path.basename(file)

    with open(file,"r") as f:
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
        pws_dict[basename[:-4]] = copy.deepcopy(pws)

for index, pw in enumerate(pws):
    stan = np.std(pw, ddof=1)
    mean = np.mean(pw)
    max_epe = np.max(pw)
    list_str = '\n'.join(['{:.2f}'.format(num) for num in [mean, stan, max_epe]])
    with open(os.path.join(saved_show_pws,f'{index}_epe_analysis.txt'), 'w') as file:
        file.write(list_str)

# for index, pw in enumerate(pws):
#     pw_re = np.transpose(pw)
#     with open(os.path.join(saved_show_pws,f'{index}_pw_transpose.csv'), 'w',newline='') as file:
#             writer = csv.writer(file)
#             writer.writerows(pw_re)








