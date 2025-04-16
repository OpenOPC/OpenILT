import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import glob

saved_show_pws = "/data/yangluo/MTILT_0.1.0_beta/exps/simpleilt_2048_exactmodel/pw_vis/"
dir = "/data/yangluo/MTILT_0.1.0_beta/exps/simpleilt_2048_exactmodel"
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
                pws.append(np.array(tmp).reshape(3,2))
                tmp = []
        pws.append(np.array(tmp).reshape(3, 2))
        pws_dict[basename[:-4]] = copy.deepcopy(pws)


def show(data, name):
    fig, ax = plt.subplots()
    matrix = ax.imshow(data, cmap='viridis')
    for i in range(pw.shape[0]):
        for j in range(pw.shape[1]):
            ax.text(j, i, str(data[i, j]), ha='center', va="center", color="white")
    ax.axis("off")
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    # plt.show()

for file_name, pws in pws_dict.items():
    saved_dir = os.path.join(saved_show_pws, f"{file_name}")
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    for index, pw in enumerate(pws):
        show(pw, os.path.join(saved_dir, f"sample_{index}_pw.jpg"))

    mean_pw = np.mean(pws, axis=0)
    show(mean_pw, os.path.join(saved_dir, f"mean_pw.jpg"))







