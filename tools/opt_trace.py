import glob
import os
import random
import torch
import matplotlib.pyplot as plt

import pyilt.evaluation as evaluation
import pylitho.simple as lithosim

# defined litho simulator
litho = lithosim.LithoSim("./config/lithosimple.txt")

def evaluate(mask, target):
    l2, pvb, epe, shot, iou, pvb_ratio = evaluation.evaluate(mask, target, litho, scale=1, shots=False)

    return l2, pvb, iou, pvb_ratio

def generate_random_colors(n):
    colors = []
    for _ in range(n):
        r = random.randint(0, 255)/255.
        g = random.randint(0, 255)/255.
        b = random.randint(0, 255)/255.
        colors.append((r, g, b))
    return colors

# mask dir path
dir = '/data/yangluo/MTILT/exps/simpleilt_db_2048'

# show the results
show_dir = os.path.join('/data/yangluo/MTILT/grad_analysis', os.path.basename(dir)+'_show_opt')
if not os.path.exists(show_dir):
    os.makedirs(show_dir)
samples = list(os.listdir(os.path.join(dir, 'samples')))
num_pkl = len(glob.glob(os.path.join(dir, 'samples', samples[0], '*_mask.pkl')))

l2_metrics = torch.zeros((len(samples)+1, num_pkl))
pvb_metrics = torch.zeros((len(samples)+1, num_pkl))
for index, sample in enumerate(samples):
    sample_path = os.path.join(dir, 'samples', f'{index}_sample')
    target_file_path = os.path.join(sample_path, 'target.pkl')
    target = torch.load(target_file_path, map_location='cuda:0')
    for i in range(num_pkl):
        mask_file_path = os.path.join(sample_path, f'{i}_mask.pkl')
        mask = torch.load(mask_file_path, map_location='cuda:0')
        l2, pvb, iou, pvb_ratio = evaluate(mask, target)
        # l2_metrics[index, i] = l2
        # pvb_metrics[index, i] = pvb
        l2_metrics[index, i] = iou
        pvb_metrics[index, i] = pvb_ratio

l2_metrics[-1,:] = torch.mean(l2_metrics[:-1, :], dim=0)
pvb_metrics[-1,:] = torch.mean(pvb_metrics[:-1,:], dim=0)


fig, axes = plt.subplots(len(samples)//2+1, 2, figsize=(10,25))
for index in range(0, len(samples), 2):


    axes[index // 2][0].scatter(pvb_metrics[index, 0].data.numpy(), l2_metrics[index, 0].data.numpy(), marker='*', facecolors='r', edgecolors='r', s=40,
                                zorder=2.5)
    axes[index // 2][1].scatter(pvb_metrics[index+1, 0].data.numpy(), l2_metrics[index+1, 0].data.numpy(), marker='*',facecolors='r', edgecolors='r',  s=40,
                                zorder=2.5)
    axes[index // 2][0].scatter(pvb_metrics[index, 1:].data.numpy(), l2_metrics[index, 1:].data.numpy(), marker='^',facecolors='k', edgecolors='k',  s=20,
                                zorder=2.5)
    axes[index // 2][1].scatter(pvb_metrics[index+1, 1:].data.numpy(), l2_metrics[index+1, 1:].data.numpy(), marker='^',facecolors='k', edgecolors='k', s=20,
                                        zorder=2.5)

    axes[index // 2][0].plot(pvb_metrics[index, :].data.numpy(), l2_metrics[index, :].data.numpy(), linestyle='--', color='y', lw=3)
    axes[index // 2][1].plot(pvb_metrics[index+1, :].data.numpy(), l2_metrics[index+1, :].data.numpy(), linestyle='--', color='y', lw=3)

    axes[index // 2][0].set_title(f'Optimization trajectories [[{index}_sample]]')
    # axes[index // 2][0].set_xlabel('pvb')
    # axes[index // 2][0].set_ylabel('l2')
    axes[index // 2][0].set_xlabel('1-pvb_ratio')
    axes[index // 2][0].set_ylabel('1-iou')
    axes[index // 2][0].set_xlim(0, 1)
    axes[index // 2][0].set_ylim(0, 1)

    axes[index // 2][1].set_title(f'Optimization trajectories [[{index + 1}_sample]]')
    # axes[index // 2][1].set_xlabel('pvb')
    # axes[index // 2][1].set_ylabel('l2')
    axes[index // 2][1].set_xlabel('1-pvb_ratio')
    axes[index // 2][1].set_ylabel('1-iou')
    axes[index // 2][1].set_xlim(0, 1)
    axes[index // 2][1].set_ylim(0, 1)

axes[index // 2+1][0].scatter(pvb_metrics[-1, 0].data.numpy(), l2_metrics[-1, 0].data.numpy(), marker='*', facecolors='r', edgecolors='r', s=40,
                                zorder=2.5)
axes[index // 2+1][0].scatter(pvb_metrics[-1, 1:].data.numpy(), l2_metrics[-1, 1:].data.numpy(), marker='^',facecolors='k', edgecolors='k',  s=20,
                            zorder=2.5)
axes[index // 2+1][0].plot(pvb_metrics[-1, :].data.numpy(), l2_metrics[-1, :].data.numpy(), linestyle='--', color='y', lw=3)

axes[index // 2+1][0].set_title(f'MEAN Optimization trajectories')
# axes[index // 2+1][0].set_xlabel('pvb')
# axes[index // 2+1][0].set_ylabel('l2')
axes[index // 2+1][0].set_xlabel('1-pvb_new')
axes[index // 2+1][0].set_ylabel('1-iou')
axes[index // 2+1][0].set_xlim(0, 1)
axes[index // 2+1][0].set_ylim(0, 1)

plt.tight_layout()
fig.subplots_adjust(hspace=.5)
fig.savefig(f'{show_dir}/opt_trajectories.png')
plt.show()





