import os
import glob
import random
import torch
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt

def norm(tensor):
    return tensor/(torch.norm(tensor))

def get_relative_angle(grad, reference):
    if torch.norm(grad) == 0. or torch.norm(reference) == 0.:
        return 0.
    reference = norm(reference)
    grad = norm(grad)
    return torch.acos(torch.clamp(torch.dot(grad, reference),-1., 1.)) * (180/math.pi)

def get_abs_angle(grad):

    return torch.atan2(norm(grad)[1], norm(grad)[0]) * (180/math.pi)  # norm(grad).data.numpy()

def generate_random_colors(n):
    colors = []
    for _ in range(n):
        r = random.randint(0, 255)/255.
        g = random.randint(0, 255)/255.
        b = random.randint(0, 255)/255.
        colors.append((r, g, b))
    return colors


colors = ["#FF0000",
          "#008000",
          "#0000FF",
          "#808080",
          "#FFA500",
          "#FFC0CB",
          "#000000",
          "#800080",
          "#A0522D",
          "#FF00FF",
          "#32CD32"]

parser = argparse.ArgumentParser(description="gradient show")
parser.add_argument("--dir", help="dir name",
                    default='/data/yangluo/MTILT_0.1.0_beta/grad_analysis/base_2048_exactmodel_corners_l2weight_0.5_show_grad')
parser.add_argument("--reference_index", help="relative corner index", default=4) # corner 4/6, 7/10 is the nom one

args = parser.parse_args()

# saved grad data dir
dir = args.dir
show_dir = dir +'_show'
if not os.path.exists(show_dir):
    os.makedirs(show_dir)
old_samples = list(os.listdir(os.path.join(dir, 'old')))
new_samples = list(os.listdir(os.path.join(dir, 'new')))

# iteration
old_grads_file = glob.glob(os.path.join(dir, 'old', old_samples[0], '*.pkl'))
order = eval(os.path.basename(old_grads_file[0])[0])
max_iteration = 0
data = torch.load(old_grads_file[0],'cpu')
tn, num = data.shape
# assert tn == 6, "the task num should be 6."

for index in range(len(old_grads_file)):
    iteration = eval(old_grads_file[index].split('.pkl')[0].split('_')[-1])
    max_iteration = max(max_iteration, iteration)
max_iteration += 1

# grad tensor
old_grads = torch.zeros((len(old_samples), max_iteration, order, tn, num))
new_grads = torch.zeros((len(new_samples), max_iteration, num))

for index in range(len(old_samples)):
    pkls = glob.glob(os.path.join(dir, 'old', old_samples[index], '*.pkl'))
    for pkl in pkls:
        iteration = eval(pkl.split('.pkl')[0].split('_')[-1])
        grad_data = torch.load(pkl, 'cpu')
        old_grads[index, iteration, ::] = grad_data

for index in range(len(new_samples)):
    pkls = glob.glob(os.path.join(dir, 'new', new_samples[index], '*.pkl'))
    for pkl in pkls:
        iteration = eval(pkl.split('.pkl')[0].split('_')[-1])
        grad_data = torch.load(pkl, 'cpu')
        new_grads[index, iteration, ::] = torch.sum(grad_data,dim=0)  # torch.sum(grad_data,dim=0) for simple sum-up GD, and grad_data for others

#############
# task relative angles
#############
# n order angles
angles = torch.zeros((len(old_samples), max_iteration, order, tn+1)) # old_grads[0,-1,0,0,:].numpy() new_grads[0,-1,:].numpy()
for index in range(old_grads.shape[0]):
    for iteration in range(old_grads.shape[1]):
        for o in range(order):
            n_order_grads = old_grads[index, iteration, o, ::]
            reference_grad = n_order_grads[args.reference_index] # tn=3: nom case
            for t in range(0, tn+1):
                if t == args.reference_index:
                    continue
                if t == tn:
                    angles[index, iteration, o, t] = get_relative_angle(new_grads[index, iteration,:], reference_grad) # new grad angle
                    continue
                angles[index, iteration, o, t] = get_relative_angle(n_order_grads[t], reference_grad)

# plot angles
valid_index = [x for x in range(tn+1) if x != args.reference_index]
angles = angles[:, :, :, valid_index]
samples_num, iteration_num, order, tn = angles.shape  #
# colors = generate_random_colors(tn)

def resort_index(reference, t):

    return t if t < reference else t+1

corners_index_resort = [resort_index(args.reference_index, t) for t in range(tn)]


np.savetxt(f'{show_dir}/avg_angles.csv', torch.mean(angles, dim=0).squeeze().numpy(),fmt='%f')
fig, axes = plt.subplots()
means = torch.mean(angles, dim=0).squeeze()
stds = torch.std(angles, dim=0).squeeze()
for i in range(tn):
    if i == tn - 1:
        axes.plot(range(20), means[:,i], c=colors[corners_index_resort[i]], label=f'new_grad')
        axes.fill_between(range(20), means[:, i] - stds[:, i], means[:, i] + stds[:, i], color=colors[corners_index_resort[i]],alpha=0.2)
    else:
        axes.plot(range(20), means[:, i], c=colors[corners_index_resort[i]], label=f'corner_{corners_index_resort[i]}')
        axes.fill_between(range(20), means[:, i] - stds[:, i], means[:, i] + stds[:, i], color=colors[corners_index_resort[i]],alpha=0.2)

plt.axhline(y=90, color='red', linestyle='--')
axes.set_title(f'corner_{args.reference_index} vs {tn - 1} corners grad 1_order_angles- AVG')
axes.legend(loc='best')
axes.set_xticks(range(iteration_num))
axes.set_xlabel('iteration')
axes.set_ylabel('relative angle')

fig.subplots_adjust(hspace=0.5)
# plt.show()
fig.savefig(f'{show_dir}/task_relative_angles_avg.png')
plt.close()


for index in range(0, samples_num):
    fig, axes = plt.subplots()
    for t in range(tn):
        # if t == args.reference_index:
        #     continue
        angle_per_sample = angles[index, :, 0, t]  # (iteration,1)
        if t == tn - 1:
            axes.plot(range(iteration_num), angle_per_sample.flatten().data.numpy(), c=colors[corners_index_resort[t]],
                      label=f'new_grad')
        else:
            axes.plot(range(iteration_num), angle_per_sample.flatten().data.numpy(), c=colors[corners_index_resort[t]],
                      label=f'corner_{corners_index_resort[t]}'
                      )
    # axes.set_title(f'corner_{args.reference_index} vs {tn-1} corners grad 1_order_angles-sample{index}')
    axes.axhline(y=90, color='red', linestyle='--')
    axes.legend(loc='best')
    axes.set_xticks(range(iteration_num))
    axes.set_xlabel('iteration')
    axes.set_ylabel('relative angle')

    fig.subplots_adjust(hspace=0.5)
    # plt.show()
    fig.savefig(f'{show_dir}/task_relative_angles_sample{index}.png')
    plt.close()


amplitudes = torch.zeros((len(old_samples), max_iteration, 11)) # 7
for index in range(len(old_samples)):
    for iteration in range(max_iteration):
        for t in range(7): # 7, 11
            if t == 6:  # 6, 10
                amplitude = torch.norm(new_grads[index, iteration, :])
            else:
                amplitude = torch.norm(old_grads[index, iteration, 0, t, :])
            amplitudes[index, iteration, t] = amplitude

# colors = generate_random_colors(7)

for index in range(0, samples_num):
    fig, axes = plt.subplots()

    for t in range(7): # 7 # 11
        amplitude_iteration = amplitudes[index, :, t]  # (iteration,1)
        if t == 6: # 6, 10
            axes.plot(range(iteration_num), amplitude_iteration.flatten().data.numpy(), c=colors[t],
                      label=f'new_grad')
            continue
        else:
            axes.plot(range(iteration_num), amplitude_iteration.flatten().data.numpy(), c=colors[t], label=f'corner_{t}')
    # axes.set_title(f'1_order_amplitude sample{index}')
    axes.legend(loc='best')
    axes.set_xticks(range(iteration_num))
    axes.set_xlabel('iteration')
    axes.set_ylabel('amplitude')

    fig.subplots_adjust(hspace=0.5)
    # plt.show()
    fig.savefig(f'{show_dir}/task_amplitudes_sample{index}.png')
    plt.close()

