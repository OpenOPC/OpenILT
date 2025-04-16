import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

# saved grad data dir
dir = '/data/yangluo/MTILT_0.1.0_beta/grad_analysis/mgda_corners_mgda_2048'
show_dir = dir+'_show'
if not os.path.exists(show_dir):
    os.makedirs(show_dir)
old_samples = list(os.listdir(os.path.join(dir, 'old')))
new_samples = list(os.listdir(os.path.join(dir, 'new')))

# iteration
old_grads_file = glob.glob(os.path.join(dir, 'old', old_samples[0], '*.pkl'))
order = eval(os.path.basename(old_grads_file[0])[0])
max_iteration = 0
data = torch.load(old_grads_file[0],'cpu')
o, tn, num = data.shape
assert tn==2, "the task num should be 2."

for index in range(len(old_grads_file)):
    iteration = eval(old_grads_file[index].split('.pkl')[0].split('_')[-1])
    max_iteration = max(max_iteration, iteration)


# grad tensor
old_grads = torch.zeros((len(old_samples), max_iteration, order, tn, num))
new_grads = torch.zeros((len(new_samples), max_iteration, num))

for index in range(len(old_samples)):
    pkls = glob.glob(os.path.join(dir, 'old', old_samples[index], '*.pkl'))
    for pkl in pkls:
        iteration = eval(pkl.split('.pkl')[0].split('_')[-1])-1
        grad_data = torch.load(pkl, 'cpu')
        old_grads[index, iteration, ::] = grad_data

for index in range(len(new_samples)):
    pkls = glob.glob(os.path.join(dir, 'new', new_samples[index], '*.pkl'))
    for pkl in pkls:
        iteration = eval(pkl.split('.pkl')[0].split('_')[-1])-1
        grad_data = torch.load(pkl, 'cpu')
        new_grads[index, iteration, ::] = grad_data

def norm(tensor):
    return tensor/(torch.norm(tensor))

def get_relative_angle(grad, reference):
    if torch.norm(grad) == 0. or torch.norm(reference) == 0.:
        return 0.
    reference = norm(reference)
    grad = norm(grad)
    return torch.acos(torch.clamp(torch.dot(grad, reference),-1., 1.)) * (180/torch.pi)

def get_abs_angle(grad):

    return torch.atan2(norm(grad)[1], norm(grad)[0]) * (180/torch.pi)  # norm(grad).data.numpy()

def generate_random_colors(n):
    colors = []
    for _ in range(n):
        r = random.randint(0, 255)/255.
        g = random.randint(0, 255)/255.
        b = random.randint(0, 255)/255.
        colors.append((r, g, b))
    return colors

#############
# task relative angles
#############
# n order angles
angles = torch.zeros((len(old_samples), max_iteration, order, tn-1))
for index in range(old_grads.shape[0]):
    for iteration in range(old_grads.shape[1]):
        for o in range(order):
            n_order_grads = old_grads[index, iteration, o, ::]
            reference_grad = n_order_grads[0]
            for t in range(1, tn): # tn=2
                angles[index, iteration, o, t-1] = get_relative_angle(n_order_grads[t], reference_grad)

# plot angles
samples_num, iteration_num, order, _ = angles.shape
colors = generate_random_colors(samples_num)
fig, axes = plt.subplots(1, order, figsize=(20,10))
for o in range(order):
    for index in range(samples_num):
        angle_per_sample = angles[index, :, o, :]  # (iteration,1)
        axes[o].plot(range(iteration_num), angle_per_sample.flatten().data.numpy(), c=colors[index], label=f'sample_{index}')
    axes[o].set_title(f'{o+1}_order_angles')
    axes[o].legend(loc='best')
    axes[o].set_xticks(range(iteration_num))
    axes[o].set_xlabel('iteration')
    axes[o].set_ylabel('relative angle')

fig.subplots_adjust(hspace=0.5)
plt.show()
fig.savefig(f'{show_dir}/task_relative_angles.png')


#############
# new grad relative angles
#############
newangles = torch.zeros((len(old_samples), max_iteration, tn+1))
for index in range(old_grads.shape[0]):
    for iteration in range(old_grads.shape[1]):
            n_order_grads = old_grads[index, iteration, 0, ::]
            reference_grad = new_grads[index, iteration, :]
            for t in range(tn): # tn=2
                newangles[index, iteration, t] = get_relative_angle(n_order_grads[t], reference_grad)
            avg_grad = sum(n_order_grads)
            newangles[index, iteration, -1] = get_relative_angle(avg_grad, reference_grad)

# plot angles
samples_num, iteration_num, _ = newangles.shape
colors = generate_random_colors(tn+1)
fig, axes = plt.subplots(samples_num//2, 2, figsize=(25,15))
for index in range(0, samples_num, 2):
    for t in range(tn+1):
        angle_per_sample0 = newangles[index, :, t]  # (iteration,1)
        angle_per_sample1 = newangles[index+1, :, t]
        if t == tn:
            label = 'avg'
        else:
            label = f'{t}_task'
        axes[index//2][0].plot(range(iteration_num), angle_per_sample0.flatten().data.numpy(), c=colors[t], label=label)
        axes[index//2][1].plot(range(iteration_num), angle_per_sample1.flatten().data.numpy(), c=colors[t], label=label)
    axes[index//2][0].set_title(f'new grad vs {tn}-task grads angle[[{index}_sample]]')
    axes[index//2][0].legend(loc='best')
    axes[index//2][0].set_xticks(range(iteration_num))
    axes[index//2][0].set_xlabel('iteration')
    axes[index//2][0].set_ylabel('relative angle')

    axes[index//2][1].set_title(f'new grad vs {tn}-task grads angle[[{index+1}_sample]]')
    axes[index//2][1].legend(loc='best')
    axes[index//2][1].set_xticks(range(iteration_num))
    axes[index//2][1].set_xlabel('iteration')
    axes[index//2][1].set_ylabel('relative angle')

fig.subplots_adjust(hspace=1.5)
plt.show()
fig.savefig(f'{show_dir}/new_grad_relative_angles.png')

#############
# new grad amplitude
#############
amplitudes = torch.zeros((len(old_samples), max_iteration, tn+1))
for index in range(old_grads.shape[0]):
    for iteration in range(old_grads.shape[1]):
            n_order_grads = old_grads[index, iteration, 0, ::]
            new_gard = new_grads[index, iteration, :]
            for t in range(tn): # tn=2
                amplitudes[index, iteration, t] = torch.norm(n_order_grads[t])
            amplitudes[index, iteration, -1] = torch.norm(new_gard)

# plot amplitude
samples_num, iteration_num, _ = amplitudes.shape
colors = generate_random_colors(tn+1)
fig, axes = plt.subplots(samples_num//2, 2, figsize=(25,15))
for index in range(0, samples_num, 2):
    for t in range(tn+1):
        amplitudes_per_sample0 = amplitudes[index, :, t]  # (iteration,1)
        amplitudes_per_sample1 = amplitudes[index+1, :, t]
        if t == tn:
            label = 'new'
        else:
            label = f'{t}_task'
        axes[index//2][0].plot(range(iteration_num), amplitudes_per_sample0.flatten().data.numpy(), c=colors[t], label=label)
        axes[index//2][1].plot(range(iteration_num), amplitudes_per_sample1.flatten().data.numpy(), c=colors[t], label=label)
    axes[index//2][0].set_title(f'new grad vs {tn}-task grads amplitudes [[{index}_sample]]')
    axes[index//2][0].legend(loc='best')
    axes[index//2][0].set_xticks(range(iteration_num))
    axes[index//2][0].set_xlabel('iteration')
    axes[index//2][0].set_ylabel('amplitudes')

    axes[index//2][1].set_title(f'new grad vs {tn}-task grads amplitudes [[{index+1}_sample]]')
    axes[index//2][1].legend(loc='best')
    axes[index//2][1].set_xticks(range(iteration_num))
    axes[index//2][1].set_xlabel('iteration')
    axes[index//2][1].set_ylabel('amplitudes')

fig.subplots_adjust(hspace=1.5)
fig.savefig(f'{show_dir}/new_grad_amplitudes.png')
plt.show()
