
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# plt.ioff()

path = '/Users/Juan/django_projects/py-ard/server/simulation_result'
file_name = 'best_just_sqrs_50_8000_92_sim_10_5200_2'

out_matrix = np.load(path + '/' + file_name + '.npy')

n_time_steps = out_matrix.shape[1] - 2

# # Plot Results
# plot_step = 20
# plt.figure()
# for i in range(0, n_time_steps, plot_step):
#     plt.clf()
#     plt.scatter(out_matrix[:, 1], out_matrix[:, 0], 1, c=out_matrix[:, 2 + i], marker='.', cmap='brg',
#                         vmin=-0.000004, vmax=0.000004)
#     # plt.scatter(out_matrix[:, 1], out_matrix[:, 0], 1, c=out_matrix[:, 2 + i], marker='.', cmap='brg')
#     plt.pause(0.0001)

max_lim = abs(out_matrix[:, 2:-1].max()/8)

fig = plt.figure()
scat = plt.scatter(out_matrix[:, 1], out_matrix[:, 0], 1, c=out_matrix[:, 2], marker='.', cmap='brg',
                        vmin=-max_lim, vmax=max_lim)


def animate(i):
    scat.set_array(out_matrix[:, 2 + i])
    # print(i)


anim = animation.FuncAnimation(fig, animate, frames=60, interval=12, repeat=False)
writer = animation.writers['ffmpeg'](fps=12)

# anim.save('demo.mp4', writer=writer, dpi=100)

# plt.show()



#
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation
#
# def main():
#     numframes = 100
#     numpoints = 10
#     color_data = np.random.random((numframes, numpoints))
#     x, y, c = np.random.random((3, numpoints))
#
#     fig = plt.figure()
#     scat = plt.scatter(x, y, c=c, s=100)
#
#     ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
#                                   fargs=(color_data, scat))
#     plt.show()
#
# def update_plot(i, data, scat):
#     scat.set_array(data[i])
#     return scat,
#
# main()

