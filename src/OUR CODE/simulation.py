import pickle
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
from pylab import *


filename = '/Users/max/Documents/Lab_b/PINNs-Review/src/OUR CODE/simulations/2dnumerical_solution2.sav'
model = pickle.load(open(filename, 'rb'))

def simulation(x):
    if x==0:
        X,Y,T,psol = model[0], model[1], model[2], model[3]
        t = T[0,0,:]


        fps = 15 # frame per sec
        frn = len(t) # frame number of the animation

        def update_plot(frame_number, psol, plot):
            plot[0].remove()
            plot[0] = ax.plot_surface(X[:,:,0], Y[:,:,0], psol[:,:,frame_number], cmap="inferno")
            ax.set_title('predicted model at time: {:.2f}'.format(t[frame_number]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        plot = [ax.plot_surface(X[:,:,0], Y[:,:,0], psol[:,:,0], color='0.75', rstride=1, cstride=1)]
        #ax.set_zlim(0, 0.0005)
        #ax.set_xlim(-0.5, 0.5)
        #ax.set_ylim(-0.5, 0.5)

        ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(psol, plot), interval=1000/fps)
        file = '/Users/max/Documents/Lab_b/PINNs-Review/src/OUR CODE/simulations/diffusion2_2d.mp4'
        ani.save(file, writer='ffmpeg',fps=fps)
    else:
        psol = model.T
        t = np.linspace(0, 2e-4, 200)
        x = np.linspace(-0.5, 0.5, 100)
        y = np.linspace(-0.5, 0.5, 100)
        X, Y, T = np.meshgrid(x,y,t)


        fps = 15 # frame per sec
        frn = 175 # frame number of the animation

        def update_plot(frame_number, psol, plot):
            plot[0].remove()
            plot[0] = ax.plot_surface(X[:,:,0], Y[:,:,0], psol[:,:,frame_number], cmap="inferno")
            ax.set_title('predicted model at time: {:.2f}'.format(t[frame_number]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        plot = [ax.plot_surface(X[:,:,0], Y[:,:,0], psol[:,:,0], color='0.75', rstride=1, cstride=1)]
        ax.set_zlim(0, 0.0018)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

        ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(psol, plot), interval=1000/fps)
        file = '/Users/max/Documents/Lab_b/PINNs-Review/src/OUR CODE/simulations/diffusion2_2d.mp4'
        ani.save(file, writer='ffmpeg',fps=fps)

simulation(1)