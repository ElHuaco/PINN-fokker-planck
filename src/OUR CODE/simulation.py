import pickle
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
from pylab import *


filename = r'D:\PINNs-Review\src\OUR CODE\models\2d_diffusion_results.sav'
model = pickle.load(open(filename, 'rb'))
X,Y,T,psol = model[0], model[1], model[2], model[3]
t = T[0,0,:]


fps = 10 # frame per sec
frn = len(t) # frame number of the animation

def update_plot(frame_number, psol, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X[:,:,0], Y[:,:,0], psol[:,:,frame_number], cmap="inferno")
    ax.set_title('predicted model at time: {:.2f}'.format(t[frame_number]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


plot = [ax.plot_surface(X[:,:,0], Y[:,:,0], psol[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(0, 3)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(psol, plot), interval=1000/fps)
file = r'D:\PINNs-Review\src\OUR CODE\simulations\diffusion_2d.mp4'
ani.save(file, writer='ffmpeg',fps=fps)