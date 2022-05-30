import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf, harmonic_potential
from mpl_toolkits.mplot3d import Axes3D

nm = 1e-9
viscosity = 1e-3
radius = 20e-10
drag = 6*np.pi*viscosity*radius

U = harmonic_potential((0,0), 3e-7)
sim = fokker_planck(temperature=300, drag=drag, extent=[200*nm, 200*nm],
            resolution=10*nm, boundary=boundary.reflecting, potential=U)

### time-evolved solution
pdf = gaussian_pdf(center=(-75*nm, -75*nm), width=30*nm)
p0 = pdf(*sim.grid)

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), constrained_layout=True)

#Plot initial
#surf = ax.plot_surface(*sim.grid/nm, p0, cmap='viridis')

# Plot final
surf = ax.plot_surface(*sim.grid/nm, Pt[-1], cmap='viridis')

#ax.set_zlim([0,np.max(Pt)])
#ax.autoscale(False)

'''
def update(i):
    global surf
    surf.remove()
    surf = ax.plot_surface(*sim.grid/nm, Pt[i], cmap='viridis')

    return [surf]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
'''
ax.set(xlabel='x (nm)', ylabel='y (nm)', zlabel='normalized PDF')

plt.show()
