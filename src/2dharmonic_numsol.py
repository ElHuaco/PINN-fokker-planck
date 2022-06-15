from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, harmonic_potential
from mpl_toolkits.mplot3d import Axes3D

nm = 1e-9
viscosity = 1e-3
radius = 20e-10
drag = 6*np.pi*viscosity*radius

U = harmonic_potential((50*nm, 50*nm), 0.2e-5)
sim = fokker_planck(temperature=300, drag=drag, extent=[200*nm, 200*nm],
            resolution=2*nm, boundary=boundary.reflecting, potential=U)



def value_to_vector(value, ndim, dtype=float):
    """convert a value to a vector in ndim"""
    value = np.asarray(value, dtype=dtype)
    if value.ndim == 0:
        vec = np.asarray(np.repeat(value, ndim), dtype=dtype)
    else:
        vec = np.asarray(value)
        if vec.size != ndim:
            raise ValueError(f'input vector ({value}) does not have the correct dimensions (ndim = {ndim})')

    return vec

def gaussian_pdf1(center=(-50*nm, -50*nm), width=30*nm):
    """A Gaussian probability distribution function
    Arguments:
        center    center of Gaussian (scalar or vector)
        width     width of Gaussian (scalar or vector)
    """

    center = np.atleast_1d(center)
    ndim = len(center)
    width = value_to_vector(width, ndim)

    def pdf(*args):
        values = np.ones_like(args[0])

        for i, arg in enumerate(args):
            values *= np.exp(-np.square((arg - center[i])/width[i]))

        return values

    return pdf

### time-evolved solution
pdf1 = gaussian_pdf1(center=(-50*nm, -50*nm), width=30*nm)
p0 = pdf1(*sim.grid)

Nsteps = 100
time, Pt = sim.propagate_interval(pdf1, 3e-5, Nsteps=Nsteps, normalize=False)
### animation
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), constrained_layout=True)

#Plot initial
#surf = ax.plot_surface(*sim.grid/nm, p0, cmap='viridis')

# Plot final
surf = ax.plot_surface(*sim.grid/nm, Pt[-1], cmap='viridis')
ax.set_zlim(0, 1)
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


print(Pt.shape)
import pickle
filename = '/Users/max/Documents/Lab_b/PINNs-Review/src/OUR CODE/simulations/2dnumerical_solution2.sav'
pickle.dump(Pt, open(filename, 'wb'))

plt.show()