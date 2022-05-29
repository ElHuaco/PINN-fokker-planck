import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, uniform_pdf, harmonic_potential, gaussian_pdf

nm = 1e-9
viscosity = 1e-3
radius = 20e-10
drag = 6*np.pi*viscosity*radius

U = harmonic_potential(0, 3e-7)
sim = fokker_planck(temperature=300, drag=drag, extent=200*nm,
            resolution=1*nm, boundary=boundary.reflecting, potential=U)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
#pdf = uniform_pdf(lambda x: (x > 100*nm) & (x < 150*nm))
pdf = gaussian_pdf(15*nm, 5*nm)
p0 = pdf(sim.grid[0])
Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 3e-3, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots()

ax.plot(sim.grid[0]/nm, steady, color='k', ls='--', alpha=.5)
ax.plot(sim.grid[0]/nm, Pt[3], color='blue', alpha=.5)
ax.plot(sim.grid[0]/nm, p0, color='red', ls='--', alpha=.3)
line, = ax.plot(sim.grid[0]/nm, p0, lw=2, color='C3')

def update(i):
    line.set_ydata(Pt[i])
    return [line]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel='x (nm)', ylabel='normalized PDF')
ax.margins(x=0)

plt.show()
