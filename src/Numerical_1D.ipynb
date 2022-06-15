from locale import normalize
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, uniform_pdf, harmonic_potential, gaussian_pdf
import pickle
import matplotlib.gridspec as gridspec
from scipy.stats import norm


nm = 1e-9
viscosity = 1e-3
radius = 20e-10
drag = 6*np.pi*viscosity*radius

U = harmonic_potential(-75*nm, 3e-7)
sim = fokker_planck(temperature=300, drag=drag, extent=350*nm,
            resolution=1*nm, boundary=boundary.reflecting, potential=U)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
#pdf = uniform_pdf(lambda x: (x > 100*nm) & (x < 150*nm))
x = np.linspace(-0.1, 0.25, 350)
pdf = gaussian_pdf(75*nm, 50*nm)
#pdf=(norm.pdf(x,loc=0.15,scale=0.05)).T
p0=(norm.pdf(sim.grid[0],loc=0.15,scale=0.05)).T
#p0 = pdf(sim.grid[0])
Nsteps = 200
t_lower = 0
A = 3e-7 / drag
t_upper = 0.69 / A  # Typical Ornstein-Uhlenbeck time is ln(2) / A
_, Pt = sim.propagate_interval(pdf, t_upper, Nsteps=Nsteps, normalize=False)

### animation
#fig, ax = plt.subplots()

#ax.plot(x, steady, color='black', ls='--', alpha=.5)
#ax.plot(x, Pt[1], color='blue', alpha=.5)
#ax.plot(x, p0, color='red', ls='--', alpha=.3)
#ax.set(xlabel='x (nm)', ylabel='normalized PDF')

filename = 'simulations/numerical_solution.sav'
pickle.dump(Pt, open(filename, 'wb'))

#x_lower = -0.1
#x_upper = 0.25

#plt.show()

t_large = t_upper * 10
_, Pt = sim.propagate_interval(pdf, t_large, Nsteps=Nsteps*10, normalize=False)
filename = 'simulations/large_t_numerical_solution.sav'
pickle.dump(Pt, open(filename, 'wb'))

