import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# α-Tellurene lattice parameters

a_te = 4.20
b_te = a_te * np.sqrt(3) / 2
grid_size = 200

def tellurene_lattice(a, b, size):
    x, y = [], []
    nx = int(size / a) + 1
    ny = int(size / b) + 1
    for i in range(-nx, nx):
        for j in range(-ny, ny):
            x.append(i * a + (j % 2) * (a / 2))
            y.append(j * b)
    return np.array(x), np.array(y)

def rotate_lattice(x, y, theta_deg):
    theta = np.radians(theta_deg)
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new

x1, y1 = tellurene_lattice(a_te, b_te, grid_size)


# Setup plot 

fig, ax = plt.subplots(figsize=(8, 9))        # a bit taller
sc1 = ax.scatter(x1, y1, color='green', s=5, label='Layer 1')
sc2 = ax.scatter([], [], color='blue', s=5, alpha=0.6, label='Layer 2')
ax.set_aspect('equal')
ax.set_xlim(-grid_size/2, grid_size/2)
ax.set_ylim(-grid_size/2, grid_size/2)
ax.set_xlabel('x (Å)')
ax.set_ylabel('y (Å)')
ax.legend(loc='upper right')

title = ax.set_title('')   # updated live


plt.subplots_adjust(top=0.82)   


# Animation update

def update(frame):
    theta = frame
    x2, y2 = rotate_lattice(x1, y1, theta)
    sc2.set_offsets(np.c_[x2, y2])
    title.set_text(f'α-Tellurene Moiré Pattern\n'
                   f'Rotation θ = {theta:.2f}°\n'
                   f'(Rigid rotation — no strain/relaxation)')
    return sc2, title


# Animation

angles = np.linspace(0, 180, 900)
anim = FuncAnimation(fig, update, frames=angles,
                     interval=50, blit=False, repeat=True)

plt.show()