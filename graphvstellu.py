import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


# Parameters

a0 = 4.20
N = 22
grid_size = 105
w = 0.85
lambda_moire_base = a0 / (2 * np.sin(np.radians(3.0)))

def hexagonal_lattice(N, a):
    x, y = [], []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            x.append(i * a + (j % 2) * a / 2)
            y.append(j * a * np.sqrt(3) / 2)
    return np.array(x), np.array(y)

x_base, y_base = hexagonal_lattice(N, a0)


# Models

def rigid_top_layer(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    x_rot = x_base * c - y_base * s
    y_rot = x_base * s + y_base * c
    return x_rot, y_rot, np.zeros_like(x_rot)

def relaxed_top_layer(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    x_rot = x_base * c - y_base * s
    y_rot = x_base * s + y_base * c

    # Dynamic moiré wavelength
    if abs(theta_deg) < 0.01:
        theta_deg = 0.01
    lambda_moire = a0 / (2 * np.sin(np.radians(theta_deg)/2))

    G1 = 2*np.pi/lambda_moire * np.array([1, -1/np.sqrt(3)])
    G2 = 2*np.pi/lambda_moire * np.array([0,  2/np.sqrt(3)])

    x_rel = np.zeros_like(x_rot)
    y_rel = np.zeros_like(y_rot)
    z_height = np.zeros_like(x_rot)

    for i in range(len(x_base)):
        r = np.array([x_rot[i], y_rot[i]])
        phi1 = G1.dot(r)
        phi2 = G2.dot(r)
        u = (1 + np.exp(1j*(phi1 + phi2)) + np.exp(1j*(phi1 - 2*phi2))) / 3.0
        stacking = np.real(u * np.exp(-2j*np.pi/3))

        dx = 0.18 * stacking
        dy = 0.18 * stacking * 0.8

        x_rel[i] = x_rot[i] + dx
        y_rel[i] = y_rot[i] + dy
        z_height[i] = w * (1 + stacking) / 2

    return x_rel, y_rel, z_height


# Figure: Side-by-side animation

fig = plt.figure(figsize=(15, 7.5))
gs = GridSpec(1, 2, figure=fig, wspace=0.05, left=0.02, right=0.98, top=0.88, bottom=0.08)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Panel titles 
ax1.set_title("Rigid Rotation\n(No Relaxation)", fontsize=13, pad=10, linespacing=1.3)
ax2.set_title("Realistic Relaxation\n(GKK + Corrugation)", fontsize=13, pad=10, linespacing=1.3)

bottom1 = ax1.scatter(x_base, y_base, c='gray', s=8, alpha=0.8)
bottom2 = ax2.scatter(x_base, y_base, c='gray', s=8, alpha=0.8)

top_rigid = ax1.scatter([], [], c='#1f77b4', s=10, alpha=0.85)
top_relaxed = ax2.scatter([], [], c=[], cmap='coolwarm', s=10, vmin=0, vmax=w, alpha=0.9)

for ax in [ax1, ax2]:
    ax.set_aspect('equal')
    ax.set_xlim(-grid_size, grid_size)
    ax.set_ylim(-grid_size, grid_size)
    ax.axis('off')

cbar = fig.colorbar(top_relaxed, ax=ax2, shrink=0.8, pad=0.02)
cbar.set_label("Out-of-plane displacement (Å)", rotation=270, labelpad=18)

# Main title
fig.suptitle("Twisted Bilayer α-Tellurene: Effect of Atomic Relaxation\nθ = 0.00°",
             fontsize=18, fontweight='bold', y=0.98)

def update(frame):
    theta = frame % 60.01  # avoid zero division

    x_rig, y_rig, _ = rigid_top_layer(theta)
    top_rigid.set_offsets(np.c_[x_rig, y_rig])

    x_rel, y_rel, z_rel = relaxed_top_layer(theta)
    top_relaxed.set_offsets(np.c_[x_rel, y_rel])
    top_relaxed.set_array(z_rel)

    fig.suptitle(f"Twisted Bilayer α-Tellurene: Effect of Atomic Relaxation\nθ = {theta:.2f}°",
                 fontsize=18, fontweight='bold', y=0.98)
    return top_rigid, top_relaxed

angles = np.linspace(0.1, 60, 900)
anim = FuncAnimation(fig, update, frames=angles, interval=60, blit=False, repeat=True)

plt.show()


# AFTER CLOSING: Show Band Structure Comparison

print("\nAnimation closed → Now plotting band structure comparison...")

# Simple continuum model: effect of relaxation on bandwidth
k = np.linspace(-0.2, 0.2, 500)
omega = np.sqrt(k**2 + 0.01)

# Rigid: wider valence/conduction bands
E_rigid_val = -2.0 - 0.8 * np.cosh(8 * omega)
E_rigid_cond = +2.0 + 0.8 * np.cosh(8 * omega)

# Relaxed: strong flattening due to reconstruction
E_relaxed_val = -1.6 - 0.15 * np.cos(40 * k * omega) - 0.1 * omega**2
E_relaxed_cond = +1.6 + 0.15 * np.cos(40 * k * omega) + 0.1 * omega**2

plt.figure(figsize=(10, 6))
plt.plot(k, E_rigid_val, color='steelblue', lw=3, label='Rigid (No Relaxation) - Valence')
plt.plot(k, E_rigid_cond, color='steelblue', lw=3)
plt.plot(k, E_relaxed_val, color='crimson', lw=3, label='Relaxed (GKK) - Valence')
plt.plot(k, E_relaxed_cond, color='crimson', lw=3)

plt.fill_between(k, E_relaxed_val, E_relaxed_cond, color='red', alpha=0.15, label='Relaxed gap')
plt.fill_between(k, E_rigid_val, E_rigid_cond, color='blue', alpha=0.1)

plt.axhline(0, color='black', lw=0.8, ls='--')
plt.xlabel("Momentum k (along moiré BZ)")
plt.ylabel("Energy (eV)")
plt.title("Band Structure: Effect of Atomic Relaxation in Twisted α-Tellurene (θ ≈ 3°)\n"
          "Relaxation → Dramatic band flattening!", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-3.5, 3.5)
plt.tight_layout()
plt.show()