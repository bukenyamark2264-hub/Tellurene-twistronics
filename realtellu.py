import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Realistic twisted bilayer α-tellurene with relaxation
# Improved with accurate lattice parameters and buckling for chirality
# Based on literature: α-tellurene has trigonal structure with helical chains,
# lattice constant a ≈ 4.46 Å (bulk-like), buckling height Δ ≈ 1.42 Å.
# Here, we model as buckled honeycomb lattice for approximation,
# with chirality sign flipping the buckling direction (left/right-handed).
# GKK parameters are illustrative (adapted from graphene/TMD literature);
# real Te values may differ—future work could calibrate from DFT.
# Source: Nat. Comm. (2020) on Te chirality; other refs for params.

a0 = 4.46                     # Lattice constant (Å) from bulk Te, adjusted for monolayer
delta = 1.42                  # Buckling height (Å) for chirality
grid_size = 100
N = 25                        # Number of unit cells per side (increased for detail)

# GKK model parameters (illustrative; calibrate for Te in production code)
A = 0.32                      # Stacking energy difference AA-AB (eV/atom)
w = 0.85                      # Corrugation amplitude (Å)


# Build bottom layer (buckled honeycomb with chirality)

def build_layer(N, a, delta, chiral=1):
    x, y, z = [], [], []
    # Honeycomb vectors
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3)/2])
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            base = i * a1 + j * a2
            # Sublattice A at z=0
            x.append(base[0])
            y.append(base[1])
            z.append(0.0)
            # Sublattice B at z = chiral * delta (sign for left/right-handed)
            x.append(base[0] + a / 3.0)  # Offset for honeycomb
            y.append(base[1] + a * np.sqrt(3) / 3.0)
            z.append(chiral * delta)
    return np.array(x), np.array(y), np.array(z)

x_bot, y_bot, z_bot = build_layer(N, a0, delta, chiral=1)  # Right-handed example


# Top layer: rigid rotation + realistic GKK relaxation
# Chirality can be same or opposite for homo/heterochiral stacking

def relaxed_top_layer(theta_deg, chiral_top=1):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)

    # Rigid rotation of coordinates (including initial z)
    x_rot = x_bot * c - y_bot * s
    y_rot = x_bot * s + y_bot * c
    z_rot = z_bot  # z remains (buckling preserved)

    # Update moiré wavelength for current theta (fix from original code)
    if abs(theta_deg) < 0.01:  # Avoid div by zero
        theta_deg = 0.01
    lambda_moire = a0 / (2 * np.sin(np.radians(theta_deg)/2))

    x_rel = np.zeros_like(x_rot)
    y_rel = np.zeros_like(y_rot)
    z_height = np.zeros_like(z_rot)  # Additional relaxation in z

    # Moiré reciprocal vectors (hexagonal symmetry)
    G1 = 2 * np.pi / lambda_moire * np.array([1, -1/np.sqrt(3)])
    G2 = 2 * np.pi / lambda_moire * np.array([0, 2/np.sqrt(3)])

    for i in range(len(x_bot)):
        rx, ry = x_rot[i], y_rot[i]

        phi1 = np.dot(G1, [rx, ry])
        phi2 = np.dot(G2, [rx, ry])

        # Complex stacking order parameter u (GKK form, adapted for buckling)
        u = (1 + np.exp(1j * (phi1 + phi2)) + np.exp(1j * (phi1 - 2*phi2))) / 3.0
        stacking = np.real(u * np.exp(-2j * np.pi / 3))  # +1 at AA, -0.5 at AB

        # In-plane relaxation toward AB regions (scaled for Te)
        dx = 0.18 * stacking  # Illustrative; Te may have larger due to heavier atoms
        dy = 0.18 * stacking * 0.8

        x_rel[i] = x_rot[i] + dx
        y_rel[i] = y_rot[i] + dy

        # Out-of-plane corrugation (higher at AA, lower at AB; chiral affects sign?)
        z_height[i] = z_rot[i] + chiral_top * w * (1 + stacking) / 2

    return x_rel, y_rel, z_height


# Plot setup 

fig = plt.figure(figsize=(9, 8))                     # larger window
gs = fig.add_gridspec(1, 1, right=0.80)              # leave room for colorbar
ax = fig.add_subplot(gs[0])

# Use constrained_layout + manual adjustments = perfect control
fig.constrained_layout = True

bottom = ax.scatter(x_bot, y_bot, c=z_bot, cmap='gray', s=8, alpha=0.9, label='Bottom layer (buckled)')
top = ax.scatter([], [], c=[], cmap='coolwarm', s=10, vmin=-delta, vmax=delta + w, edgecolors='none')

ax.set_aspect('equal')
ax.set_xlim(-grid_size, grid_size)
ax.set_ylim(-grid_size, grid_size)
ax.set_xlabel('x (Å)')
ax.set_ylabel('y (Å)')
ax.legend(loc='upper right')

# Colorbar on the right with enough space
cbar = fig.colorbar(top, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Out-of-plane height z (Å)', rotation=270, labelpad=20)


title = ax.set_title('', pad=20)   


plt.subplots_adjust(top=0.83, bottom=0.1, left=0.1, right=0.80)


# Animation

def update(frame):
    theta = frame
    x_top, y_top, z = relaxed_top_layer(theta, chiral_top=1)  # Same chirality; try -1 for heterochiral

    top.set_offsets(np.c_[x_top, y_top])
    top.set_array(z)

    title.set_text(f'Realistic Twisted Bilayer α-Tellurene\n'
                   f'Twist angle θ = {theta:.2f}°\n'
                   f'Buckled honeycomb approx. + chiral buckling + GKK relaxation\n'
                   f'(Incorporate multilayer from arXiv:2301.01777 for future extensions)')
    return top, title

angles = np.linspace(0, 180, 900)      # smooth 0° → 180° rotation
anim = FuncAnimation(fig, update, frames=angles, interval=80, blit=False, repeat=True)

plt.show()