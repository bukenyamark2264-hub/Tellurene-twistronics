import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Twisted Bilayer α-Tellurene
# Geometric + Interlayer Physics + Relaxation
# Continuum phenomenological model
# ============================================================

# -----------------------------
# 1. Physical Parameters
# -----------------------------

a0 = 4.20                 # lattice constant (Å) – approximate
grid_size = 150          # visualization window (Å)
N = 30                   # number of unit cells

V0 = 1.0                 # stacking energy amplitude
alpha_relax = 0.15       # in-plane relaxation strength
z_amp = 1.0              # out-of-plane corrugation amplitude


# -----------------------------
# 2. Build Base Hexagonal Lattice
# -----------------------------

def build_lattice(N, a):
    x, y = [], []
    a1 = np.array([a, 0])
    a2 = np.array([a/2, a*np.sqrt(3)/2])

    for i in range(-N, N+1):
        for j in range(-N, N+1):
            r = i*a1 + j*a2
            x.append(r[0])
            y.append(r[1])
    return np.array(x), np.array(y)


x1, y1 = build_lattice(N, a0)        #my bottom layer coordinates


# -----------------------------
# 3. Rotation Function
# -----------------------------

def rotate(x, y, theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    xr = c*x - s*y
    yr = s*x + c*y
    return xr, yr


# -----------------------------
# 4. Moiré Wavelength
# -----------------------------

def moire_wavelength(theta_deg, a):
    if abs(theta_deg) < 0.01: # abs represents the absolute value of the angle to avoid division dy zero.
        theta_deg = 0.01
    theta = np.radians(theta_deg)
    return a / (2 * np.sin(theta/2)) # DO NOT FORGET TO LOOK HERE


# -----------------------------
# 5. Stacking Energy Field   Potential Energy Surface (PES): the landscape of energy that atoms feel when they sit on top of each other
# -----------------------------

def stacking_energy(x, y, lambda_m):
    G = 2 * np.pi / lambda_m

    # Two reciprocal directions (hexagonal approx)
    G1 = np.array([G, 0])
    G2 = np.array([G/2, G*np.sqrt(3)/2])

    phi1 = G1[0]*x + G1[1]*y
    phi2 = G2[0]*x + G2[1]*y

    E = V0 * (np.cos(phi1) + np.cos(phi2) + np.cos(phi1 - phi2)) # Energy field construction
    return E  # It builds a smooth periodic energy landscape representing how favorable stacking is at each position.


# -----------------------------
# 6. In-Plane Relaxation
# -----------------------------

def relax_positions(x, y, E, lambda_m):
    G = 2 * np.pi / lambda_m

    # approximate gradient of energy
    dEx = -G * np.sin(G * x)
    dEy = -G * np.sin(G * y)

    x_rel = x - alpha_relax * dEx
    y_rel = y - alpha_relax * dEy

    return x_rel, y_rel


# -----------------------------
# 7. Out-of-Plane Corrugation
# -----------------------------

def corrugation(E):
    # Higher stacking energy → larger height
    z = z_amp * (E / np.max(np.abs(E)))
    return z


# -----------------------------
# 8. Plot Setup
# -----------------------------

fig, ax = plt.subplots(figsize=(9, 8))

bottom = ax.scatter(x1, y1, c='black', s=5, alpha=0.7, label='Bottom Layer')
top = ax.scatter([], [], c=[], cmap='coolwarm', s=10)

ax.set_aspect('equal')
ax.set_xlim(-grid_size, grid_size)
ax.set_ylim(-grid_size, grid_size)
ax.set_xlabel('x (Å)')
ax.set_ylabel('y (Å)')
ax.legend(loc='upper right')

cbar = fig.colorbar(top, ax=ax)
cbar.set_label("Stacking Energy / Height")

title = ax.set_title("")


# -----------------------------
# 9. Animation Update
# -----------------------------

def update(frame):

    theta = frame

    # Rotate top layer
    x2, y2 = rotate(x1, y1, theta)

    # Compute Moiré wavelength
    lambda_m = moire_wavelength(theta, a0)

    # Compute stacking energy
    E = stacking_energy(x2, y2, lambda_m)

    # Relax positions
    x2_rel, y2_rel = relax_positions(x2, y2, E, lambda_m)

    # Out-of-plane height
    z = corrugation(E)

    # Update scatter
    top.set_offsets(np.c_[x2_rel, y2_rel])
    top.set_array(z)

    title.set_text(
        f"Twisted Bilayer α-Tellurene\n"
        f"Twist Angle θ = {theta:.2f}°\n"
        f"Continuum Stacking + Relaxation Model"
    )

    return top, title


# -----------------------------
# 10. Run Animation
# -----------------------------

angles = np.linspace(0.5, 10, 300)  # small angles more interesting
anim = FuncAnimation(fig, update, frames=angles,
                     interval=80, blit=False, repeat=True)

# ================================
# Moiré Wavelength vs Twist Angle
# ================================

theta_vals = np.linspace(0.5, 10, 100)

lambda_vals = [moire_wavelength(t, a0) for t in theta_vals]

plt.figure(figsize=(6,5))
plt.plot(theta_vals, lambda_vals)

plt.xlabel("Twist Angle (degrees)")
plt.ylabel("Moiré Wavelength (Å)")
plt.title("Moiré Wavelength vs Twist Angle")

plt.grid(True)
plt.savefig("moire_wavelength_vs_angle.png", dpi=600, bbox_inches='tight')

plt.show()
plt.show()