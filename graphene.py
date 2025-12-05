import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


# 1. USER-EDITABLE SETTINGS


initial_angle = 0          # Starting twist angle (degrees)
rotation_speed = 0.10      # Rotation speed (degrees per frame)
N = 20                     # Number of unit cells in each direction
a = 1.0                    # Lattice unit length (arbitrary for visualization; physical a_graphene=0.246 nm used in formulas)
frames = 300               # Total number of animation frames
interval = 50              # Time (ms) between frames


# 2. Material model: λ(θ), Bandwidth(θ), Resistance(θ), phase


# Physical lattice constant for graphene (sources: Wikipedia on Graphene, PMC/NIH review)
a_graphene = 0.246  # nm

# Magic angle ~1.1° (sources: MIT Physics, Wikipedia Twistronics, Princeton News, ALS/Berkeley Lab)
theta_magic = 1.1

def moire_wavelength(theta_deg):
    """Standard formula: λ = a / (2 sin(θ/2)) (sources: PNAS tunable moiré bands, ScienceDirect electrostatic in TBG)"""
    theta = np.radians(theta_deg)
    if abs(theta) < 1e-6:
        return np.nan
    return a_graphene / (2 * np.sin(theta / 2))

def bandwidth(theta, W_min=0.002, beta=0.5):
    """Phenomenological model for moiré bandwidth: minimum near magic angle where bands flatten.
    Real values: near magic, W ~ few meV (sources: Nature Comm. imaging moiré, Phys. Rev. X)"""
    return W_min + beta * (theta - theta_magic)**2  # eV (simplified)

def resistance(theta, R_base=120, alpha=10000, epsilon=0.01):
    """Phenomenological model for resistance: peak near magic angle due to correlated insulator.
    (Inverted from original for realism; real TBG shows high R in insulating states at magic/half-filling.
    Sources: Science paper on tuning superconductivity (Fig.1B resistance map), Phys. Rev. Lett. global phase diagram.
    Note: Real resistance is strongly doping-dependent; this is a 1D simplification vs twist angle."""
    return R_base + alpha / ((theta - theta_magic)**2 + epsilon)  # Ω (peak at magic)

def classify_phase(theta):
    """Simplified phase classification near magic angle (sources: Physics Today gallery of phases, Phys. Rev. Lett. phase diagram).
    Real phases depend on doping (density n) and temperature; e.g., superconductor near ±n_s/2, insulator at half-filling.
    Here: Insulator at exact magic, super nearby, semimetal elsewhere."""
    if abs(theta - theta_magic) < 0.05:
        return "Correlated Insulator"
    if 0.9 <= theta <= 1.3:
        return "Superconducting (with doping)"
    return "Semimetal"


# 3. Graphene lattice generator (standard honeycomb)


def graphene_lattice(a=1, N=20):
    """Generates 2D graphene lattice coordinates using standard vectors.
    Vectors: a1 = [3/2 a, sqrt(3)/2 a], a2 = [3/2 a, -sqrt(3)/2 a] (armchair orientation).
    Basis: [0,0] and [a/2, sqrt(3)/2 a].
    Sources: pybinding tutorial, VASP phonon tutorial for graphene structure."""
    a1 = np.array([a * 3/2, a * np.sqrt(3)/2])
    a2 = np.array([a * 3/2, -a * np.sqrt(3)/2])
    basis = [np.array([0, 0]), np.array([a / 2, a * np.sqrt(3)/2])]
    
    pts = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for b in basis:
                pts.append(i * a1 + j * a2 + b)
    return np.array(pts)


# 4. Rotation function


def rotate(points, angle_deg):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return points @ R.T


# 5. Generate lattices


layer1 = graphene_lattice(a, N)      # Fixed bottom layer
layer2_original = graphene_lattice(a, N)  # Rotating top layer


# 6. Animation setup


fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect("equal")
# Adjusted limits for better moiré visibility; units arbitrary (scale to nm by multiplying by a_graphene if needed)
ax.set_xlim(-30,30)
ax.set_ylim(-30,30)
ax.set_xlabel('x (arbitrary units)')
ax.set_ylabel('y (arbitrary units)')

scatter1 = ax.scatter([], [], s=5, color="blue", label="Layer 1 (fixed)")
scatter2 = ax.scatter([], [], s=5, color="red", label="Layer 2 (rotating)")

# Text for twist angle
title_text = ax.text(0.5, 1.02, "", transform=ax.transAxes,
                     ha="center", fontsize=12)
# Text for Moiré wavelength
moire_text = ax.text(0.5, 1.08, "", transform=ax.transAxes,
                     ha="center", fontsize=12)

ax.legend(loc="upper right")


# 7. Animation update function


def update(i):
    angle = initial_angle + i * rotation_speed
    rotated_layer = rotate(layer2_original, angle)
    
    # Update scatter
    scatter1.set_offsets(layer1)
    scatter2.set_offsets(rotated_layer)
    
    # Update live texts
    title_text.set_text(f"Twist Angle = {angle:.2f}°")
    lam = moire_wavelength(angle)
    moire_text.set_text(f"Moiré λ = {lam:.3f} nm" if not np.isnan(lam) else "Moiré λ = ∞ (aligned)")
    
    return scatter1, scatter2, title_text, moire_text


# 8. Run animation


ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
plt.show()


# 9. Phase Diagram + Resistance/Bandwidth Curves


theta_range = np.linspace(0.01, 3, 400)  # Avoid zero for wavelength
lambda_list = [moire_wavelength(t) for t in theta_range]
res_list = [resistance(t) for t in theta_range]
bw_list = [bandwidth(t) for t in theta_range]
phase_color = ['red' if classify_phase(t)=="Superconducting (with doping)"
               else 'orange' if classify_phase(t)=="Correlated Insulator"
               else 'blue'
               for t in theta_range]

fig2, ax2 = plt.subplots(figsize=(8,5))
ax3 = ax2.twinx()  # Twin axis for bandwidth

# Resistance on left axis (peak at magic)
ax2.scatter(theta_range, res_list, c=phase_color, s=6, label='Resistance')
ax2.set_ylabel("Resistance (Ω)", color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Bandwidth on right axis (minimum at magic)
ax3.plot(theta_range, bw_list, color='green', linewidth=2, label='Bandwidth')
ax3.set_ylabel("Bandwidth (eV)", color='green')
ax3.tick_params(axis='y', labelcolor='green')

ax2.set_xlabel("Twist Angle (degrees)")
ax2.set_title("Resistance, Bandwidth, and Phase Diagram\n(Note: Simplified; real phases doping-dependent)")

# Legend
legend_elements = [
    Line2D([0],[0], marker='o', color='w', label='Superconducting (with doping)',
           markerfacecolor='red', markersize=8),
    Line2D([0],[0], marker='o', color='w', label='Correlated Insulator',
           markerfacecolor='orange', markersize=8),
    Line2D([0],[0], marker='o', color='w', label='Semimetal',
           markerfacecolor='blue', markersize=8),
    Line2D([0],[0], color='green', linewidth=2, label='Bandwidth')
]
ax2.legend(handles=legend_elements, title="Phase / Property")
plt.show()