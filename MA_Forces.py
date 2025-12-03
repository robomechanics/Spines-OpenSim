import opensim as osim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. Load model ----------
model = osim.Model("/Users/cobiora/Documents/OpenSim/4.5/Models/Data S1 - Musculoskeletal models/Dimetrodon/Dimetrodon_Hindlimb_Model_var1.osim")
state = model.initSystem()

# Identify joint and muscle of interest
joint_name = "R_hip_depression"
muscle_name = "R_ITa"

muscle = model.getMuscles().get(muscle_name)
coord = model.getCoordinateSet().get(joint_name)

# ---------- 2. Define posture sweep ----------
angles_deg = np.linspace(-90, 20, 20)
angles_rad = np.deg2rad(angles_deg)

moment_arms = []
L_i = []
L_o = []
force_based_MA = []

# ---------- 3. Compute moment arms ----------
for ang in angles_rad:
    coord.setValue(state, ang)
    model.realizePosition(state)
    r = muscle.computeMomentArm(state, coord)  # input lever arm (m)
    moment_arms.append(r)
    L_i.append(abs(r))

# ---------- 4. Compute geometric output lever arm ----------
hip = model.getJointSet().get("R_hip")
hip_center = hip.getParentFrame().getPositionInGround(state)

foot_body = model.getBodySet().get("R_pes")
foot_station = osim.Vec3(0.0, -0.15, 0.0)
foot_pos = foot_body.findStationLocationInGround(state, foot_station)

bodies = model.getBodySet()
print("Bodies:")
for i in range(bodies.getSize()):
    print(bodies.get(i).getName())

# Perpendicular distance to vertical (proxy for output lever arm)
vertical_dir = np.array([0, 1, 0])
vec = np.array([foot_pos[i] - hip_center[i] for i in range(3)])
Lo_scalar = np.linalg.norm(np.cross(vec, vertical_dir))
L_o = [Lo_scalar] * len(L_i)

# ---------- 5. Force-based mechanical advantage (Blob 2001) ----------
# MA = rm / Rgrf
# where rm = muscle moment arm, Rgrf = GRF moment arm (distance from hip to GRF vector)
# If no GRF vector available, approximate Rgrf ~ Lo_scalar or set a proxy based on body geometry.

# Example GRF direction (assumed vertical ground reaction)
grf_dir = np.array([0, 1, 0])

# Approximate Rgrf for each posture (hip-to-foot vector ⟂ to GRF)
Rgrf_list = []
for ang in angles_rad:
    coord.setValue(state, ang)
    model.realizePosition(state)
    hip_center = hip.getParentFrame().getPositionInGround(state)
    foot_pos = foot_body.findStationLocationInGround(state, foot_station)
    r_vec = np.array([foot_pos[i] - hip_center[i] for i in range(3)])
    Rgrf = np.linalg.norm(np.cross(r_vec, grf_dir))
    Rgrf_list.append(Rgrf)

# Calculate force-based mechanical advantage
force_based_MA = np.array(L_i) / np.array(Rgrf_list)

# ---------- 6. Estimate work & kinetic energy ----------
Fmax = 454.417     # N
L_opt = 0.01       # m
eps_max = 0.4
delta_max = eps_max * L_opt
Wm = Fmax * delta_max           # J

m_eff = 88.489                   # kg
vmax = 10 * L_opt               # m/s
K = 0.5 * m_eff * (vmax / force_based_MA)**2

# ---------- 7. Compute transmission efficiency ----------
eta_raw = K / Wm
eta = eta_raw / np.max(eta_raw)

# ---------- 8. Report optimal gearing ----------
opt_index = np.argmax(eta)
print(f"Optimal gearing (force-based) G_opt = {force_based_MA[opt_index]:.3f}")

# ---------- 9. Plot η vs G ----------
plt.figure(figsize=(6,4))
plt.plot(force_based_MA, eta, 'o-', lw=2)
plt.xlabel("Mechanical advantage (force-based)  G = r_m / R_GRF")
plt.ylabel("Normalized transmission efficiency  η")
plt.title(r"Dimetrodon hindlimb – force-based optimal gearing. $G_{{opt}} = {:.3f}$".format(force_based_MA[opt_index]))
plt.grid(True)
plt.tight_layout()
plt.savefig("Dimetrodon_force_based_eta_vs_G.png", dpi=300)
plt.show()