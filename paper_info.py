import opensim as osim
import numpy as np
import matplotlib.pyplot as plt

# ---------- helper: safe Vec3 -> numpy ----------
def vec3_to_np(v):
    return np.array([v.get(0), v.get(1), v.get(2)], dtype=float)

# ---------- 1. Load model ----------
model = osim.Model("/Users/cobiora/Documents/OpenSim/4.5/Models/Data S1 - Musculoskeletal models/Dimetrodon/Dimetrodon_Hindlimb_Model_var1.osim")
state = model.initSystem()

# Identify joint and muscle of interest
joint_name  = "R_hip_depression"   # coordinate name
muscle_name = "R_ITa"              # muscle name

muscle = model.getMuscles().get(muscle_name)
coord  = model.getCoordinateSet().get(joint_name)

# ============================================================
#   LIMB LENGTHS (using joints and R_pes)
# ============================================================
print("\n---------- LIMB LENGTHS ----------")

# Guess joint names – adjust if your model uses different names
hip_joint   = model.getJointSet().get("R_hip")
knee_joint  = model.getJointSet().get("R_knee")
ankle_joint = model.getJointSet().get("R_ankle")

hip_center   = vec3_to_np(hip_joint.getParentFrame().getPositionInGround(state))
knee_center  = vec3_to_np(knee_joint.getParentFrame().getPositionInGround(state))
ankle_center = vec3_to_np(ankle_joint.getParentFrame().getPositionInGround(state))

# Choose a station on the pes (tweak as needed)
pes_body    = model.getBodySet().get("R_pes")
pes_station = osim.Vec3(0.0, -0.10, 0.0)
pes_pos     = vec3_to_np(pes_body.findStationLocationInGround(state, pes_station))

femur_len = np.linalg.norm(knee_center  - hip_center)
crus_len  = np.linalg.norm(ankle_center - knee_center)
limb_len  = np.linalg.norm(pes_pos      - hip_center)

print(f"Femur length (hip→knee)       : {femur_len:.3f} m")
print(f"Crus length (knee→ankle)      : {crus_len:.3f} m")
print(f"Total limb length (hip→pes)   : {limb_len:.3f} m")


# ============================================================
#   LIMB MASSES & WEIGHTS
# ============================================================
print("\n---------- LIMB MASSES & WEIGHTS ----------")

g = 9.81
hindlimb_bodies = ["R_thigh", "R_crus_tibial", "R_crus_fibular", "R_pes"]

total_mass   = 0.0
total_weight = 0.0

for bname in hindlimb_bodies:
    body = model.getBodySet().get(bname)
    m = body.getMass()
    w = m * g
    print(f"{bname:15s} mass = {m:.4f} kg, weight = {w:.2f} N")
    total_mass   += m
    total_weight += w

print(f"\nTotal hindlimb mass   : {total_mass:.4f} kg")
print(f"Total hindlimb weight : {total_weight:.2f} N")


# ============================================================
#   2. Define posture sweep
# ============================================================
angles_deg = np.linspace(-90, 20, 20)
angles_rad = np.deg2rad(angles_deg)

moment_arms = []
L_i = []
Rgrf_list = []

# ============================================================
#   3. Compute moment arms (input lever arms)
# ============================================================
for ang in angles_rad:
    coord.setValue(state, ang)
    model.realizePosition(state)
    r = muscle.computeMomentArm(state, coord)  # input lever arm (m)
    moment_arms.append(r)
    L_i.append(abs(r))

# ============================================================
#   4. Compute GRF moment arm (output lever arm)
# ============================================================
hip_center = vec3_to_np(hip_joint.getParentFrame().getPositionInGround(state))

foot_body    = model.getBodySet().get("R_pes")
foot_station = osim.Vec3(0.0, -0.15, 0.0)

vertical_dir = np.array([0, 1, 0], dtype=float)

for ang in angles_rad:
    coord.setValue(state, ang)
    model.realizePosition(state)
    hip_center = vec3_to_np(hip_joint.getParentFrame().getPositionInGround(state))
    foot_pos   = vec3_to_np(foot_body.findStationLocationInGround(state, foot_station))
    r_vec      = foot_pos - hip_center
    Rgrf = np.linalg.norm(np.cross(r_vec, vertical_dir))
    Rgrf_list.append(Rgrf)

Rgrf_array = np.array(Rgrf_list)
L_i_array  = np.array(L_i)

force_based_MA = L_i_array / Rgrf_array  # G = r_m / R_GRF

# ============================================================
#   6. Work & kinetic energy
# ============================================================
Fmax     = 212.868       # N
L_opt    = 0.01         # m
eps_max  = 0.4
delta_max = eps_max * L_opt
Wm       = Fmax * delta_max           # J

m_eff = 32.599                            # kg
vmax  = 10 * L_opt                     # m/s
K     = 0.5 * m_eff * (vmax / force_based_MA)**2

# ============================================================
#   7. Transmission efficiency
# ============================================================
eta_raw = K / Wm
eta     = eta_raw / np.max(eta_raw)

# ============================================================
#   8. Optimal gearing
# ============================================================
opt_index = np.argmax(eta)
print(f"\nOptimal gearing (force-based) G_opt = {force_based_MA[opt_index]:.3f}")


# ============================================================
#   NEW: Joint stiffness estimation
# ============================================================
print("\n---------- JOINT STIFFNESS (muscle-driven, finite difference) ----------")

def set_all_activations(model, state, a=1.0):
    muscles = model.updMuscles()
    for i in range(muscles.getSize()):
        muscles.get(i).setActivation(state, a)

def joint_muscle_torque(model, state, coord):
    """
    Net muscle torque about a coordinate:
        τ = Σ (moment_arm_i * muscle_force_i)
    """
    model.realizeDynamics(state)
    tau = 0.0
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        m = muscles.get(i)
        r = m.computeMomentArm(state, coord)  # m
        F = m.getActuation(state)             # N (muscle force)
        tau += r * F
    return tau

# Get list of coordinate names to check which joints exist
coord_set = model.getCoordinateSet()
coord_names = [coord_set.get(i).getName() for i in range(coord_set.getSize())]

# Hindlimb joint coordinates we care about (will skip if not present)
target_coords = [
    "R_hip_depression",
    "R_knee_flexion",
    "R_ankle_extension",   # adjust if it's R_ankle_flexion in your model
]

for cname in target_coords:
    if cname not in coord_names:
        print(f"{cname}: not found in model, skipping.")
        continue

    # fresh state for this joint stiffness estimate
    state = model.initSystem()
    coord = model.updCoordinateSet().get(cname)

    # choose a reference angle (here: -20 deg; you can change)
    theta0_deg = -20.0
    theta0 = np.deg2rad(theta0_deg)
    coord.setValue(state, theta0)

    # set muscle activations (max active stiffness; change if needed)
    set_all_activations(model, state, a=1.0)

    # baseline torque
    tau0 = joint_muscle_torque(model, state, coord)

    # small perturbation
    dtheta_deg = 1.0
    dtheta     = np.deg2rad(dtheta_deg)
    coord.setValue(state, theta0 + dtheta)
    tau1 = joint_muscle_torque(model, state, coord)

    k_eff = (tau1 - tau0) / dtheta  # N·m/rad

    print(f"{cname:18s} k_eff ≈ {k_eff:8.2f} N·m/rad at {theta0_deg:+5.1f}° (Δθ = {dtheta_deg:.1f}°)")


# ============================================================
#   9. Plot η vs G
# ============================================================
plt.figure(figsize=(6,4))
plt.plot(force_based_MA, eta, 'o-', lw=2)
plt.xlabel("Mechanical advantage (force-based)  G = r_m / R_GRF")
plt.ylabel("Normalized transmission efficiency  η")
plt.title(
    fr"Dimetrodon hindlimb – force-based optimal gearing\n"
    fr"$G_{{\mathrm{{opt}}}} = {force_based_MA[opt_index]:.3f}$"
)
plt.grid(True)
plt.tight_layout()
plt.savefig("Dimetrodon_force_based_eta_vs_G.png", dpi=300)
plt.show()
