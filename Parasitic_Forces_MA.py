import opensim as osim
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Model & analysis settings
# -----------------------------
model_path  = "/Users/cobiora/Documents/OpenSim/4.5/Models/Data S1 - Musculoskeletal models/Dimetrodon/Dimetrodon_Hindlimb_Model_var1.osim"
joint_name  = "R_hip_depression"   # coordinate of interest
muscle_name = "R_ITa"              # hip extensor

# Posture sweep (in deg → rad)
angles_deg = np.linspace(-90, 20, 20)
angles_rad = np.deg2rad(angles_deg)

# Biomechanical parameters (adapt to your species / paper)
body_mass = 40.15    # kg, total body mass (example)
m_eff     = 3.6      # kg, effective limb mass
F_max     = 46.706   # N, max isometric muscle force (from model / paper)
L_opt     = 0.01     # m, optimal fibre length
eps_max   = 0.4      # max strain
g         = 9.81     # m/s^2
vmax_fact = 0.10     # vmax ≈ 10 * L_opt (Hill-ish simplification)

# -----------------------------
# 2. Load model & get handles
# -----------------------------
model = osim.Model(model_path)
state = model.initSystem()

muscle = model.getMuscles().get(muscle_name)
coord  = model.getCoordinateSet().get(joint_name)

hip_joint   = model.getJointSet().get("R_hip")
foot_body   = model.getBodySet().get("R_pes")
foot_station = osim.Vec3(0.0, -0.15, 0.0)   # contact point in pes frame (adjust if needed)

grf_dir = np.array([0.0, 1.0, 0.0])  # assume vertical GRF

# -----------------------------
# 3. Compute force-based MA: G = r_m / R_GRF
# -----------------------------
moment_arms = []
Rgrf_list   = []

for ang in angles_rad:
    coord.setValue(state, ang)
    model.realizePosition(state)
    
    # Muscle moment arm (input lever arm)
    r_m = muscle.computeMomentArm(state, coord)  # m
    moment_arms.append(abs(r_m))
    
    # Hip & foot positions in ground
    hip_center  = hip_joint.getParentFrame().getPositionInGround(state)
    foot_pos    = foot_body.findStationLocationInGround(state, foot_station)
    r_vec       = np.array([foot_pos[i] - hip_center[i] for i in range(3)])
    
    # GRF moment arm about hip (distance from hip to GRF line of action)
    R_grf = np.linalg.norm(np.cross(r_vec, grf_dir))
    Rgrf_list.append(R_grf)

moment_arms = np.array(moment_arms)   # r_m
Rgrf_list   = np.array(Rgrf_list)     # R_GRF
G           = moment_arms / Rgrf_list  # force-based mechanical advantage

# -----------------------------
# 4. Muscle work & kinetic energy vs G
# -----------------------------
delta_max = eps_max * L_opt
W_m       = F_max * delta_max          # J, total muscle work capacity
v_max     = vmax_fact * L_opt          # m/s, max shortening velocity

#############################################
Gamma1 = W_m / (m_eff * v_max**2)

# kappa_hat (ungeared reduced parasitic energy)
kappa_hat = (body_mass * g * L_opt) / W_m

# ---------------------------------------------
# Transmission efficiency via Eq. 15
# ---------------------------------------------
eta_eq15 = Gamma1 / (G**2 + kappa_hat * G)
eta_eq15 = eta_eq15 / np.max(eta_eq15)
#############################################

# Output velocity scales as v_out = v_max / G (for a simple lever)
v_out = v_max / G
K     = 0.5 * m_eff * v_out**2         # J, kinetic energy of output mass

# Baseline efficiency without parasitic forces
eta_no_parasitic = K / W_m
eta_no_parasitic /= np.max(eta_no_parasitic)

# -----------------------------
# 5. Include parasitic losses via κ (gravity as parasitic force)
#    κ_g = (m_body * g) / (F_max * G)
# -----------------------------
P_g    = body_mass * g                 # N, gravitational load
kappa  = P_g / (F_max * G)             # dimensionless reduced parasitic force
kappa_clipped = np.minimum(kappa, 1.0) # cap at 1: at κ>=1, no net forward work

# Effective kinetic energy after losses
K_eff = K * (1.0 - kappa_clipped)

# Transmission efficiency including parasitics
eta = K_eff / W_m
eta /= np.max(eta)                     # normalize to [0,1]

# -----------------------------
# 6. Report optimal gearing
# -----------------------------
opt_idx_no_par = np.argmax(eta_no_parasitic)
opt_idx_par    = np.argmax(eta)
opt_idx_par_eq15 = np.argmax(eta_eq15)

print("Force-based MA values G:", G)
print(f"G_opt (no parasitic loss)      = {G[opt_idx_no_par]:.3f}")
print(f"G_opt (with parasitic losses)  = {G[opt_idx_par]:.3f}")
print(f"G_opt (with parasitic losses, eq.15)  = {G[opt_idx_par_eq15]:.3f}")

# -----------------------------
# 7. Plot η(G) with and without κ
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(G, eta_no_parasitic, "o--", label="η(G) without parasitic losses")
#plt.plot(G, eta, "o-", label="η(G) with gravity losses (κ_g)")
plt.plot(G, eta_eq15, "o-", label="η(G) with gravity losses (κ_g) (Eq. 15)")
#plt.axvline(G[opt_idx_par], linestyle=":", label=f"G_opt ≈ {G[opt_idx_par]:.3f}")
plt.xlabel("Force-based mechanical advantage  G = r_m / R_GRF")
plt.ylabel("Normalized transmission efficiency  η")
plt.title("Dimetrodon hindlimb – optimal gearing with parasitic losses")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Dimetrodon_eta_vs_G_with_kappa.png", dpi=300)
plt.show()
