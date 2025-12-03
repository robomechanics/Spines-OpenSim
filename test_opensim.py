import opensim as osim
import numpy as np

model = osim.Model("/Users/cobiora/Documents/OpenSim/4.5/Models/Data S1 - Musculoskeletal models/Dimetrodon/Dimetrodon_Hindlimb_Model_var1.osim")
state = model.initSystem()

coord_name = "R_hip_depression"
coord = model.updCoordinateSet().get(coord_name)

theta0_deg = -20.0
theta0 = np.deg2rad(theta0_deg)
coord.setValue(state, theta0)
model.realizePosition(state)

def vec3_to_np(v):
    return np.array([v.get(0), v.get(1), v.get(2)], dtype=float)

def set_all_activations(model, state, a=1.0):
    for i in range(model.getMuscles().getSize()):
        m = model.updMuscles().get(i)
        m.setActivation(state, a)

def joint_muscle_torque(model, state, coord, coord_name):
    """
    Sum muscle torques about a given coordinate.
    τ = Σ (moment_arm_i * force_i)
    """
    model.realizeDynamics(state)

    tau = 0.0
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        m = muscles.get(i)

        # Moment arm of this muscle about the coordinate
        r = m.computeMomentArm(state, coord)  # m

        # Muscle force (actuation in OpenSim is force for muscles)
        F = m.getActuation(state)             # N

        tau += r * F
    return tau  # N·m

set_all_activations(model, state, a=1.0)
# baseline torque
coord.setValue(state, theta0)
model.realizeDynamics(state)
tau0 = joint_muscle_torque(model, state, coord, coord_name)

# small perturbation
dtheta = np.deg2rad(1.0)  # 1 degree
coord.setValue(state, theta0 + dtheta)
model.realizeDynamics(state)
tau1 = joint_muscle_torque(model, state, coord, coord_name)

# effective stiffness
k_eff = (tau1 - tau0) / dtheta   # N·m/rad
print(f"{coord_name} k_eff at {theta0_deg:.1f}° ≈ {k_eff:.2f} N·m/rad")

hindlimb_coords = ["R_hip_depression", "R_knee_extension", "R_ankle_extension"]

for cname in hindlimb_coords:
    coord = model.updCoordinateSet().get(cname)
    theta0_deg = -40.0  # or pick joint-specific neutral
    theta0 = np.deg2rad(theta0_deg)

    coord.setValue(state, theta0)
    set_all_activations(model, state, 1.0)
    model.realizeDynamics(state)
    tau0 = joint_muscle_torque(model, state, coord, cname)

    dtheta = np.deg2rad(1.0)
    coord.setValue(state, theta0 + dtheta)
    model.realizeDynamics(state)
    tau1 = joint_muscle_torque(model, state, coord, cname)
    print(joint_muscle_torque(model, state, coord, cname))
    k_eff = (tau1 - tau0) / dtheta
    print(f"{cname}: k_eff ≈ {k_eff:.2f} N·m/rad at {theta0_deg:.1f}°")

forces = model.getForceSet()
print("Forces in model:")
for i in range(forces.getSize()):
    f = forces.get(i)
    print(i, f.getConcreteClassName(), f.getName())

clf = osim.CoordinateLimitForce.safeDownCast(f)
if clf:
    print("upper_stiffness:", clf.getUpperStiffness())
    print("lower_stiffness:", clf.getLowerStiffness())
