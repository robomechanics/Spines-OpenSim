import opensim as osim
import numpy as np
import matplotlib.pyplot as plt

G_CONST = 9.81

# ----------------------------
# Core function
# ----------------------------
def compute_force_based_landscape(
    model_path: str,
    joint_name: str,
    muscle_name: str,
    angles_deg: np.ndarray,
    Fmax: float,
    m_eff: float,
    L_opt: float,
    eps_max: float = 0.30,
    epsdot_max: float = 10.0,
    foot_body_name: str = "R_pes",
    hip_joint_name: str = "R_hip",
    foot_station_body: tuple = (0.0, -0.15, 0.0),
    grf_dir: np.ndarray = np.array([0.0, 1.0, 0.0]),
    # environment settings
    use_gravity_external_force: bool = True,
    Fe_override=None,   # float or None; None means use defaults (see below)
    debug: bool = False,
):
    """
    Returns:
      G(theta): force-based mechanical advantage from OpenSim geometry
      K_norm_max(theta): Figure-5 style normalized kinetic energy capacity at each theta
      G_star: G at which K_norm_max is maximal (among sampled angles)

    Gravity/constant-force case:
      K_norm_max(G) = min(gamma1/G^2, 1 - kappa1/G)
      where:
        Wmax = Fmax * delta_max
        delta_max = eps_max * L_opt
        vmax = epsdot_max * L_opt
        gamma1 = m_eff * vmax^2 / (2*Wmax)
        kappa1 = Fe * delta_max / Wmax
    """

    model = osim.Model(model_path)
    state = model.initSystem()

    muscle = model.getMuscles().get(muscle_name)
    coord  = model.getCoordinateSet().get(joint_name)

    angles_rad = np.deg2rad(angles_deg)

    # --- muscle moment arm r_m(theta)
    r_m_list = []
    for ang in angles_rad:
        coord.setValue(state, float(ang))
        model.realizePosition(state)
        r_m_list.append(abs(muscle.computeMomentArm(state, coord)))
    r_m_list = np.array(r_m_list, dtype=float)

    # --- GRF moment arm about hip R_GRF(theta)
    hip_joint = model.getJointSet().get(hip_joint_name)
    foot_body = model.getBodySet().get(foot_body_name)
    foot_station = osim.Vec3(*foot_station_body)

    Rgrf_list = []
    for ang in angles_rad:
        coord.setValue(state, float(ang))
        model.realizePosition(state)

        hip_center = hip_joint.getParentFrame().getPositionInGround(state)
        foot_pos   = foot_body.findStationLocationInGround(state, foot_station)

        r_vec = np.array([foot_pos[i] - hip_center[i] for i in range(3)], dtype=float)
        Rgrf = np.linalg.norm(np.cross(r_vec, grf_dir))
        Rgrf_list.append(Rgrf)

    Rgrf_list = np.array(Rgrf_list, dtype=float)

    # --- mechanical advantage
    eps = 1e-9
    G = r_m_list / (Rgrf_list + eps)

    # --- landscape parameters
    delta_max = eps_max * L_opt
    Wmax = Fmax * delta_max

    vmax = epsdot_max * L_opt
    gamma1 = (m_eff * vmax**2) / (2.0 * Wmax)

    # external force Fe
    if Fe_override is not None:
        Fe = float(Fe_override)
    else:
        # default behavior: if gravity mode is on but you didn't override Fe,
        # fall back to m_eff*g (not recommended for extinct; use Fe_override instead)
        Fe = (m_eff * G_CONST) if use_gravity_external_force else 0.0

    kappa1 = (Fe * delta_max) / Wmax

    gamma = gamma1 / (G**2)
    one_minus_kappa = 1.0 - (kappa1 / G)

    K_norm_max = np.minimum(gamma, one_minus_kappa)

    # Hide infeasible region
    K_norm_max = np.where(K_norm_max > 0, K_norm_max, np.nan)

    # choose optimum among sampled angles
    if np.all(np.isnan(K_norm_max)):
        G_star = np.nan
        if debug:
            print("\n[DEBUG] All NaN landscape")
            print("  model:", model_path.split("/")[-1])
            print("  muscle:", muscle_name, "joint:", joint_name)
            print("  Fmax:", Fmax, "L_opt:", L_opt, "m_eff:", m_eff)
            print("  delta_max:", delta_max, "Wmax:", Wmax)
            print("  gamma1:", gamma1, "kappa1:", kappa1, "Fe:", Fe)
            print("  G range:", np.nanmin(G), "to", np.nanmax(G))
            print("  (1-kappa/G) min/max:", np.nanmin(one_minus_kappa), np.nanmax(one_minus_kappa))
    else:
        opt_idx = int(np.nanargmax(K_norm_max))
        G_star = float(G[opt_idx])

    return G, K_norm_max, G_star


# ----------------------------
# Convenience: pick Fe from body mass
# ----------------------------
def estimate_Fe_from_body_mass(body_mass_kg: float, limb_fraction: float = 0.5) -> float:
    """
    Approximate constant external force that the limb must work against.
    limb_fraction = portion of body weight attributed to the modeled limb.
    e.g. 0.25, 0.5, 1.0 for sensitivity checks.
    """
    return float(body_mass_kg) * G_CONST * float(limb_fraction)


# ----------------------------
# Define animals (ADD body_mass for extinct!)
# ----------------------------
angles_deg = np.linspace(-90, 20, 25)

animals = [
    {
        "name": "Dimetrodon",
        "model_path": "/Users/cobiora/Documents/OpenSim/4.5/Models/Data S1 - Musculoskeletal models/Dimetrodon/Dimetrodon_Hindlimb_Model_var1.osim",
        "joint_name": "R_hip_depression",
        "muscle_name": "R_ITa",
        "Fmax": 212.868,
        "m_eff": 32.599,
        "L_opt": 0.01,
        "body_mass": 200.0,      # <-- PUT YOUR BEST ESTIMATE HERE (kg)
        "limb_fraction": 0.5,    # <-- 0.25–1.0 are reasonable to try
    },
    {
        "name": "Regisaurus",
        "model_path": "/Users/cobiora/Documents/OpenSim/4.5/Models/Data S1 - Musculoskeletal models/Regisaurus/Regisaurus_Hindlimb_Model_var1.osim",
        "joint_name": "R_hip_depression",
        "muscle_name": "R_ITa",
        "Fmax": 46.706,
        "m_eff": 3.6,
        "L_opt": 0.1,
        "body_mass": 12.0,
        "limb_fraction": 0.5,
    },
    # add the rest similarly...
]


# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(7, 5))

for a in animals:
    # preferred: gravity case with Fe from body mass
    Fe = estimate_Fe_from_body_mass(a["body_mass"], a.get("limb_fraction", 0.5))

    G, Kland, G_star = compute_force_based_landscape(
        model_path=a["model_path"],
        joint_name=a["joint_name"],
        muscle_name=a["muscle_name"],
        angles_deg=angles_deg,
        Fmax=a["Fmax"],
        m_eff=a["m_eff"],
        L_opt=a["L_opt"],
        foot_body_name=a.get("foot_body_name", "R_pes"),
        hip_joint_name=a.get("hip_joint_name", "R_hip"),
        foot_station_body=a.get("foot_station_body", (0.0, -0.15, 0.0)),
        use_gravity_external_force=True,
        Fe_override=Fe,         # <-- key change
        debug=True,
    )

    # Fallback if still infeasible: inertial-only (kappa=0)
    if np.isnan(G_star):
        print(f"{a['name']}: gravity case infeasible with Fe={Fe:.2f} N; falling back to inertial-only.")
        G, Kland, G_star = compute_force_based_landscape(
            model_path=a["model_path"],
            joint_name=a["joint_name"],
            muscle_name=a["muscle_name"],
            angles_deg=angles_deg,
            Fmax=a["Fmax"],
            m_eff=a["m_eff"],
            L_opt=a["L_opt"],
            foot_body_name=a.get("foot_body_name", "R_pes"),
            hip_joint_name=a.get("hip_joint_name", "R_hip"),
            foot_station_body=a.get("foot_station_body", (0.0, -0.15, 0.0)),
            use_gravity_external_force=False,
            Fe_override=None,
            debug=True,
        )

    # Sort by G for nicer plotting (angles won't be monotonic in G)
    order = np.argsort(G)
    Gs = G[order]
    Ks = Kland[order]

    plt.plot(Gs, Ks, marker="o", linewidth=2, label=f"{a['name']} (G*={G_star:.3g})")

    if not np.all(np.isnan(Kland)):
        opt_i = int(np.nanargmax(Kland))
        plt.scatter([G[opt_i]], [Kland[opt_i]], s=70)

    print(f"{a['name']}: G* = {G_star}")

plt.xscale("log")
plt.xlabel("Mechanical advantage (force-based)  G = r_m / R_GRF")
plt.ylabel("Normalized kinetic energy capacity, $K_{norm,max}$")
plt.title("Figure-5 style optimal gearing landscapes (gravity with Fe override)")
plt.grid(True, which="major", linewidth=0.5)
plt.grid(False, which="minor")
plt.legend()
plt.tight_layout()
plt.savefig("multi_species_landscape_vs_G.png", dpi=300)
plt.show()