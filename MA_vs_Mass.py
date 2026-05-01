import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
EXCEL_PATH = "/Users/cobiora/spines_research/bishop_et_al.__table_s1.xlsx"

HIND_SHEET = "Raw data - hindlimb"
FORE_SHEET = "Raw data - forelimb"

RHO = 1060.0       # kg/m^3
SIGMA_MAX = 3.0e5  # Pa
EPS_MAX = 0.30
EPSDOT_MAX = 10.0  # 1/s
G_CONST = 9.81

# Mechanical advantage range
G_vals = np.logspace(-3, 2.5, 1200)

# ============================================================
# MANUAL MASS OVERRIDES
# Put species masses here if you want to replace the spreadsheet value.
# Leave empty if you want to use the mass already in the sheet.
# Units: kg
# ============================================================
MANUAL_MASS = {
    # "Homo sapiens": 70.0,
    # "Alligator mississippiensis": 90.0,
    # "Varanus komodoensis": 45.0,
}

# ============================================================
# BASIC HELPERS
# ============================================================
def external_force_gravity(body_mass_kg: float) -> float:
    return body_mass_kg * G_CONST


def find_col(df: pd.DataFrame, contains: str) -> str:
    hits = [c for c in df.columns if contains.lower() in str(c).lower()]
    if not hits:
        raise KeyError(
            f"Couldn't find a column containing '{contains}'.\n"
            f"Available columns:\n{list(df.columns)}"
        )
    return hits[0]


def map_group_to_3(g: str):
    g = str(g).strip().lower()

    if ("biped" in g) or ("bird" in g) or ("avian" in g):
        return "bipeds"

    if "mammal" in g:
        return "mammals"

    if g == "reptile":
        return "reptiles"

    if (
        ("reptile" in g)
        or ("croc" in g)
        or ("alligator" in g)
        or ("lizard" in g)
        or ("snake" in g)
        or ("turtle" in g)
        or ("lepidosaur" in g)
    ):
        return "reptiles"

    return None


# ============================================================
# LANDSCAPE CALCULATION
# ============================================================
def compute_landscape_from_pcsa_fasc(body_mass: float, pcsa: float, fasc_len: float) -> np.ndarray:
    """
    Figure-5 style landscape:
      K_norm,max(G) = min(gamma1/G^2, 1 - kappa1/G)
    """
    Fmax = SIGMA_MAX * pcsa
    Wmax = Fmax * (EPS_MAX * fasc_len)
    vmax = fasc_len * EPSDOT_MAX
    Fe = external_force_gravity(body_mass)

    gamma1 = (body_mass * vmax**2) / (2.0 * Wmax)
    kappa1 = (Fe * (EPS_MAX * fasc_len)) / Wmax

    gamma = gamma1 / (G_vals**2)
    one_minus_kappa = 1.0 - (kappa1 / G_vals)

    K_norm_max = np.minimum(gamma, one_minus_kappa)
    K_norm_max = np.where(K_norm_max > 0, K_norm_max, np.nan)
    return K_norm_max


def find_optimal_gearing(body_mass: float, pcsa: float, fasc_len: float):
    """
    Returns:
        peak_G, peak_K
    """
    K = compute_landscape_from_pcsa_fasc(body_mass, pcsa, fasc_len)

    if np.all(np.isnan(K)):
        return np.nan, np.nan

    idx = np.nanargmax(K)
    return G_vals[idx], K[idx]


# ============================================================
# LOAD + CLEAN ONE SHEET
# ============================================================
def load_and_clean_sheet(excel_path: str, sheet_name: str):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

    print(f"\n--- Loading sheet: {sheet_name} ---")
    print("After read:", df.shape)

    col_species = find_col(df, "species")
    col_group = find_col(df, "group")
    col_muscle = find_col(df, "muscle")
    col_bodym = find_col(df, "body mass")
    col_muscle_mass = find_col(df, "muscle mass")
    col_pennation = find_col(df, "pennation")
    col_fasc_len = find_col(df, "fascicle length")

    for c in [col_bodym, col_muscle_mass, col_pennation, col_fasc_len]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            col_species,
            col_group,
            col_muscle,
            col_bodym,
            col_muscle_mass,
            col_pennation,
            col_fasc_len,
        ]
    )
    df = df[(df[col_bodym] > 0) & (df[col_muscle_mass] > 0) & (df[col_fasc_len] > 0)]
    print("After drop/filter:", df.shape)

    theta = np.deg2rad(df[col_pennation])
    df["PCSA (m2)"] = (df[col_muscle_mass] * np.cos(theta)) / (RHO * df[col_fasc_len])
    df["PCSA (m2)"] = pd.to_numeric(df["PCSA (m2)"], errors="coerce")
    df = df.dropna(subset=["PCSA (m2)"])
    df = df[df["PCSA (m2)"] > 0]
    print("After PCSA recompute:", df.shape)

    df["group3"] = df[col_group].apply(map_group_to_3)
    unmapped = sorted(
        df.loc[df["group3"].isna(), col_group]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )
    if unmapped:
        print("WARNING: unmapped group labels:", unmapped)

    df = df.dropna(subset=["group3"])
    print("Groups present:", sorted(df["group3"].unique()))

    cols = {
        "species": col_species,
        "group": col_group,
        "muscle": col_muscle,
        "bodym": col_bodym,
        "muscle_mass": col_muscle_mass,
        "pennation": col_pennation,
        "fasc_len": col_fasc_len,
    }

    return df, cols


# ============================================================
# FULL-LIMB AGGREGATION
# ============================================================
def aggregate_full_limb(df: pd.DataFrame, cols: dict, limb_label: str) -> pd.DataFrame:
    """
    Aggregate all muscles for each species into one effective limb.

    Effective values:
      - limb_pcsa = sum of PCSAs
      - limb_fasc_len = PCSA-weighted mean fascicle length
      - body_mass = manual override if present, otherwise first sheet value
    """
    species_col = cols["species"]
    bodym_col = cols["bodym"]
    fasc_col = cols["fasc_len"]
    muscle_col = cols["muscle"]

    rows = []

    for (grp, sp), sdf in df.groupby(["group3", species_col], sort=True):
        sdf = sdf.copy()

        total_pcsa = sdf["PCSA (m2)"].sum()
        if total_pcsa <= 0:
            continue

        weighted_fasc_len = np.average(sdf[fasc_col], weights=sdf["PCSA (m2)"])

        if sp in MANUAL_MASS:
            body_mass = MANUAL_MASS[sp]
        else:
            body_mass = sdf[bodym_col].iloc[0]

        rows.append(
            {
                "group3": grp,
                "species": sp,
                "limb": limb_label,
                "body_mass": body_mass,
                "limb_pcsa": total_pcsa,
                "limb_fasc_len": weighted_fasc_len,
                "muscle_count": len(sdf),
                "muscles_used": "; ".join(sdf[muscle_col].astype(str).tolist()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["group3", "species"]).reset_index(drop=True)
    return out


# ============================================================
# ADD OPTIMAL GEARING
# ============================================================
def add_optimal_gearing_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    peak_G = []
    peak_K = []

    for _, row in out.iterrows():
        g_star, k_star = find_optimal_gearing(
            body_mass=row["body_mass"],
            pcsa=row["limb_pcsa"],
            fasc_len=row["limb_fasc_len"],
        )
        peak_G.append(g_star)
        peak_K.append(k_star)

    out["optimal_G"] = peak_G
    out["peak_K"] = peak_K
    return out


# ============================================================
# PLOTTING
# ============================================================
def plot_mass_vs_optimal_gearing(df: pd.DataFrame, limb_name: str):
    """
    One graph for one limb type.
    """
    if df.empty:
        print(f"No data to plot for {limb_name}")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(df["body_mass"], df["optimal_G"])

    for _, row in df.iterrows():
        plt.annotate(
            row["species"],
            (row["body_mass"], row["optimal_G"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Body mass (kg)")
    plt.ylabel("Optimal gearing, $G^*$")
    plt.title(f"{limb_name.capitalize()}limb: Optimal Gearing vs Body Mass")
    plt.grid(True, which="major", linewidth=0.5)
    plt.grid(False, which="minor")
    plt.tight_layout()
    plt.show()


def plot_mass_vs_optimal_gearing_by_group(df: pd.DataFrame, limb_name: str):
    """
    Optional grouped version for easier interpretation.
    """
    if df.empty:
        return

    for grp in ["reptiles", "mammals", "bipeds"]:
        gdf = df[df["group3"] == grp].copy()
        if gdf.empty:
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(gdf["body_mass"], gdf["optimal_G"])

        for _, row in gdf.iterrows():
            plt.annotate(
                row["species"],
                (row["body_mass"], row["optimal_G"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Body mass (kg)")
        plt.ylabel("Optimal gearing, $G^*$")
        plt.title(f"{grp.capitalize()} — {limb_name.capitalize()}limb: Optimal Gearing vs Body Mass")
        plt.grid(True, which="major", linewidth=0.5)
        plt.grid(False, which="minor")
        plt.tight_layout()
        plt.show()


def plot_hind_and_fore_together(hind_df: pd.DataFrame, fore_df: pd.DataFrame):
    """
    One combined graph with hind and fore plotted together.
    """
    plt.figure(figsize=(9, 6))

    if not hind_df.empty:
        plt.scatter(hind_df["body_mass"], hind_df["optimal_G"], marker="o", label="Hindlimb")
    if not fore_df.empty:
        plt.scatter(fore_df["body_mass"], fore_df["optimal_G"], marker="s", label="Forelimb")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Body mass (kg)")
    plt.ylabel("Optimal gearing, $G^*$")
    plt.title("Optimal Gearing vs Body Mass")
    plt.grid(True, which="major", linewidth=0.5)
    plt.grid(False, which="minor")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    # Load sheets
    hind_raw, hind_cols = load_and_clean_sheet(EXCEL_PATH, HIND_SHEET)
    fore_raw, fore_cols = load_and_clean_sheet(EXCEL_PATH, FORE_SHEET)

    # Aggregate full limbs
    hind_agg = aggregate_full_limb(hind_raw, hind_cols, limb_label="hind")
    fore_agg = aggregate_full_limb(fore_raw, fore_cols, limb_label="fore")

    # Compute optimal gearing
    hind_summary = add_optimal_gearing_columns(hind_agg)
    fore_summary = add_optimal_gearing_columns(fore_agg)

    print("\n============================================================")
    print("HINDLIMB SUMMARY")
    print("============================================================")
    print(
        hind_summary[
            ["group3", "species", "body_mass", "limb_pcsa", "limb_fasc_len", "optimal_G", "peak_K"]
        ]
    )

    print("\n============================================================")
    print("FORELIMB SUMMARY")
    print("============================================================")
    print(
        fore_summary[
            ["group3", "species", "body_mass", "limb_pcsa", "limb_fasc_len", "optimal_G", "peak_K"]
        ]
    )

    # Save results
    hind_summary.to_csv("hindlimb_optimal_gearing_vs_mass.csv", index=False)
    fore_summary.to_csv("forelimb_optimal_gearing_vs_mass.csv", index=False)

    print("\nSaved:")
    print("  hindlimb_optimal_gearing_vs_mass.csv")
    print("  forelimb_optimal_gearing_vs_mass.csv")

    # Plots
    plot_mass_vs_optimal_gearing(hind_summary, "hind")
    plot_mass_vs_optimal_gearing(fore_summary, "fore")

    # Optional extra plots
    plot_hind_and_fore_together(hind_summary, fore_summary)
    plot_mass_vs_optimal_gearing_by_group(hind_summary, "hind")
    plot_mass_vs_optimal_gearing_by_group(fore_summary, "fore")


if __name__ == "__main__":
    main()