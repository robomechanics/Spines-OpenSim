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

# Set True if you want one figure per species in addition to group plots
MAKE_SPECIES_PLOTS = True

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
    Figure-5 style landscape in gravity case:
      K_norm,max(G) = min(gamma1/G^2, 1 - kappa1/G)

    Here pcsa and fasc_len can represent either a single muscle or an
    aggregated full-limb effective value.
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
      - limb_pcsa = sum of PCSAs across muscles
      - limb_fasc_len = PCSA-weighted mean fascicle length
      - body_mass = first body mass for species
      - muscle_count = number of muscles used
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


def build_matched_full_limb_table(hind_agg: pd.DataFrame, fore_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only species present in both hind and fore aggregated tables.
    """
    common_species = sorted(set(hind_agg["species"]).intersection(set(fore_agg["species"])))

    if not common_species:
        return pd.DataFrame()

    hind_match = hind_agg[hind_agg["species"].isin(common_species)].copy()
    fore_match = fore_agg[fore_agg["species"].isin(common_species)].copy()

    merged = pd.merge(
        hind_match,
        fore_match,
        on=["group3", "species"],
        suffixes=("_hind", "_fore"),
        how="inner",
    )

    merged = merged.sort_values(["group3", "species"]).reset_index(drop=True)
    return merged


# ============================================================
# PLOTTING
# ============================================================
def plot_group_full_limb_comparison(matched_df: pd.DataFrame, group_name: str):
    gdf = matched_df[matched_df["group3"] == group_name].copy()

    if gdf.empty:
        print(f"No matched full-limb species for group: {group_name}")
        return

    plt.figure(figsize=(10, 6))

    for _, row in gdf.iterrows():
        sp = row["species"]

        Kh = compute_landscape_from_pcsa_fasc(
            body_mass=row["body_mass_hind"],
            pcsa=row["limb_pcsa_hind"],
            fasc_len=row["limb_fasc_len_hind"],
        )

        Kf = compute_landscape_from_pcsa_fasc(
            body_mass=row["body_mass_fore"],
            pcsa=row["limb_pcsa_fore"],
            fasc_len=row["limb_fasc_len_fore"],
        )

        line = plt.plot(G_vals, Kh, linewidth=2, label=f"{sp} hind")[0]
        plt.plot(
            G_vals,
            Kf,
            linewidth=2,
            linestyle="--",
            color=line.get_color(),
            label=f"{sp} fore",
        )

    plt.xscale("log")
    plt.xlabel("Mechanical advantage, G")
    plt.ylabel("Normalized kinetic energy capacity, $K_{norm,max}$")
    plt.title(f"{group_name.capitalize()} — Full Hindlimb vs Full Forelimb")
    plt.grid(True, which="major", linewidth=0.5)
    plt.grid(False, which="minor")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_species_full_limb_comparison(matched_df: pd.DataFrame):
    for _, row in matched_df.iterrows():
        sp = row["species"]

        Kh = compute_landscape_from_pcsa_fasc(
            body_mass=row["body_mass_hind"],
            pcsa=row["limb_pcsa_hind"],
            fasc_len=row["limb_fasc_len_hind"],
        )

        Kf = compute_landscape_from_pcsa_fasc(
            body_mass=row["body_mass_fore"],
            pcsa=row["limb_pcsa_fore"],
            fasc_len=row["limb_fasc_len_fore"],
        )

        plt.figure(figsize=(7, 5))
        plt.plot(G_vals, Kh, linewidth=2, label="Hindlimb")
        plt.plot(G_vals, Kf, linewidth=2, linestyle="--", label="Forelimb")

        plt.xscale("log")
        plt.xlabel("Mechanical advantage, G")
        plt.ylabel("Normalized kinetic energy capacity, $K_{norm,max}$")
        plt.title(f"{sp} — Full Hindlimb vs Full Forelimb")
        plt.grid(True, which="major", linewidth=0.5)
        plt.grid(False, which="minor")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# OPTIONAL NUMERIC SUMMARIES
# ============================================================
def add_peak_metrics(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add peak K and the G value where the peak occurs for hind and fore.
    """
    out_rows = []

    for _, row in matched_df.iterrows():
        Kh = compute_landscape_from_pcsa_fasc(
            body_mass=row["body_mass_hind"],
            pcsa=row["limb_pcsa_hind"],
            fasc_len=row["limb_fasc_len_hind"],
        )
        Kf = compute_landscape_from_pcsa_fasc(
            body_mass=row["body_mass_fore"],
            pcsa=row["limb_pcsa_fore"],
            fasc_len=row["limb_fasc_len_fore"],
        )

        hind_idx = np.nanargmax(Kh) if np.any(~np.isnan(Kh)) else None
        fore_idx = np.nanargmax(Kf) if np.any(~np.isnan(Kf)) else None

        out_rows.append(
            {
                **row.to_dict(),
                "hind_peak_K": Kh[hind_idx] if hind_idx is not None else np.nan,
                "hind_peak_G": G_vals[hind_idx] if hind_idx is not None else np.nan,
                "fore_peak_K": Kf[fore_idx] if fore_idx is not None else np.nan,
                "fore_peak_G": G_vals[fore_idx] if fore_idx is not None else np.nan,
                "delta_peak_K_fore_minus_hind": (
                    (Kf[fore_idx] if fore_idx is not None else np.nan)
                    - (Kh[hind_idx] if hind_idx is not None else np.nan)
                ),
                "delta_peak_G_fore_minus_hind": (
                    (G_vals[fore_idx] if fore_idx is not None else np.nan)
                    - (G_vals[hind_idx] if hind_idx is not None else np.nan)
                ),
            }
        )

    return pd.DataFrame(out_rows)


# ============================================================
# MAIN
# ============================================================
def main():
    # Load both raw sheets
    hind_df, hind_cols = load_and_clean_sheet(EXCEL_PATH, HIND_SHEET)
    fore_df, fore_cols = load_and_clean_sheet(EXCEL_PATH, FORE_SHEET)

    # Aggregate full limbs
    hind_agg = aggregate_full_limb(hind_df, hind_cols, limb_label="hind")
    fore_agg = aggregate_full_limb(fore_df, fore_cols, limb_label="fore")

    print("\n============================================================")
    print("AGGREGATED HINDLIMB TABLE")
    print("============================================================")
    print(hind_agg[["group3", "species", "limb_pcsa", "limb_fasc_len", "muscle_count"]])

    print("\n============================================================")
    print("AGGREGATED FORELIMB TABLE")
    print("============================================================")
    print(fore_agg[["group3", "species", "limb_pcsa", "limb_fasc_len", "muscle_count"]])

    # Match species present in both
    matched_df = build_matched_full_limb_table(hind_agg, fore_agg)

    if matched_df.empty:
        print("\nNo species were matched between hindlimb and forelimb sheets.")
        return

    print("\n============================================================")
    print("MATCHED FULL-LIMB TABLE")
    print("============================================================")
    print(
        matched_df[
            [
                "group3",
                "species",
                "limb_pcsa_hind",
                "limb_fasc_len_hind",
                "muscle_count_hind",
                "limb_pcsa_fore",
                "limb_fasc_len_fore",
                "muscle_count_fore",
            ]
        ]
    )

    # Add peak metrics
    summary_df = add_peak_metrics(matched_df)

    print("\n============================================================")
    print("FULL-LIMB PEAK SUMMARY")
    print("============================================================")
    print(
        summary_df[
            [
                "group3",
                "species",
                "hind_peak_K",
                "hind_peak_G",
                "fore_peak_K",
                "fore_peak_G",
                "delta_peak_K_fore_minus_hind",
                "delta_peak_G_fore_minus_hind",
            ]
        ]
    )

    # Save tables
    matched_df.to_csv("matched_full_limb_hind_fore_summary.csv", index=False)
    summary_df.to_csv("matched_full_limb_hind_fore_peak_summary.csv", index=False)

    print("\nSaved:")
    print("  matched_full_limb_hind_fore_summary.csv")
    print("  matched_full_limb_hind_fore_peak_summary.csv")

    # Group plots
    for grp in ["reptiles", "mammals", "bipeds"]:
        plot_group_full_limb_comparison(summary_df, grp)

    # # Species plots
    # if MAKE_SPECIES_PLOTS:
    #     plot_species_full_limb_comparison(summary_df)


if __name__ == "__main__":
    main()