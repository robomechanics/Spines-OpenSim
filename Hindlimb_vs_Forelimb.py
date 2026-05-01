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
def compute_landscape(row, col_bodym: str, col_fasc_len: str) -> np.ndarray:
    """
    Figure-5 style landscape in gravity case:
      K_norm,max(G) = min(gamma1/G^2, 1 - kappa1/G)
    """
    body_mass = float(row[col_bodym])
    pcsa = float(row["PCSA (m2)"])
    Lm = float(row[col_fasc_len])

    Fmax = SIGMA_MAX * pcsa
    Wmax = Fmax * (EPS_MAX * Lm)
    vmax = Lm * EPSDOT_MAX

    Fe = external_force_gravity(body_mass)

    gamma1 = (body_mass * vmax**2) / (2.0 * Wmax)
    kappa1 = (Fe * (EPS_MAX * Lm)) / Wmax

    gamma = gamma1 / (G_vals**2)
    one_minus_kappa = 1.0 - (kappa1 / G_vals)

    K_norm_max = np.minimum(gamma, one_minus_kappa)

    # Hide infeasible region
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

    # Required columns
    col_species = find_col(df, "species")
    col_group = find_col(df, "group")
    col_muscle = find_col(df, "muscle")
    col_bodym = find_col(df, "body mass")
    col_muscle_mass = find_col(df, "muscle mass")
    col_pennation = find_col(df, "pennation")
    col_fasc_len = find_col(df, "fascicle length")

    # Numeric coercion
    for c in [col_bodym, col_muscle_mass, col_pennation, col_fasc_len]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing essentials
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

    # Recompute PCSA
    theta = np.deg2rad(df[col_pennation])
    df["PCSA (m2)"] = (df[col_muscle_mass] * np.cos(theta)) / (RHO * df[col_fasc_len])
    df["PCSA (m2)"] = pd.to_numeric(df["PCSA (m2)"], errors="coerce")
    df = df.dropna(subset=["PCSA (m2)"])
    df = df[df["PCSA (m2)"] > 0]
    print("After PCSA recompute:", df.shape)

    # Map groups
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
# PICK ONE REPRESENTATIVE MUSCLE PER SPECIES
# ============================================================
def get_one_muscle_per_species(
    df: pd.DataFrame,
    group_name: str,
    species_col: str,
    group_col: str,
    muscle_col: str,
    overrides: dict,
) -> pd.DataFrame:
    """
    For each species in the requested group, pick the first-listed muscle
    unless a species-specific override is provided.

    overrides example:
        {"Homo sapiens": "recfem_r"}
    """
    group_name = group_name.lower()

    out = (
        df[df[group_col] == group_name]
        .sort_index()
        .drop_duplicates(subset=[species_col], keep="first")
        .copy()
    )

    if overrides is not None:
        for sp, muscle in overrides.items():
            if sp not in set(out[species_col].values):
                continue

            cand = df[
                (df[group_col] == group_name)
                & (df[species_col] == sp)
                & (df[muscle_col].astype(str).str.strip().str.lower() == muscle.lower())
            ].sort_index()

            if not cand.empty:
                out = out[out[species_col] != sp]
                out = pd.concat([out, cand.iloc[[0]]], ignore_index=True)

    out = out.sort_values(by=species_col, kind="stable").reset_index(drop=True)
    return out


# ============================================================
# BUILD MATCHED HIND/FORE DATASET
# ============================================================
def build_matched_species_table(
    hind_df: pd.DataFrame,
    fore_df: pd.DataFrame,
    hind_cols: dict,
    fore_cols: dict,
    group_name: str,
    hind_overrides: dict,
    fore_overrides: dict,
) -> pd.DataFrame:
    hind_sel = get_one_muscle_per_species(
        hind_df,
        group_name=group_name,
        species_col=hind_cols["species"],
        group_col="group3",
        muscle_col=hind_cols["muscle"],
        overrides=hind_overrides,
    )

    fore_sel = get_one_muscle_per_species(
        fore_df,
        group_name=group_name,
        species_col=fore_cols["species"],
        group_col="group3",
        muscle_col=fore_cols["muscle"],
        overrides=fore_overrides,
    )

    common_species = sorted(
        set(hind_sel[hind_cols["species"]]).intersection(
            set(fore_sel[fore_cols["species"]])
        )
    )

    if not common_species:
        return pd.DataFrame()

    hind_match = hind_sel[hind_sel[hind_cols["species"]].isin(common_species)].copy()
    fore_match = fore_sel[fore_sel[fore_cols["species"]].isin(common_species)].copy()

    hind_match = hind_match.sort_values(by=hind_cols["species"]).reset_index(drop=True)
    fore_match = fore_match.sort_values(by=fore_cols["species"]).reset_index(drop=True)

    rows = []
    for sp in common_species:
        hrow = hind_match[hind_match[hind_cols["species"]] == sp].iloc[0]
        frow = fore_match[fore_match[fore_cols["species"]] == sp].iloc[0]

        rows.append(
            {
                "group3": group_name,
                "species": sp,
                "hind_muscle": hrow[hind_cols["muscle"]],
                "fore_muscle": frow[fore_cols["muscle"]],
                "hind_body_mass": hrow[hind_cols["bodym"]],
                "fore_body_mass": frow[fore_cols["bodym"]],
                "hind_fasc_len": hrow[hind_cols["fasc_len"]],
                "fore_fasc_len": frow[fore_cols["fasc_len"]],
                "hind_pcsa": hrow["PCSA (m2)"],
                "fore_pcsa": frow["PCSA (m2)"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_group_comparison(
    matched_df: pd.DataFrame,
    hind_df: pd.DataFrame,
    fore_df: pd.DataFrame,
    hind_cols: dict,
    fore_cols: dict,
    group_name: str,
):
    if matched_df.empty:
        print(f"No matched species for group: {group_name}")
        return

    plt.figure(figsize=(10, 6))

    for _, row in matched_df.iterrows():
        sp = row["species"]

        hrow = hind_df[
            (hind_df["group3"] == group_name)
            & (hind_df[hind_cols["species"]] == sp)
            & (hind_df[hind_cols["muscle"]] == row["hind_muscle"])
        ].iloc[0]

        frow = fore_df[
            (fore_df["group3"] == group_name)
            & (fore_df[fore_cols["species"]] == sp)
            & (fore_df[fore_cols["muscle"]] == row["fore_muscle"])
        ].iloc[0]

        Kh = compute_landscape(hrow, col_bodym=hind_cols["bodym"], col_fasc_len=hind_cols["fasc_len"])
        Kf = compute_landscape(frow, col_bodym=fore_cols["bodym"], col_fasc_len=fore_cols["fasc_len"])

        # Plot hind first, then fore with dashed line
        line = plt.plot(G_vals, Kh, linewidth=2, label=f"{sp} hind")[0]
        plt.plot(G_vals, Kf, linewidth=2, linestyle="--", color=line.get_color(), label=f"{sp} fore")

    plt.xscale("log")
    plt.xlabel("Mechanical advantage, G")
    plt.ylabel("Normalized kinetic energy capacity, $K_{norm,max}$")
    plt.title(f"{group_name.capitalize()} — Hindlimb vs Forelimb")
    plt.grid(True, which="major", linewidth=0.5)
    plt.grid(False, which="minor")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_species_comparison(
    matched_df: pd.DataFrame,
    hind_df: pd.DataFrame,
    fore_df: pd.DataFrame,
    hind_cols: dict,
    fore_cols: dict,
):
    if matched_df.empty:
        return

    for _, row in matched_df.iterrows():
        sp = row["species"]
        grp = row["group3"]

        hrow = hind_df[
            (hind_df["group3"] == grp)
            & (hind_df[hind_cols["species"]] == sp)
            & (hind_df[hind_cols["muscle"]] == row["hind_muscle"])
        ].iloc[0]

        frow = fore_df[
            (fore_df["group3"] == grp)
            & (fore_df[fore_cols["species"]] == sp)
            & (fore_df[fore_cols["muscle"]] == row["fore_muscle"])
        ].iloc[0]

        Kh = compute_landscape(hrow, col_bodym=hind_cols["bodym"], col_fasc_len=hind_cols["fasc_len"])
        Kf = compute_landscape(frow, col_bodym=fore_cols["bodym"], col_fasc_len=fore_cols["fasc_len"])

        plt.figure(figsize=(7, 5))
        plt.plot(G_vals, Kh, linewidth=2, label="Hindlimb")
        plt.plot(G_vals, Kf, linewidth=2, linestyle="--", label="Forelimb")

        plt.xscale("log")
        plt.xlabel("Mechanical advantage, G")
        plt.ylabel("Normalized kinetic energy capacity, $K_{norm,max}$")
        plt.title(f"{sp} — Hindlimb vs Forelimb")
        plt.grid(True, which="major", linewidth=0.5)
        plt.grid(False, which="minor")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    # Load both sheets
    hind_df, hind_cols = load_and_clean_sheet(EXCEL_PATH, HIND_SHEET)
    fore_df, fore_cols = load_and_clean_sheet(EXCEL_PATH, FORE_SHEET)

    # --------------------------------------------------------
    # Species-specific muscle overrides
    # Edit these if you want a specific muscle rather than the
    # first-listed muscle for a species.
    # --------------------------------------------------------
    hind_overrides = {
        "Homo sapiens": "recfem_r",
    }

    fore_overrides = {
        # Example:
        # "Homo sapiens": "tric_long"
    }

    all_matches = []

    for grp in ["reptiles", "mammals", "bipeds"]:
        matched_df = build_matched_species_table(
            hind_df=hind_df,
            fore_df=fore_df,
            hind_cols=hind_cols,
            fore_cols=fore_cols,
            group_name=grp,
            hind_overrides=hind_overrides,
            fore_overrides=fore_overrides,
        )

        if matched_df.empty:
            print(f"\nNo matched species found for {grp}")
            continue

        print(f"\nMatched species for {grp}:")
        print(matched_df[["species", "hind_muscle", "fore_muscle"]])

        all_matches.append(matched_df)

        # Group-level plot
        plot_group_comparison(
            matched_df=matched_df,
            hind_df=hind_df,
            fore_df=fore_df,
            hind_cols=hind_cols,
            fore_cols=fore_cols,
            group_name=grp,
        )

        # Species-by-species plots
        plot_species_comparison(
            matched_df=matched_df,
            hind_df=hind_df,
            fore_df=fore_df,
            hind_cols=hind_cols,
            fore_cols=fore_cols,
        )

    if all_matches:
        final_matches = pd.concat(all_matches, ignore_index=True)
        print("\n============================================================")
        print("FINAL MATCHED SPECIES TABLE")
        print("============================================================")
        print(final_matches)

        # Optional: save summary
        final_matches.to_csv("matched_hind_fore_species_summary.csv", index=False)
        print("\nSaved summary table to: matched_hind_fore_species_summary.csv")
    else:
        print("\nNo matched hindlimb/forelimb species were found.")


if __name__ == "__main__":
    main()