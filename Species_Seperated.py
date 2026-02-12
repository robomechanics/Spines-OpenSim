import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Settings
# ----------------------------
EXCEL_PATH = "/Users/cobiora/spines_research/bishop_et_al.__table_s1.xlsx"
SHEET = "Raw data - hindlimb"

RHO = 1060.0      # kg/m^3
SIGMA_MAX = 3.0e5 # Pa
EPS_MAX = 0.30
EPSDOT_MAX = 10.0 # 1/s
G_CONST = 9.81

G_vals = np.logspace(-3, 2.5, 1200) # 1e-3 .. 10

def external_force_gravity(body_mass_kg: float) -> float:
    return body_mass_kg * G_CONST

# ----------------------------
# Helpers
# ----------------------------
def find_col(df: pd.DataFrame, contains: str) -> str:
    hits = [c for c in df.columns if contains.lower() in str(c).lower()]
    if not hits:
        raise KeyError(f"Couldn't find a column containing '{contains}'. Columns:\n{list(df.columns)}")
    return hits[0]

def map_group_to_3(g: str) -> str | None:
    g = str(g).strip().lower()

    # bipeds: birds/avian/biped
    if ("biped" in g) or ("bird" in g) or ("avian" in g):
        return "bipeds"

    # mammals
    if "mammal" in g:
        return "mammals"

    # reptiles (broad + singular)
    if g == "reptile":
        return "reptiles"
    if ("reptile" in g) or ("croc" in g) or ("alligator" in g) or ("lizard" in g) or ("snake" in g) or ("turtle" in g) or ("lepidosaur" in g):
        return "reptiles"

    return None

def get_first_muscle_by_group(df: pd.DataFrame, group_name: str, species_col: str, group_col: str) -> pd.DataFrame:
    group_name = group_name.lower()
    out = (
        df[df[group_col] == group_name]
        .sort_index()  # preserve Excel order
        .drop_duplicates(subset=[species_col], keep="first")
        .copy()
    )
    return out

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

    gamma1 = (body_mass * vmax**2) / (2.0 * Wmax)         # 1/2mv^2/W at G=1
    kappa1 = (Fe * (EPS_MAX * Lm)) / Wmax                  # mg*delta / W at G=1

    gamma = gamma1 / (G_vals**2)
    one_minus_kappa = 1.0 - (kappa1 / G_vals)

    K_norm_max = np.minimum(gamma, one_minus_kappa)

    # hide infeasible region (negative capacity)
    K_norm_max = np.where(K_norm_max > 0, K_norm_max, np.nan)
    return K_norm_max

# ----------------------------
# Load + clean
# ----------------------------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
print("After read:", df.shape)

# Required columns
col_species = find_col(df, "species")
col_group   = find_col(df, "group")
col_muscle  = find_col(df, "muscle")
col_bodym   = find_col(df, "body mass")

# Columns needed to recompute PCSA from the Excel-style formula:
col_muscle_mass = find_col(df, "muscle mass")      # F in your Excel formula
col_pennation   = find_col(df, "pennation")        # H
col_fasc_len    = find_col(df, "fascicle length")  # G

# Numeric coercion
for c in [col_bodym, col_muscle_mass, col_pennation, col_fasc_len]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows missing essentials
df = df.dropna(subset=[col_species, col_group, col_muscle, col_bodym, col_muscle_mass, col_pennation, col_fasc_len])
df = df[(df[col_bodym] > 0) & (df[col_muscle_mass] > 0) & (df[col_fasc_len] > 0)]
print("After drop/filter:", df.shape)

mass_kg = df[col_muscle_mass]

theta = np.deg2rad(df[col_pennation])
df["PCSA (m2)"] = (mass_kg * np.cos(theta)) / (RHO * df[col_fasc_len])
df["PCSA (m2)"] = pd.to_numeric(df["PCSA (m2)"], errors="coerce")
df = df.dropna(subset=["PCSA (m2)"])
df = df[df["PCSA (m2)"] > 0]
print("After PCSA recompute:", df.shape)

# Map groups
df["group3"] = df[col_group].apply(map_group_to_3)
unmapped = sorted(df.loc[df["group3"].isna(), col_group].astype(str).str.strip().str.lower().unique())
if unmapped:
    print("WARNING: unmapped group labels:", unmapped)
df = df.dropna(subset=["group3"])
print("Groups present:", sorted(df["group3"].unique()))

# ----------------------------
# Plot: one figure per group, legend = species
# ----------------------------
for grp in ["reptiles", "mammals", "bipeds"]:
    gdf = get_first_muscle_by_group(df, grp, species_col=col_species, group_col="group3")

    if gdf.empty:
        print(f"No rows for group: {grp}")
        continue

    plt.figure(figsize=(8, 5))

    for _, r in gdf.iterrows():
        species = r[col_species]
        K = compute_landscape(r, col_bodym=col_bodym, col_fasc_len=col_fasc_len)
        plt.plot(G_vals, K, linewidth=1, alpha=0.9, label=species)

    plt.xscale("log")
    plt.xlabel("Mechanical advantage, G")
    plt.ylabel("Normalized kinetic energy capacity, $K_{norm,max}$")
    plt.title(f"{grp.capitalize()} — first-listed muscle per species")
    plt.grid(True, which="major", linewidth=0.5)
    plt.grid(False, which="minor")

    # Put legend outside so it doesn't cover curves
    plt.legend(title="Species", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)
    plt.tight_layout()
    plt.show()
