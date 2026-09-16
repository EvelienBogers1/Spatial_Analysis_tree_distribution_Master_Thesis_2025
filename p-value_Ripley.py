import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# =====================
# Helper functions
# =====================

def debug_T_distribution(T_obs, T_sims):
    print("DEBUG: T_obs =", T_obs)
    print("DEBUG: T_sims mean =", T_sims.mean(), "std =", T_sims.std(),
          "min =", T_sims.min(), "max =", T_sims.max())
    # empirical p without +1 small-sample correction
    p_raw = np.mean(np.abs(T_sims) >= np.abs(T_obs))
    print("DEBUG: raw two-sided p (no +1) =", p_raw)
    # print some quantiles
    for q in [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]:
        print(f"  quantile {q}: {np.quantile(T_sims, q):.6g}")

    # histogram
    plt.figure(figsize=(5,3))
    plt.hist(T_sims, bins=40)
    plt.axvline(T_obs, color='k', linestyle='--', label='T_obs')
    plt.axvline(T_sims.mean(), color='r', linestyle=':', label='mean(T_sims)')
    plt.legend()
    plt.title("T_sims distribution")
    plt.show()

def compute_T_from_L(K_obs, r_vals, r_min_frac=1e-6):
    # integrate |L(r)| or L(r) with a moderate weight; here simple trapz:
    L = np.sqrt(np.maximum(K_obs, 0) / np.pi) - r_vals
    # choose moderate weights (avoid 1/r extremes)
    r_min = max(1e-6, 0.01 * r_vals.max())
    weights = 1.0 / np.maximum(r_vals, r_min)
    T = np.trapz(weights * L, r_vals)
    return T

def compute_K(points, r_vals, area, edge_correction=True):
    n = points.shape[0]
    lam = n / area
    dmat = distance_matrix(points, points)
    np.fill_diagonal(dmat, np.inf)

    K_vals = []
    for r in r_vals:
        count = (dmat <= r).sum()
        K = count / (n * lam)
        K_vals.append(K)
    return np.array(K_vals)


def compute_T(K_obs, r_vals):
    # print(f"Computing T with K_obs shape: {K_obs.shape} and r_vals shape: {r_vals.shape}")
    K_csr = np.pi * r_vals**2
    diff = K_obs - K_csr
    weights = 1.0 / np.maximum(r_vals, 1e-6)  # avoid div by 0
    T = np.trapz(weights * diff, r_vals)
    # print(f"Computed T: {T}")
    return T


def simulate_CSR(n, area, r_vals, n_sims=199, bbox=(0,1,0,1)):
    T_sims = []
    for _ in range(n_sims):
        xs = np.random.uniform(bbox[0], bbox[1], n)
        ys = np.random.uniform(bbox[2], bbox[3], n)
        pts = np.column_stack((xs, ys))
        K_sim = compute_K(pts, r_vals, area)
        T_sim = compute_T_from_L(K_sim, r_vals)     #TESTING
        T_sims.append(T_sim)
    return np.array(T_sims)


# =====================
# Main analysis function
# =====================

def test_CSR(points, r_max=50, n_r=50, n_sims=199, bbox=None):
    xmin, xmax = points[:,0].min(), points[:,0].max()
    ymin, ymax = points[:,1].min(), points[:,1].max()
    area = (xmax - xmin) * (ymax - ymin)
    if bbox is None:
        bbox = (xmin, xmax, ymin, ymax)

    w = xmax - xmin
    h = ymax - ymin
    r_max = min(w, h) / 4.0

    # print("window:", w, "x", h, "r_max:", r_max)

    r_vals = np.linspace(1, r_max, n_r)

    K_obs = compute_K(points, r_vals, area)
    T_obs = compute_T_from_L(K_obs, r_vals)             #TESTING

    T_sims = simulate_CSR(points.shape[0], area, r_vals, n_sims=n_sims, bbox=bbox)

    # debug_T_distribution(T_obs, T_sims)

    # print(f"shape of T_sims: {T_sims.shape}, T_obs: {T_obs.shape}")
    # print(f"T_obs: {T_obs}, T_sims (first 5): {T_sims[:5]}")
    
    # # Centered version to get rid of bias
    mu = T_sims.mean()
    T_obs_c = T_obs - mu
    T_sims_c = T_sims - mu
    p_value = (np.sum(np.abs(T_sims_c) >= np.abs(T_obs_c)) + 1) / (n_sims + 1)

    # p_value = (np.sum(np.abs(T_sims) >= np.abs(T_obs)) + 1) / (n_sims + 1)

    return T_obs, p_value, r_vals, K_obs


def analyze_dataset(filename, r_max=50, n_r=50, n_sims=199):
    
    # =====================
    # Species filtering and remapping
    # =====================

    # species_mapping = [ 
    #     ("Fagus sylvatica", "TeBS"), 
    #     ("Ulmus sp.", "TeBS"), 
    #     ("Ulmus glabra", "TeBS"), 
    #     ("Fraxinus excelsior", "TeBS"), 
    #     ("Tilia", "TeBS"), 
    #     ("Quercus robur", "TeBS"),  
    #     ("Betula sp.", "IBS"), 
    #     ("Betula spp.", "IBS"), 
    #     ("Alnus glutinosa", "IBS"), 
    #     ("Populus tremula", "IBS"),  
    #     ("Picea abies", "BNE"), 
    # ]

    species_mapping = [
        ("TeBS", "TeBS"),
        ("IBS", "IBS"),
        ("BNE", "BNE"),
    ]

    valid_species = set([s for s, _ in species_mapping])

    replace_map = {
        "Betula pendula": "Betula sp.",
        "Betula pubescens": "Betula sp.",
        "Picea albies": "Picea abies",
        "Picea": "Picea abies",
    }

    
    df = pd.read_csv(filename)

    print(f"Species in df before filtering: {df['entext'].unique()}")

    df['entext'] = df['entext'].replace(replace_map)

    print(f"Species in df after replacing: {df['entext'].unique()}")

    # filter only valid species
    df = df[df["entext"].isin(valid_species)].copy()

    print(f"Species in df after species mapping: {df['entext'].unique()}")

    if df.empty:
        print(f"Results for {filename}: No valid species found after filtering.")
        return {}

    results = {}

    if df.shape[0] > 1:
        pts = df[["x", "y"]].values
        T_obs, p_value, r_vals, K_obs = test_CSR(pts, r_max, n_r, n_sims)
        results[("all", "all")] = {"T_obs": T_obs, "p_value": p_value}

    for year, g in df.groupby("year"):
        if g.shape[0] > 1:
            pts = g[["x", "y"]].values
            T_obs, p_value, _, _ = test_CSR(pts, r_max, n_r, n_sims)
            results[(year, "all")] = {"T_obs": T_obs, "p_value": p_value}

    for sp, g in df.groupby("entext"):
        if g.shape[0] > 1:
            pts = g[["x", "y"]].values
            T_obs, p_value, _, _ = test_CSR(pts, r_max, n_r, n_sims)
            results[("all", sp)] = {"T_obs": T_obs, "p_value": p_value}

    for (year, sp), g in df.groupby(["year", "entext"]):
        if g.shape[0] > 1:
            pts = g[["x", "y"]].values
            T_obs, p_value, _, _ = test_CSR(pts, r_max, n_r, n_sims)
            results[(year, sp)] = {"T_obs": T_obs, "p_value": p_value}

    print(f"Results for {filename}:")
    if not results:
        print("  No subsets with more than one point.")
    for k, v in results.items():
        year, sp = k
        # print(f"")
        print(f"  Year={year}, Species={sp} -> T={v['T_obs']:.4f}, p={v['p_value']:.4f}")

    return results


# =====================
# Run for all datasets
# =====================

# files = [
    # "suserup_data_clean.csv",
    # "biskopstorp_data_clean.csv",
    # "dalby_data_clean.csv",
    # "UKR_data_clean.csv",
# ]

files = [
    "projected_points_Sierpinski_df_redo_result_1.2.csv",
    "projected_points_Sierpinski_df_redo_result_2.2.csv",
    "projected_points_Sierpinski_df_redo_result_3.2.csv",
    "projected_points_Sierpinski_df_redo_result_4.2.csv",
    # "projected_points_Sierpinski_df_unmod1.1.csv",
    # "projected_points_Sierpinski_df_unmod1.2.csv",
    # "projected_points_Sierpinski_df_unmod2.1.csv",
    # "projected_points_Sierpinski_df_unmod2.2.csv",
    # "projected_points_Sierpinski_df_unmod3.1.csv",
    # "projected_points_Sierpinski_df_unmod3.2.csv",
    # "projected_points_Sierpinski_df_unmod1024.csv",
    # "projected_points_Sierpinski_df_unmod_redo_1.2.csv",
    # "projected_points_Sierpinski_df_unmod_redo_1024.csv",
    # "projected_points_Sierpinski_df_9216.csv",
    # "projected_points_Sierpinski_df_result1.1.csv",
    # "projected_points_Sierpinski_df_result1.2.csv",
    # "projected_points_Sierpinski_df_result2.1.csv",
    # "projected_points_Sierpinski_df_result2.2.csv",
    # "projected_points_Sierpinski_df_result3.1.csv",
    # "projected_points_Sierpinski_df_result3.2.csv",
    # "projected_points_Sierpinski_df_result4.1.csv",
    # "projected_points_Sierpinski_df_result4.2.csv",
]

all_results = {}
for f in files:
    all_results[f] = analyze_dataset(f)
