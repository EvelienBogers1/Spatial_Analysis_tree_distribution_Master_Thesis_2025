# variance_boxplots.py

import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------
# Settings
# ----------------------------
# patch_area = 1024
# test_names = [f"LPJ-GUESS_alpha-test{i+1}" for i in range(7)]

suffix = "2"   # <-- change to "2" for the second run
results = [f"result{i}.{suffix}" for i in range(1, 5)]
areas   = [1024, 2048, 4096, 8192]


# ----------------------------
# Load data
# ----------------------------
aligned_outputs = []
ents = []
labels = []  # keep track of "Result + Area"

# for test_name in test_names:
#     aligned = np.load(f"{test_name}_aligned_output_{patch_area}.npy")
#     with open(f"{test_name}_ent_{patch_area}.pkl", "rb") as f:
#         ent = pickle.load(f)
#     aligned_outputs.append(aligned)
#     ents.append(ent)

# print(f"Loaded {len(aligned_outputs)} alpha tests.")

for result, area in zip(results, areas):
    aligned = np.load(f"LPJ-GUESS_{result}_aligned_output_{area}.npy")
    with open(f"LPJ-GUESS_{result}_ent_{area}.pkl", "rb") as f:
        ent = pickle.load(f)
    aligned_outputs.append(aligned)
    ents.append(ent)
    labels.append(f"{result}")

print(f"Loaded {len(aligned_outputs)} results.")

# ----------------------------
# Build dataframe with variances
# ----------------------------
data = []

# for test_idx, aligned_output in enumerate(aligned_outputs):
#     ent = ents[test_idx]
#     test_number = test_idx + 1   # just use 1–7
#     num_patches, num_species, _, bins = aligned_output.shape

#     for plot_idx in range(num_patches):
#         for sp_idx, species in enumerate(ent):
#             var_nd = np.nanvar(aligned_output[plot_idx, sp_idx, 0, :])
#             var_ripley = np.nanvar(aligned_output[plot_idx, sp_idx, 2, :])

#             data.append({
#                 "Test": test_number,
#                 "Species": species,
#                 "Function": "Ω(r)",
#                 "Variance": var_nd
#             })
#             data.append({
#                 "Test": test_number,
#                 "Species": species,
#                 "Function": "L(r)",
#                 "Variance": var_ripley
#             })

for res_idx, aligned_output in enumerate(aligned_outputs):
    ent = ents[res_idx]
    label = labels[res_idx]
    num_patches, num_species, _, bins = aligned_output.shape

    for plot_idx in range(num_patches):
        for sp_idx, species in enumerate(ent):
            var_nd = np.nanvar(aligned_output[plot_idx, sp_idx, 0, :])
            var_ripley = np.nanvar(aligned_output[plot_idx, sp_idx, 2, :])

            data.append({
                "Result": label,
                "Species": species,
                "Function": "Ω(r)",
                "Variance": var_nd
            })
            data.append({
                "Result": label,
                "Species": species,
                "Function": "L(r)",
                "Variance": var_ripley
            })


df = pd.DataFrame(data)
print(f"Dataframe built with {len(df)} rows.")

# ----------------------------
# Plotting
# ----------------------------
sns.set(style="whitegrid")

# Two clear contrasting colors
pal_list = sns.color_palette("tab10", 2)
palette = {"Ω(r)": pal_list[0], "L(r)": pal_list[1]}

# Facet by species in columns → side by side, but keep each narrower
g = sns.catplot(
    data=df, x="Result", y="Variance",
    hue="Function", col="Species",
    kind="box", dodge=True, sharey=False,
    height=4, aspect=1.0,
    palette=palette,
    legend=False  # disable default legend
)

# Add jittered points
for ax, species in zip(g.axes.flat, df["Species"].unique()):
    sns.stripplot(
        data=df[df["Species"] == species],
        x="Result", y="Variance", hue="Function",
        dodge=True, marker="o", alpha=0.45, linewidth=0.4,
        palette=palette, ax=ax, legend=False, size=3
    )
    ax.set_yscale("log")
    ax.set_xlabel("Result")
    ax.set_ylabel("Variance")

# # Adjust title and spacing
# g.fig.suptitle(f"Variance of Ω(r) and L(r) across 7 alpha tests (Patch area={patch_area})", fontsize=14, y=0.98)
# g.fig.subplots_adjust(top=0.85, right=0.83)  # leave space on right for legend

# # --- create a separate axis for legend ---
# legend_ax = g.fig.add_axes([0.87, 0.45, 0.03, 0.15])  # [left, bottom, width, height]
# legend_ax.axis("off")  # hide axis

# # Create legend handles
# handles = [
#     Patch(facecolor=palette["Ω(r)"], edgecolor="black", label="Ω(r)"),
#     Patch(facecolor=palette["L(r)"], edgecolor="black", label="L(r)")
# ]

# legend_ax.legend(handles=handles, title="Function", loc="center")

# # plt.savefig("variance_boxplots_by_species.png", dpi=300, bbox_inches="tight")
# plt.show()

g.fig.suptitle(f"Variance of Ω(r) and L(r) across 4 results", fontsize=14, y=0.98)
g.fig.subplots_adjust(top=0.85, right=0.83)

# Legend axis
legend_ax = g.fig.add_axes([0.87, 0.45, 0.03, 0.15])
legend_ax.axis("off")
handles = [
    Patch(facecolor=palette["Ω(r)"], edgecolor="black", label="Ω(r)"),
    Patch(facecolor=palette["L(r)"], edgecolor="black", label="L(r)")
]
legend_ax.legend(handles=handles, title="Function", loc="center")

plt.show()