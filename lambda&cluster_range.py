import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
import alphashape
from scipy.spatial import distance_matrix
import pandas as pd

def calculate_area(coords, alpha=0.01):
    points = list(zip(coords['x'], coords['y']))
    polygon = alphashape.alphashape(points, alpha)
    return polygon.area

def calculate_cluster_range(data):
    specie_names = data['entext'].unique()
    print(f"Species names: {specie_names}")
    study_area = calculate_area(data[['x','y']])
    n = len(specie_names)
    print('NOTE: Any species with less than 500 points will be skipped.')
    results = []
    for j in range(n):
        df_specie = data[data['entext'] == specie_names[j]]
        number_points = df_specie['N'].count()
        if number_points < 500:
            continue
        intensity = number_points / study_area
        #print(f'Intensity (lambda) for species {specie_names[j]}: {intensity} points/m²')

        # Calculate nearest neighbour distance
        coords = df_specie[['x', 'y']].to_numpy()
        dist_matrix = distance_matrix(coords, coords)
        # Replace 0s (self-distances) with np.inf to avoid selecting them
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_distances = dist_matrix.min(axis=1)
        distance_nearest_neighbour = max(nearest_distances.mean(), 1e-6)  # Average nearest neighbor distance
        cluster_range = distance_nearest_neighbour / np.log(2)
        results.append({
            'species': specie_names[j],
            'intensity': intensity,
            'cluster_range': cluster_range
        })

        # print(f'Cluster range for {specie_names[j]}: {cluster_range} m')

    # # species for Suserup
    # species_mask = {
    #     # --- TeBS ---
    #     "Fraxinus excelsior": "TeBS",
    #     "Ulmus sp.": "TeBS",
    #     "Fagus sylvatica": "TeBS",
    #     "Quercus robur": "TeBS",
    #     "Quercus sp.": "TeBS",
    #     "Acer pseudoplatanus": "TeBS",
    #     "Acer platanoides": "TeBS",
    #     "Tilia sp.": "TeBS",
    #     "Carpinus betulus": "TeBS",
    #     "Aesculus hippocastanum": "TeBS",
    #     "Malus sylvestris": "TeBS",
    #     "Euonymus europaeus": "TeBS",
    #     "Corylus avellana": "TeBS",
    #     "Crataegus": "TeBS",
    #     "Prunus cerasifera": "TeBS",
    #     "Prunus avium": "TeBS",
    #     # --- IBS ---
    #     "Sambucus sp.": "IBS",
    #     "Salix sp.": "IBS",
    #     "Salix caprea": "IBS",
    #     "Salix cinerea": "IBS",
    #     "Betula sp.": "IBS",
    #     "Betula pubescens": "IBS",
    #     "Betula pendula": "IBS",
    #     "Alnus glutinosa": "IBS",
    #     "Sorbus aucuparia": "IBS",
    #     "Viburnum opulus": "IBS",
    #     "Cornus sp.": "IBS",
    #     # --- BINE ---
    #     "Larix x marschlinsii / L. x eurolepis": "BINE",  # larch = deciduous conifer, light-demanding
    #     # --- Other / unknown ---
    #     "Other decideous": "Other",
    #     "unknown": "Other",
    # }

#     # Dalby mask
#     species_mask = {
#     # --- TeBS ---
#     "Ulmus glabra": "TeBS",
#     "Fagus sylvatica": "TeBS",
#     "Quercus robur": "TeBS",
#     "Fraxinus excelsior": "TeBS",
#     # --- IBS ---
#     "Populus tremula": "IBS",
#     "Alnus glutinosa": "IBS",
#     # --- Other ---
#     "Salix caprea, ...": "Other",
# }

# # Biskopstorp mask
#     species_mask = {
#     # --- TeBS ---
#     "Fagus sylvatica": "TeBS",
#     "Quercus robur": "TeBS",
#     "Tilia": "TeBS",
#     # --- IBS ---
#     "Betula spp.": "IBS",
#     "Alnus glutinosa": "IBS",
#     "Sorbus aucuparia": "IBS",
#     "Frangula alnus": "IBS",   # alder buckthorn, light-demanding understory species
#     # --- BNE ---
#     "Abies alba": "BNE",       # Silver fir, shade-tolerant evergreen
#     "Picea": "BNE",            # generic spruce
#     "Picea albies": "BNE",     # (probably meant Picea abies = Norway spruce)
#     # --- BINE ---
#     "Pinus sylvestris": "BINE",  # Scots pine, light-demanding conifer
#     # --- Other ---
#     "Juniperus communis": "Other"  # Juniper, evergreen shrub/small tree, doesn’t fit neatly
#     }

    # UKR mask
    species_mask = {
    # --- TeBS ---
    "Fagus sylvatica": "TeBS",
    "Fraxinus excelsior": "TeBS",
    "Acer platanoides": "TeBS",
    "Acer pseudoplatanus": "TeBS",
    "Ulmus glabra": "TeBS",
    "Prunus avium": "TeBS",   # intermediate, but best placed in TeBS here
    # --- IBS ---
    "Sorbus aria": "IBS",     # whitebeam, light-demanding broadleaf
}

    df = pd.DataFrame(results)

    # Add category column using mapping
    df["category"] = df["species"].map(species_mask)

    # Filter valid ones
    filtered_results = df.dropna(subset=["category"])

    # --- Compute averages per category ---
    average_intensities = filtered_results.groupby("category")["intensity"].mean()

    average_TeBS = average_intensities.get("TeBS", float("nan"))
    average_IBS = average_intensities.get("IBS", float("nan"))
    average_BNE = average_intensities.get("BNE", float("nan"))
    average_BINE = average_intensities.get("BINE", float("nan"))

    # --- Compute averages per category ---
    average_intensities = filtered_results.groupby("category")["intensity"].mean()

    # --- Compute min and max per category ---
    min_intensities = filtered_results.groupby("category")["intensity"].min()
    max_intensities = filtered_results.groupby("category")["intensity"].max()

    # --- Print results ---
    print("Average intensities:")
    print(average_intensities)

    print("\nMinimum intensities:")
    print(min_intensities)

    print("\nMaximum intensities:")
    print(max_intensities)
    
    return results


# data = pd.read_csv('suserup_data_clean.csv')
# data = pd.read_csv('dalby_data_clean.csv')
# data = pd.read_csv('biskopstorp_data_clean.csv')
data = pd.read_csv('UKR_data_clean.csv')

results = calculate_cluster_range(data)