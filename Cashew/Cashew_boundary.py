import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_path = r"C:\Users\Lienl\OneDrive\Documenten\Uni\Lund University\Master\Thesis project\Cashew\Combine_all_data_13032025 - Copy.csv"
data = pd.read_csv(data_path)

CP1_data = data[data['Plot ID Fin'] == 'CP1']
CP2_data = data[data['Plot ID Fin'] == 'CP2']
CP3_data = data[data['Plot ID Fin'] == 'CP3']

print(len(CP1_data))
print(len(CP2_data))
print(len(CP3_data))
#28   17   22

# plt.plot(CP1_data['2025_X'], CP1_data['2025_Y'], 'o', markersize=5)
# plt.title('CP1_2025')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# plt.plot(CP2_data['2025_X'], CP2_data['2025_Y'], 'o',markersize=5)
# plt.title('CP2_2025')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# plt.plot(CP3_data['2025_X'], CP3_data['2025_Y'], 'o', markersize=5)
# plt.title('CP3_2025')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()




# # Load the shapefile using geopandas
# gdf = gpd.read_file(r"C:\Users\Lienl\OneDrive\Documenten\Uni\Lund University\Master\Thesis project\Cashew\Forest inventory plot\Forest inventory Plot.shp")

# # Extract the coordinates and labels (replace 'label_column' with the actual label column name)
# coordinates = gdf.geometry.apply(lambda point: (point.x, point.y))  # Extract x, y from geometry
# labels = gdf['Plot_ID']  # Replace 'label_column' with the actual name of the column that holds labels

# # Plot the shapefile
# fig, ax = plt.subplots(figsize=(10, 10))
# gdf.plot(ax=ax, edgecolor="black", facecolor="none")  # Boundary outline

# # Plot each point and add its label
# for label, (lon, lat) in zip(labels, coordinates):
#     ax.scatter(lon, lat, color='red', zorder=5)  # Plot the point in red
#     ax.text(lon + 0.001, lat + 0.001, label, color='black', fontsize=12, ha='left', va='bottom')  # Add the label

# # Add title and labels to axes
# plt.title("Forest Inventory Plot")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# #plt.show()

# selected_points = gdf[gdf['Plot_ID'].isin(['CH 1', 'CH 2', 'CH 3'])]

# # Debug: Check the selected points
# print("Selected points:")
# print(selected_points[['Plot_ID', 'X_WGS84UTM', 'Y_WGS84UTM']])
# print("CRS of the shapefile:", gdf.crs)

# gdf = gdf.to_crs(epsg=32633)
# selected_points['X_WGS84UTM'] = pd.to_numeric(selected_points['X_WGS84UTM'], errors='coerce')
# selected_points['Y_WGS84UTM'] = pd.to_numeric(selected_points['Y_WGS84UTM'], errors='coerce')
# distances = {}

# for i, plot_id_1 in enumerate(selected_points['Plot_ID']):
#     for j, plot_id_2 in enumerate(selected_points['Plot_ID']):
#         if i < j:  # To avoid calculating distance twice
#             point_1 = selected_points[selected_points['Plot_ID'] == plot_id_1]
#             point_2 = selected_points[selected_points['Plot_ID'] == plot_id_2]
            
#             # Get the UTM coordinates from the X_WGS84UTM and Y_WGS84UTM columns
#             x1, y1 = point_1['X_WGS84UTM'].values[0], point_1['Y_WGS84UTM'].values[0]
#             x2, y2 = point_2['X_WGS84UTM'].values[0], point_2['Y_WGS84UTM'].values[0]
            
#             # Calculate the Euclidean distance (in meters)
#             distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#             distances[f"{plot_id_1} to {plot_id_2}"] = distance

# # Print the distances
# if distances:
#     for pair, distance in distances.items():
#         print(f"Distance between {pair}: {distance:.2f} meters")
# else:
#     print("No distances calculated. Please check the selected points.")



# Current output:
output_data_CP1 = r"C:\Users\Lienl\OneDrive\Documenten\Uni\Lund University\Master\Thesis project\CP1_final_output_test_filtered_plot.csv"
# Load the data
data_CP1 = pd.read_csv(output_data_CP1)
plt.figure(figsize=(6,10))
plt.plot(data_CP1['x'], data_CP1['y'], 'o', markersize=5)
plt.title('CP1_FSCT_output_map_filtered_plot')
plt.xlabel('X')
plt.ylabel('Y')
for i,txt in enumerate(data_CP1['TreeId']):
    plt.annotate(txt, (data_CP1['x'].iloc[i],data_CP1['y'].iloc[i]), fontsize=5)
plt.show()

# # Current FSCT output:
# output_data_CP2 = r"C:\Users\Lienl\OneDrive\Documenten\Uni\Lund University\Master\Thesis project\CP2_final_output.csv"
# # Load the data
# data_CP2 = pd.read_csv(output_data_CP2)
# plt.figure(figsize=(6,10))
# plt.plot(data_CP2['x'], data_CP2['y'], 'o',markersize=5)
# plt.title('CP2_FSCT_output_map')
# plt.xlabel('X')
# plt.ylabel('Y')
# for i,txt in enumerate(data_CP2['TreeId']):
#     plt.annotate(txt, (data_CP2['x'].iloc[i],data_CP2['y'].iloc[i]), fontsize=5)
# plt.show()

# # Current output:
# output_data_CP3 = r"C:\Users\Lienl\OneDrive\Documenten\Uni\Lund University\Master\Thesis project\CP3_final_output.csv"
# # Load the data
# data_CP3 = pd.read_csv(output_data_CP3)
# plt.figure(figsize=(6,10))
# plt.plot(data_CP3['x'], data_CP3['y'], 'o',markersize=5)
# plt.title('CP3_FSCT_output_map')
# plt.xlabel('X')
# plt.ylabel('Y')
# for i,txt in enumerate(data_CP3['TreeId']):
#     plt.annotate(txt, (data_CP3['x'].iloc[i],data_CP3['y'].iloc[i]), fontsize=5)
# plt.show()