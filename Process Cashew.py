import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Load the data
path_to_data = "C:/Users/Lienl/OneDrive/Documenten/Uni/Lund University/Master/Thesis project/Cashew/results/CP1/FSCT/tree_data.csv"
data_raw = pd.read_csv(path_to_data)

# Filter with CCI column
data_filtered1 = data_raw[data_raw['CCI_at_BH'] >= 0.5]

plt.plot(data_raw['x_tree_base'],data_raw['y_tree_base'],'o',label='Raw data')
plt.title('Raw data tree locations')
plt.show()

plt.plot(data_filtered1['x_tree_base'],data_filtered1['y_tree_base'],'o',label='Filtered data 1')
plt.title('Filtered data tree locations')
plt.show()


# # Filtering method 3
# # Define the threshold distance (adjust as needed)
# data_filtered3 = data_filtered1.reset_index(drop=True)
# threshold = 0.5  # Example: 0.5 meters
# tree = KDTree(data_filtered3[['x_tree_base', 'y_tree_base']])
# pairs = tree.query_pairs(threshold)
# indices_to_remove = set()
# for i, j in pairs:
#     # Compare 'CCI_at_BH' and remove the one with the lower value
#     if data_filtered3.loc[i, 'CCI_at_BH'] < data_filtered3.loc[j, 'CCI_at_BH']:
#         removed_idx = i
#     else:
#         removed_idx = j
#     indices_to_remove.add(removed_idx)
#     print(f"Removing treeId {data_filtered3.loc[removed_idx, 'TreeId']} with CCI_at_BH {data_filtered3.loc[removed_idx, 'CCI_at_BH']}")
# # Keep only the points that are not in the removal list
# filtered_df = data_filtered3.drop(index=list(indices_to_remove))
# # Save the filtered data
# #filtered_df.to_csv("filtered_data.csv", index=False)
# print(f"Removed {len(indices_to_remove)} points that were too close.")
# print('Raw data shape: ',np.shape(data_raw))
# print('Filtered data1 shape: ',np.shape(data_filtered1))
# print('Filtered data2 shape: ',np.shape(data_filtered2))
# print('Filtered data3 shape: ',np.shape(filtered_df))

# # plt.plot(data_filtered2['x_tree_base'],data_filtered2['y_tree_base'],'o',label='Filtered data 2')
# # plt.title('Filtered data tree locations')
# # plt.show()

# # print(data_filtered.head())
# # print(data_filtered['TreeId'].unique())

# # print(data_filtered[data_filtered['TreeId']==348])
# # print(data_filtered[data_filtered['TreeId']==350])