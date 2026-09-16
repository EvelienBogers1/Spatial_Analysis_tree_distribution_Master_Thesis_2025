import pandas as pd
import numpy as np


data_path = r"C:\Users\Lienl\OneDrive\Documenten\Uni\Lund University\Master\Thesis project\CP3_final_output_test.csv"
tree_data = pd.read_csv(data_path)
print('tree_data',len(tree_data))

# Filtering for Pheaktra's paper
filtered_plot = tree_data[
    ((tree_data['x'] >= -2) & (tree_data['x'] <= 6) & (tree_data['y'] >= -2) & (tree_data['y'] <= 6) & (tree_data['dbh'] >= 0.01) & (tree_data['dbh'] <= 0.05)) |
    ((tree_data['x'] >= -2) & (tree_data['x'] <= 11) & (tree_data['y'] >= -2) & (tree_data['y'] <= 11) & (tree_data['dbh'] > 0.05)) |
    ((tree_data['x'] >= -2) & (tree_data['x'] <= 16) & (tree_data['y'] >= -2) & (tree_data['y'] <= 32) & (tree_data['dbh'] > 0.15)) |
    ((tree_data['x'] >= -2) & (tree_data['x'] <= 32) & (tree_data['y'] >= -2) & (tree_data['y'] <= 53) & (tree_data['dbh'] > 0.30))
].copy()

# Reset tree id numbers
tree_data['TreeId'] = range(1, len(tree_data) + 1)
filtered_plot['TreeId'] = range(1, len(filtered_plot) + 1) 

print('filtered',len(filtered_plot))

# Save results
filtered_plot.to_csv("CP3_final_output_test_filtered_plot.csv", index=False)
print("Data frame saved")