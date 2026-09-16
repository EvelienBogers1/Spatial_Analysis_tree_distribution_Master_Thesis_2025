import pandas as pd
import numpy as np
import argparse
from scipy.spatial import ConvexHull
from alphashape import alphashape

def compute_crown_metrics(veg_csv, use_concave_hull=False, alpha=1.5, chunk_size=100000):
    print("Reading vegetation csv file in chunks...")
    
    header = ['X', 'Y', 'Z', 'R', 'G', 'B', 'tree_id', 'height_above_dtm', 'label']
    tree_points = {}  # Dictionary to store points for each tree
    crown_metrics = {}
    last_tree_id = None
    remaining_data = pd.DataFrame(columns=header)
    
    for chunk in pd.read_csv(veg_csv, delimiter=' ', comment='/', header=None, names=header, skip_blank_lines=True, chunksize=chunk_size):
        chunk = pd.concat([remaining_data, chunk])  # Add remaining data from the previous chunk
        chunk = chunk[chunk['tree_id'] != 0]  # Remove unclassified points
        
        unique_trees = chunk['tree_id'].unique()
        
        # Identify trees to process and those to carry over
        process_trees = []
        for tree_id in unique_trees:
            if tree_id in tree_points or last_tree_id is None or tree_id != last_tree_id:
                process_trees.append(tree_id)
            tree_data = chunk[chunk['tree_id'] == tree_id]
            if tree_id not in tree_points:
                tree_points[tree_id] = []
            tree_points[tree_id].extend(tree_data[['X', 'Y', 'Z']].values.tolist())
        
        # Process trees that are complete
        for tree_id in process_trees[:-1]:  # Exclude the last tree to keep its points for the next chunk
            if tree_id in tree_points:
                points = np.array(tree_points.pop(tree_id))
                print(f"Processing tree ID: {tree_id}")
                
                if use_concave_hull:
                    hull = alphashape(points, alpha)
                else:
                    hull = ConvexHull(points)
                
                height = points[:, 2].max() - points[:, 2].min()
                volume = hull.volume if hasattr(hull, 'volume') else hull.area  # Alpha shapes might not have volume
                area = hull.area
                
                crown_metrics[tree_id] = {'H_c': height,'V_c': volume, 'A_c': area}
        
        last_tree_id = unique_trees[-1] if len(unique_trees) > 0 else last_tree_id
        remaining_data = chunk[chunk['tree_id'] == last_tree_id]  # Keep last tree's data for next chunk
    
    # Process the last remaining trees
    for tree_id, points in tree_points.items():
        points = np.array(points)
        print(f"Processing final tree ID: {tree_id}")
        
        if use_concave_hull:
            hull = alphashape(points, alpha)
        else:
            hull = ConvexHull(points)
        
        height = points[:, 2].max() - points[:, 2].min()
        volume = hull.volume if hasattr(hull, 'volume') else hull.area  # Alpha shapes might not have volume
        area = hull.area
        
        crown_metrics[tree_id] = {'H_c': height,'V_c': volume, 'A_c': area}
    
    return crown_metrics


def compute_crown_metrics_for_filtered_trees(veg_csv, filtered_tree_ids, use_concave_hull=False, alpha=1.5):
    print("Processing filtered trees with the concave method...")
    
    header = ['X', 'Y', 'Z', 'R', 'G', 'B', 'tree_id', 'height_above_dtm', 'label']
    tree_points = {}  # Dictionary to store points for each tree
    crown_metrics = {}
    
    # Read the veg_csv and filter for relevant tree_ids only
    chunk_size = 100000
    for chunk in pd.read_csv(veg_csv, delimiter=' ', comment='/', header=None, names=header, skip_blank_lines=True, chunksize=chunk_size):
        chunk = chunk[chunk['tree_id'].isin(filtered_tree_ids)]  # Filter relevant trees
        unique_trees = chunk['tree_id'].unique()
        
        for tree_id in unique_trees:
            if tree_id not in tree_points:
                tree_points[tree_id] = []
            tree_data = chunk[chunk['tree_id'] == tree_id]
            tree_points[tree_id].extend(tree_data[['X', 'Y', 'Z']].values.tolist())
    
    # Process filtered trees
    for tree_id, points in tree_points.items():
        points = np.array(points)
        print(f"Processing (concave) tree ID: {tree_id}")

        hull = alphashape(points, alpha)
        
        volume = hull.volume if hasattr(hull, 'volume') else hull.area  # Alpha shapes might not have volume
        area = hull.area
        
        crown_metrics[tree_id] = {'V_c_concave': volume, 'A_c_concave': area}
    
    return crown_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("veg_csv", type=str, help="Path to vegetation points CSV file")
    parser.add_argument("tree_data", type=str, help="Path to tree data CSV file")
    parser.add_argument("output_file", type=str, help="Path to output CSV file")
    parser.add_argument("--concave", action='store_true', help="Use alpha shape instead of convex hull")
    parser.add_argument("--alpha", type=float, default=1.5, help="Alpha parameter for alpha shape (default: 1.5)")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for reading CSV (default: 100000)")
    args = parser.parse_args()
    
    crown_metrics = compute_crown_metrics(args.veg_csv, use_concave_hull=args.concave, alpha=args.alpha, chunk_size=args.chunk_size)
    
    # Load tree data
    tree_data = pd.read_csv(args.tree_data)
    crown_df = pd.DataFrame.from_dict(crown_metrics, orient='index').reset_index().rename(columns={'index': 'tree_id'})
    
    # Merge with tree data
    tree_data = tree_data.merge(crown_df, left_on='TreeId', right_on='tree_id', how='left').drop(columns=['tree_id'])
    
    # Filter out trees with CCI_at_BH < 0.5
    if 'CCI_at_BH' in tree_data.columns:
        tree_data = tree_data[tree_data['CCI_at_BH'] >= 0.5]

    tree_data['V_b'] = tree_data['Volume_1'] - tree_data['Volume_2']
    tree_data = tree_data.rename(columns={'Volume_1': 'V_t'})
    tree_data = tree_data.rename(columns={'Volume_2': 'V_s'})
    tree_data = tree_data.rename(columns={'x_tree_base': 'x'})
    tree_data = tree_data.rename(columns={'y_tree_base': 'y'})
    tree_data = tree_data.rename(columns={'z_tree_base': 'z'})
    tree_data = tree_data.rename(columns={'Height': 'H'})
    tree_data = tree_data.rename(columns={'DBH': 'dbh'})
    tree_data['PlotId'] = tree_data['PlotId'].replace({'Kulen_CP1_2024-10-29_0.005m': 'CP1', 'Kulen_CP2_2024-10-29_0.005m': 'CP2', 'Kulen_CP3_2024-10-29_0.005m': 'CP3'})
    # tree_data['PlotId'] = os.path.basename(args.output_file).split('_')[0]

    tree_data = tree_data[['PlotId','TreeId','x','y','z','dbh','H','V_t','V_s','H_c','V_c','A_c','V_b','CCI_at_BH','Crown_mean_x','Crown_mean_y','Crown_top_x','Crown_top_y','Crown_top_z','mean_understory_height_in_5m_radius']]

    # # Filtering for Pheaktra's paper
    # tree_data = tree_data[(tree_data['x'] > -3) & (tree_data['x'] < 33)]
    # tree_data = tree_data[(tree_data['y'] > -3) & (tree_data['y'] < 53)]  
    # filtered_plot = tree_data[
    #     ((tree_data['x'] >= -2) & (tree_data['x'] <= 6) & (tree_data['y'] >= -2) & (tree_data['y'] <= 6) & (tree_data['dbh'] >= 0.01) & (tree_data['dbh'] <= 0.05)) |
    #     ((tree_data['x'] >= -2) & (tree_data['x'] <= 11) & (tree_data['y'] >= -2) & (tree_data['y'] <= 11) & (tree_data['dbh'] > 0.05)) |
    #     ((tree_data['x'] >= -2) & (tree_data['x'] <= 16) & (tree_data['y'] >= -2) & (tree_data['y'] <= 32) & (tree_data['dbh'] > 0.15)) |
    #     ((tree_data['x'] >= -2) & (tree_data['x'] <= 32) & (tree_data['y'] >= -2) & (tree_data['y'] <= 53) & (tree_data['dbh'] > 0.30))
    # ].copy()




    # Save the tree IDs to process after filtering
    tree_ids_to_process = tree_data['TreeId'].unique()

    # Now compute the crown metrics, applying concave hull only to the filtered tree IDs
    crown_metrics = compute_crown_metrics_for_filtered_trees(args.veg_csv, tree_ids_to_process, use_concave_hull=True, alpha=args.alpha)

    crown_df = pd.DataFrame.from_dict(crown_metrics, orient='index').reset_index().rename(columns={'index': 'tree_id'})
    tree_data = tree_data.merge(crown_df, left_on='TreeId', right_on='tree_id', how='left').drop(columns=['tree_id'])




    # Reset tree id numbers
    tree_data['TreeId'] = range(1, len(tree_data) + 1)
    # filtered_plot['TreeId'] = range(1, len(filtered_plot) + 1)  # if using filtered plot

    # Save results
    tree_data.to_csv(f"{args.output_file}.csv", index=False)
    # filtered_plot.to_csv(f"{args.output_file}_filtered_plot.csv", index=False)    # if using filtered plot
    print("Data frame saved to", args.output_file)

if __name__ == "__main__":
    main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# USAGE:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# run the script from command line with 3 arguments:
# 1) path to the vegetation points csv file (from FSCT, veg_points_sorted.csv)
# 2) path to the tree data csv file (from FSCT, tree_data.csv)
# 3) path to the output csv file (.csv will be created at given location, with given name)
# 
# If you wish to do any filtering of point locations, you probably want to do that at lines 141-149, 168, 172
# 
# Example use in command line:
# python Process_FSCT_output.py "./Lund University/Master/Thesis project/Cashew/results/CP3/FSCT/veg_points_sorted.csv" "./Lund University/Master/Thesis project/Cashew/results/CP3/FSCT/tree_data.csv" "./CP3_final_output.csv"