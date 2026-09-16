import pandas as pd
import numpy as np
import argparse
from scipy.spatial import ConvexHull
from alphashape import alphashape

def compute_crown_metrics(veg_csv, use_concave_hull=False, alpha=1.5):
    print("Reading vegetation csv file...")
    chunks = pd.read_csv(veg_csv, delimiter=' ', comment='/', header=None, skip_blank_lines=True)
    
    # Fix header
    header = ['X', 'Y', 'Z', 'R', 'G', 'B', 'tree_id', 'height_above_dtm', 'label']
    chunks.columns = header
    print("Columns in the first chunk:", chunks.columns, len(chunks.columns))
    
    # Remove unclassified points
    veg_data = chunks[chunks['tree_id'] != 0]
    
    crown_metrics = {}
    for tree_id, group in veg_data.groupby('tree_id'):
        points = group[['X', 'Y', 'Z']].values
        
        if use_concave_hull:
            hull = alphashape(points, alpha)
        else:
            hull = ConvexHull(points)
        
        volume = hull.volume if hasattr(hull, 'volume') else hull.area  # Alpha shapes might not have volume
        area = hull.area
        
        crown_metrics[tree_id] = {'crown_volume': volume, 'crown_area': area}
    
    return crown_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("veg_csv", type=str, help="Path to vegetation points CSV file")
    parser.add_argument("tree_data", type=str, help="Path to tree data CSV file")
    parser.add_argument("output_file", type=str, help="Path to output CSV file")
    parser.add_argument("--concave", action='store_true', help="Use alpha shape instead of convex hull")
    parser.add_argument("--alpha", type=float, default=1.5, help="Alpha parameter for alpha shape (default: 1.5)")
    args = parser.parse_args()
    
    crown_metrics = compute_crown_metrics(args.veg_csv, use_concave_hull=args.concave, alpha=args.alpha)
    
    # Load tree data
    tree_data = pd.read_csv(args.tree_data)
    crown_df = pd.DataFrame.from_dict(crown_metrics, orient='index').reset_index().rename(columns={'index': 'tree_id'})
    
    # Merge with tree data
    tree_data = tree_data.merge(crown_df, left_on='TreeId', right_on='tree_id', how='left').drop(columns=['tree_id'])
    
    # Save results
    tree_data.to_csv(args.output_file, index=False)
    print("Crown metrics saved to", args.output_file)

if __name__ == "__main__":
    main()
