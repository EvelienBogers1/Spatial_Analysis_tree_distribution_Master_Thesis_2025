import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import generate_sierpinski as generate_sierpinski_curve
import generate_moore as generate_moore_curve
import os
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
from matplotlib import cm
import matplotlib.colors as mcolors

def normalize_curve_length(curve):
    """
    Scales a 2D curve so that its total arc length becomes 1.

    Parameters:
        curve (np.ndarray): Array of shape (N, 2) representing the curve.

    Returns:
        np.ndarray: Normalized curve with arc length 1.
    """
    deltas = np.diff(curve, axis=0)
    segment_lengths = np.sqrt((deltas**2).sum(axis=1))
    total_length = segment_lengths.sum()

    # Scale curve so total length = 1
    scale_factor = 1 / total_length
    return curve * scale_factor

def FractalMethod(curve, curve_type, output_name, output_location, light_df, plot_size=None, number_of_points=None, df=None, multi=False, plot=False):
    """
    Projects 1D uniform points onto a 2D curve of unit length.

    Parameters:
        curve (np.ndarray): The curve coordinates (assumed to have arc length 1).
        curve_type (str): The curve type ('Sierpinski' or 'Moore').
        number_of_points (int): Number of points to project onto the curve. Defaults to None.
        df (pd.DataFrame): Optional DataFrame containing point data.  Defaults to None.
        multi (bool): If True, proccess multiple plots from the DataFrame. Defaults to False
        plot (bool): Whether to plot the projected points. Defaults to False.

    Returns:
        np.ndarray: 2D projected points.
    """
    if number_of_points is None and df is None:
        raise ValueError("Either number_of_points or df must be provided.")
    
    if curve_type not in ('Sierpinski', 'Moore'):
        raise ValueError("Unsupported curve type. Only 'Sierpinski' or 'Moore' is supported.")

    # Compute arc-length parameterization
    deltas = np.diff(curve, axis=0)
    segment_lengths = np.sqrt((deltas**2).sum(axis=1))
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    normalized_cumulative = cumulative_lengths / cumulative_lengths[-1]

    if isinstance(number_of_points, int):
        t_values = np.random.uniform(0, 1, size=number_of_points)
        t_values.sort()
        projected_points = []
        if plot_size is None:
            plot_size = 120

        for t in t_values:
            idx = np.searchsorted(normalized_cumulative, t) - 1
            idx = np.clip(idx, 0, len(deltas) - 1)
            local_t = (t - normalized_cumulative[idx]) / (normalized_cumulative[idx + 1] - normalized_cumulative[idx])
            point = curve[idx] + local_t * deltas[idx]
            projected_points.append(point)

        projected_points = np.array(projected_points)

        output_df = pd.DataFrame({
            'N': np.arange(1, number_of_points + 1),
            'x': projected_points[:, 0],
            'y': projected_points[:, 1],
            'species': 'projected',
            'entext': 'projected'
        })
        save_method_name = 'uniform_random'
        plot_name = 'uniform_random'

    if isinstance(df, pd.DataFrame):
        if multi:
            all_results = []
            for i in (df['plot_ID'].unique()):
                working_df = df[(df['plot_ID'] == i)]
                t_values = working_df['position'].values
                projected_points = []
                if len(t_values) == 0:
                    print(f"No position data for plot ID {i}. Skipping.")
                    continue

                for t in t_values:
                    idx = np.searchsorted(normalized_cumulative, t) - 1
                    idx = np.clip(idx, 0, len(deltas) - 1)
                    local_t = (t - normalized_cumulative[idx]) / (normalized_cumulative[idx + 1] - normalized_cumulative[idx])
                    point = curve[idx] + local_t * deltas[idx]
                    if len(point) == 0:
                        continue
                    else:
                        projected_points.append(point)

                projected_points = np.array(projected_points)

                output_df = pd.DataFrame({
                    'plot_ID': working_df['plot_ID'].values,
                    'N': working_df['N'].values,
                    'x': projected_points[:, 0],
                    'y': projected_points[:, 1],
                    'year': working_df['year'].values,
                    'species': working_df['species'].astype(str).values,
                    'entext': working_df['entext'].astype(str).values,
                    'cmass': working_df['cmass'].values,
                    'crown': working_df['crown'].values
                })

                all_results.append(output_df)

            output_df = pd.concat(all_results, ignore_index=True)

        else:    
            t_values = df['position'].values
            projected_points = []

            for t in t_values:
                idx = np.searchsorted(normalized_cumulative, t) - 1
                idx = np.clip(idx, 0, len(deltas) - 1)
                local_t = (t - normalized_cumulative[idx]) / (normalized_cumulative[idx + 1] - normalized_cumulative[idx])
                point = curve[idx] + local_t * deltas[idx]
                projected_points.append(point)

            projected_points = np.array(projected_points)

            print('df columns: ',df.columns)
            output_df = pd.DataFrame({
                'N': df['N'].values,
                'x': projected_points[:, 0],
                'y': projected_points[:, 1],
                'year': df['year'].values,
                'species': df['species'].astype(str).values,
                'entext': df['entext'].astype(str).values,
                'cmass': df['Cmass'].values,
                'crown': df['CrownA'].values
            })
        save_method_name = f'df_{int((plot_size)**2)}'
        plot_name = 'LPJ-GUESS DataFrame'

    save_path = output_name
    output_df.to_csv(save_path, index=False)
    print(f"Saved projected points to {save_path} {plot_name}")

    if plot==True:
        color_map = {
            'TeBS': 'orange',
            'IBS': 'green',
            'BNE': 'blue' 
            }
    

        # Compute cumulative distance along curve
        curve_deltas = np.diff(curve, axis=0)
        curve_seg_len = np.sqrt((curve_deltas**2).sum(axis=1))
        curve_cumlen = np.insert(np.cumsum(curve_seg_len), 0, 0)

        # Interpolate for smooth curve
        num_fine = 5000
        fine_s = np.linspace(0, curve_cumlen[-1], num_fine)
        interp_x = interp1d(curve_cumlen, curve[:,0], kind='linear')
        interp_y = interp1d(curve_cumlen, curve[:,1], kind='linear')
        fine_curve = np.column_stack([interp_x(fine_s), interp_y(fine_s)])



        for i in range(len(output_df['plot_ID'].unique())):
            id = output_df['plot_ID'].unique()[i]
            working_df = output_df[output_df['plot_ID'] == i]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(curve[:, 0], curve[:, 1], label='Normalized Curve', color='gray', linewidth=1, alpha=0.5)
            df_pid = output_df[output_df['plot_ID'] == id]
            print(output_df.columns)

            for sp, color in color_map.items():
                sp_points = df_pid[df_pid['entext'] == sp]
                print(f"{sp}: {len(sp_points)} points")
                if not sp_points.empty:
                    ax.scatter(
                        sp_points['x'], sp_points['y'],
                        color=color,
                        s=25,
                        label=sp,
                        zorder=3 # added for circles
                    )

                    # for _, row in sp_points.iterrows():
                    #     radius = np.sqrt(row['crown'] / np.pi)  # convert area to radius
                    #     circle = plt.Circle(
                    #         (row['x'], row['y']),
                    #         radius=radius,
                    #         color=color,
                    #         alpha=0.3,
                    #         zorder=2
                    #     )
                    #     ax.add_patch(circle)

                    # For each tree
                    for _, row in sp_points.iterrows():
                        scaled_crown = row['crown'] * area
                        radius = np.sqrt(scaled_crown / np.pi)

                        # Find nearest point on fine curve
                        dist_to_curve = np.sqrt((fine_curve[:,0]-row['x'])**2 + (fine_curve[:,1]-row['y'])**2)
                        center_idx = np.argmin(dist_to_curve)

                        # Find indices forward and backward along curve within radius
                        s_center = fine_s[center_idx]
                        mask = (fine_s >= s_center - radius) & (fine_s <= s_center + radius)

                        ax.plot(fine_curve[mask,0], fine_curve[mask,1], color=color, linewidth=9, alpha=0.2)


            ax.set_aspect('equal')
            ax.set_ylabel('Distance (m)')
            ax.set_xlabel('Distance (m)')
            if number_of_points is not None:
                ax.set_title(f'Projection onto {curve_type} curve, {plot_name}, {number_of_points} points on {int(plot_size**2)} m2' + (f', Plot ID {i+1}' if multi else ''))
            else:
                plot_label = plot_name + ' ' + output_name.replace('.csv', '').split('_')[-1]
                ax.set_title(
                    f'Projection onto {curve_type} curve, {plot_label}, {int(plot_size**2)} m2'
                    + (f', Plot ID {i+1}' if multi else ''))
            ax.legend(
                title="Species",
                bbox_to_anchor=(1.05, 0.8),
                loc='center left',
                borderaxespad=0.
            )
            ax.grid(True)

    # # -------- Left subplot: original projection --------
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    #         ax1.plot(curve[:, 0], curve[:, 1], label='Normalized Curve', color='gray', linewidth=1, alpha=0.5)
    #         df_pid = output_df[output_df['plot_ID'] == id]

    #         for sp, color in color_map.items():
    #             sp_points = df_pid[df_pid['entext'] == sp]
    #             if not sp_points.empty:
    #                 ax1.scatter(sp_points['x'], sp_points['y'], color=color, s=25, label=sp, zorder=3)

    #                 for _, row in sp_points.iterrows():
    #                     scaled_crown = row['crown'] * area
    #                     radius = np.sqrt(scaled_crown / np.pi)

    #                     dist_to_curve = np.sqrt((fine_curve[:,0]-row['x'])**2 + (fine_curve[:,1]-row['y'])**2)
    #                     center_idx = np.argmin(dist_to_curve)

    #                     s_center = fine_s[center_idx]
    #                     mask = (fine_s >= s_center - radius) & (fine_s <= s_center + radius)
    #                     ax1.plot(fine_curve[mask,0], fine_curve[mask,1], color=color, linewidth=9, alpha=0.2)

    #         ax1.set_aspect('equal')
    #         ax1.set_ylabel('Distance (m)')
    #         ax1.set_xlabel('Distance (m)')
    #         ax1.set_title(f'Projection onto {curve_type} curve, {plot_name}')
    #         ax1.legend(title="Species", bbox_to_anchor=(1.05, 0.8), loc='center left', borderaxespad=0.)
    #         ax1.grid(True)

    #         # -------- Right subplot: curve shaded by light --------
    #         if light_df is not None:
    #             light_values = np.array(light_df).flatten()
    #             n_segments = len(light_values)
    #             curve_len = fine_s[-1]
    #             segment_length = curve_len / n_segments

    #             cmap = cm.get_cmap('Greys')  # 0=dark, 1=light
    #             norm = mcolors.Normalize(vmin=0, vmax=1)

    #             for j in range(n_segments):
    #                 start_s = j * segment_length
    #                 end_s = (j + 1) * segment_length
    #                 mask = (fine_s >= start_s) & (fine_s <= end_s)
    #                 ax2.plot(
    #                     fine_curve[mask, 0],
    #                     fine_curve[mask, 1],
    #                     color=cmap(norm(light_values[j])),
    #                     linewidth=4
    #                 )

    #             ax2.set_aspect('equal')
    #             ax2.set_ylabel("Distance (m)")
    #             ax2.set_xlabel("Distance (m)")
    #             ax2.set_title("Curve Shaded by Light")
    #             ax2.grid(True)


            prefix = ''.join(filter(str.isdigit, results_name.split('alpha')[0]))
            suffix = results_name.split('_')[-1]

            # save instead of show
            save_file = os.path.join(output_location, f"{prefix}map{suffix}projection_plot_{id+1}.png")
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close(fig)  # close to avoid overlap in loop

            print(f"Saved plot for Plot ID {id+1} → {save_file}")

    return np.array(projected_points)

curve_type = 'Sierpinski' # Change to 'Sierpinski' or 'Moore' as needed
order = 4
#curve_path = f'{curve_type}_curve_{order}.csv'
#curve = np.loadtxt(curve_path, delimiter=',', skiprows=1)

# # THIS IS THE NAMING CONVENTION ETC THAT IS CHANGED EVERY TIME I RUN THIS FOR RESULTS
area = 8192
results_name = 'redo_result_4.2'
output_location = "C:\\Users\\Lienl\\OneDrive\\Documenten\\Uni\\Lund University\\Master\\Thesis project\\results+plots\\from LPJ-GUESS\\results\\unmod\\maps"

curve = generate_sierpinski_curve.generate_sierpinski_curve(order, area)
#curve = generate_sierpinski_curve.generate_moore_curve(order, area)

#curve = normalize_curve_length(curve)

number_of_points = 500
# df = pd.read_csv(f'LPJ_processed_output_{area}.csv') 
# df = pd.read_csv(f'LPJ_processed_output_0807.csv') 
df = pd.read_csv(f'LPJ_processed_output_{results_name}.csv')
light_df = 1
# light_df = pd.read_csv(f'{results_name}_ff_struct.out', delim_whitespace=True)
# light_df = light_df[light_df['Year'] == 2022]

# output_name = f'projected_points_{curve_type}_df_{area}.csv'
output_name = f'projected_points_{curve_type}_df_{results_name}.csv'

projected_points = FractalMethod(curve, curve_type, output_name, output_location, light_df, plot_size=np.sqrt(area), df=df, multi=True, plot=True)
#projected_points = FractalMethod(curve, curve_type, plot_size=None, number_of_points=number_of_points, multi=False, plot=True)