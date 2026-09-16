import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import colorsys
import argparse
from collections import defaultdict
from shapely import area, normalize
from shapely.geometry import MultiPoint
import alphashape
import pickle

print("Script started")  

class PointPatternAnalysis:
    def __init__(self,data,n,plotname,map=False, bin_size=10, loglog=False, years=[], multi=False, area=None, normalize=False, plot_limit=None):
        self.data = data
        self.years = years
        self.n = n
        self.plotname = plotname
        self.map=map
        self.bin_size = bin_size
        self.loglog = loglog
        self.multi = multi
        self.area = area
        self.normalize = normalize
        self.plot_limit = plot_limit

    def data_processing(self,data,year=[]):
        if not isinstance(year, int):
            data = data[['N','x','y','species','entext']]
        else:
            data = data[data['year'] == year]
            data = data[['N','x','y','species','entext']]
        self.data = data
        return data

    def calculate_area(self, coords, alpha=0.01):
        points = list(zip(coords['x'], coords['y']))
        polygon = alphashape.alphashape(points, alpha)
        return polygon.area
    
    def weight_function(self, li, lj, study_area):
        """Compute weight correction for edge effects."""
        xmin, xmax, ymin, ymax = study_area
        x, y = li  # Center of the circle
        xj, yj = np.atleast_1d(lj[0]), np.atleast_1d(lj[1])  # Ensure these are arrays

        dij = np.sqrt((xj - x) ** 2 + (yj - y) ** 2)  # Distance to all trees
        r = np.atleast_1d(dij)  # Ensure r is an array

        w = np.ones_like(r)  # Default weight = 1 (fully inside)

        # Check for edges
        for boundary, distance_scalar in [
            (xmin, x - xmin),  # Left edge
            (xmax, xmax - x),  # Right edge
            (ymin, y - ymin),  # Bottom edge
            (ymax, ymax - y)   # Top edge
        ]:
            distance = np.full_like(r, distance_scalar)  # Convert scalar to array matching r
            mask = (distance < r)  # Trees affected by this boundary

            if np.any(mask):  # Only compute if at least one tree is near the edge
                clipped_ratio = np.clip(distance[mask] / r[mask], -1, 1)  # Prevent invalid values
                A_segment = (
                    r[mask] ** 2 * np.arccos(clipped_ratio) -
                    distance[mask] * np.sqrt(np.maximum(r[mask] ** 2 - distance[mask] ** 2, 0))
                )
                w[mask] *= (1 - (A_segment / (np.pi * r[mask] ** 2)))  # Adjust weight
        return w

    def neighbourhood_density(self,data,bin_size,area,x_lim):
        total_tree = len(data['species']) 
        total_area = area if area is not None else print("Area not provided")
        x = np.arange(0, x_lim + 2*bin_size, bin_size)
        mean_density = total_tree / total_area
        omega = np.zeros((total_tree, len(x)))
        omega_ripley = np.zeros((total_tree, len(x)))
        xmin, xmax = np.min(data['x']), np.max(data['x'])
        ymin, ymax = np.min(data['y']), np.max(data['y'])
        study_area = (xmin, xmax, ymin, ymax)
        for i in range(total_tree):
            count = 0
            current_tree = data[['x','y']].iloc[i]
            dx = data['x'] - current_tree['x']
            dy = data['y'] - current_tree['y']
            dist = np.sqrt(dx**2 + dy**2)
            w = self.weight_function((current_tree['x'], current_tree['y']), (data['x'].to_numpy(), data['y'].to_numpy()), study_area)
            for j in range(len(x)):
                count = np.sum((1 / w) *((dist >= x[j]) & (dist <= x[j]+bin_size)))
                count_ripley = (total_area/total_tree) * (np.sum((1 / w) * ((dist <= x[j] + bin_size)) )) #/ total_tree)))
                D_x = count/(np.pi*((x[j]+bin_size)**2-x[j]**2))
                omega[i,j] = D_x/mean_density
                omega_ripley[i,j] = np.sqrt((count_ripley/np.pi)) - x[j]
        omega_mean = np.mean(omega, axis=0)
        omega_std = np.std(omega, axis=0)
        omega_se = 1.96 * omega_std / np.sqrt(total_tree)  # 95% confidence interval
        omega_mean_ripley = np.mean(omega_ripley, axis=0)  
        omega_std_ripley = np.std(omega_ripley, axis=0)
        omega_se_ripley = 1.96 * omega_std_ripley / np.sqrt(total_tree)  # 95% confidence interval
        return omega_mean, omega_se, omega_mean_ripley, omega_se_ripley
    
    def automate(self,data,n,bin_size,area,x_lim,years=[],normalize=False):
        initial_n = n
        if len(years) == 0:
            output_size = int((x_lim / bin_size) + 3)
            output = np.zeros((n,4,output_size))    # use n+1 if uniform set is also wanted
            df = self.data_processing(data)
            if 'entext' not in df.columns:
                print("WARNING: 'entext' column missing from processed dataframe!")
                print(f"Available columns: {df.columns.tolist()}")
            specie_names = df['entext'].value_counts().nlargest(n).index.tolist()

            if len(specie_names) < n:
                print(f"Requested {n} species, but only found {len(specie_names)}. Adjusting n.")
                n = len(specie_names)

            entext = specie_names
            for j in range(n):
                df_specie = df[df['entext'] == specie_names[j]]
                if len(df_specie) == 0:
                    print(f"WARNING: No trees found for species {specie_names[j]}")
                    continue
                nd_mean, nd_se, nd_mean_ripley, nd_se_ripley = self.neighbourhood_density(df_specie,bin_size,area,x_lim)
                output[j, 0, :] = nd_mean        # Store mean
                output[j, 1, :] = nd_se          # Store standard error
                output[j, 2, :] = nd_mean_ripley # Store Ripley's K mean
                output[j, 3, :] = nd_se_ripley   # Store Ripley's K standard error
            # # Define parameters for uniform random set
            # num_trees = int(df['entext'].value_counts().nlargest(n).mean())
            # xmin, xmax = 0, x_lim  # Spatial extent
            # ymin, ymax = 0, x_lim  # Spatial extent
            # x_uniform = np.random.uniform(xmin, xmax, num_trees)
            # y_uniform = np.random.uniform(ymin, ymax, num_trees)
            # uniform = pd.DataFrame({
            #     'N': np.arange(1, num_trees + 1),  # Creates a sequence from 1 to num_trees
            #     'x': x_uniform,
            #     'y': y_uniform,
            #     'species': 'Uniform distribution',
            #     'entext': 'Uniform distribution'
            # })
            # nd_mean, nd_se, nd_mean_ripley, nd_se_ripley = self.neighbourhood_density(uniform,bin_size,area)
            # output[n, 0, :] = nd_mean        # Store mean
            # output[n, 1, :] = nd_se          # Store standard error
            # output[n, 2, :] = nd_mean_ripley # Store Ripley's K mean
            # output[n, 3, :] = nd_se_ripley   # Store Ripley's K standard error
            # entext.append('Uniform distribution')
        if len(years) > 0:

            output_size = int((x_lim / bin_size) + 3)
            # # output_size sometimes gives the bug that it should be either +2 or +3 depending on if shape of nd_mean matches or not

            output = np.zeros((len(years),n,4,output_size)) # use n+1 if uniform set is also wanted
            for i in range(len(years)):
                print('Currently processing the year:', years[i])
                df = self.data_processing(data,years[i])
                if 'entext' not in df.columns:
                    print("WARNING: 'entext' column missing from processed dataframe!")
                    print(f"Available columns: {df.columns.tolist()}")
                    continue
                specie_names = df['entext'].value_counts().nlargest(n).index.tolist()
                print(f"Species names in year {years[i]}: {specie_names}")
                n = initial_n
                if len(specie_names) < n:
                    print(f"Requested {n} species, but only found {len(specie_names)}. Adjusting n for this year.")
                    n = len(specie_names)

                entext = specie_names
                for j in range(n):
                    # print('check if number of trees for species', specie_names[j], 'in year', years[i], 'is less than 500')
                    # if df[df['entext'] == specie_names[j]].shape[0] < 500:
                    #     print(f"Less than 500 trees found for species {specie_names[j]} in year {years[i]}, skipping.")
                    #     continue

                    df_specie = df[df['entext'] == specie_names[j]]
                    if len(df_specie) == 0:
                        print(f"WARNING: No trees found for species {specie_names[j]} in year {years[i]}")
                        continue
                    nd_mean, nd_se, nd_mean_ripley, nd_se_ripley = self.neighbourhood_density(df_specie,bin_size,area,x_lim)

                    # Added this for normalization of pdf
                    # Assuming area is the total area of a square for simplicity
                    if normalize:
                        side_length = np.sqrt(area)
                        diagonal = np.sqrt(2) * side_length

                        def normalize_density(density, bin_size, diagonal):
                            """
                            Normalize density so the integral equals 1 between 0 and diagonal.
                            """
                            # Truncate beyond diagonal
                            cutoff_bins = int(np.floor(diagonal / bin_size))
                            density_cut = density[:cutoff_bins]

                            # Compute integral only for valid range
                            integral = np.sum(density_cut * bin_size)

                            if integral != 0:
                                density = density / integral  # normalize entire array
                            
                            return density

                        nd_mean = normalize_density(nd_mean, bin_size, diagonal)
                        nd_se = normalize_density(nd_se, bin_size, diagonal)
                        # nd_mean_ripley = normalize_density(nd_mean_ripley, bin_size, diagonal)
                        # nd_se_ripley = normalize_density(nd_se_ripley, bin_size, diagonal)

                        output[i, j, 0, :] = nd_mean
                        output[i, j, 1, :] = nd_se
                        output[i, j, 2, :] = nd_mean_ripley
                        output[i, j, 3, :] = nd_se_ripley

                    else:
                        output[i, j, 0, :] = nd_mean
                        output[i, j, 1, :] = nd_se
                        output[i, j, 2, :] = nd_mean_ripley
                        output[i, j, 3, :] = nd_se_ripley

                    x_max_L = np.argmax(nd_mean_ripley) * bin_size    

                    # print(f'number of {specie_names[j]} trees in year', years[i], ':', len(df_specie))
                    # print(f'Omega mean 5-10m: {nd_mean[1]} at distance 5-10m of species: {specie_names[j]} for year: {years[i]}')
                    # print(f'max L(x): {max(nd_mean_ripley)} at distance {x_max_L:.2f}m of species: {specie_names[j]} for year: {years[i]}')
                
                # # Debug
                #     print('i, j, nd_mean:', i, j, nd_mean)
                #     print('shape of output at line 173: ', output.shape)
                #     print('nd_mean: ', nd_mean)
                #     print(f'check at line 175 if nd_mean saved in output for year {years[i]}:', output[i, j, 0, :])
 
                # # # Define parameters for uniform random set
                # num_trees = int(df['entext'].value_counts().nlargest(n).mean())
                # xmin, xmax = 0, x_lim  # Spatial extent
                # ymin, ymax = 0, x_lim  # Spatial extent
                # x_uniform = np.random.uniform(xmin, xmax, num_trees)
                # y_uniform = np.random.uniform(ymin, ymax, num_trees)
                # uniform = pd.DataFrame({
                #     'N': np.arange(1, num_trees + 1),  # Creates a sequence from 1 to num_trees
                #     'x': x_uniform,
                #     'y': y_uniform,
                #     'species': 'Uniform distribution',
                #     'entext': 'Uniform distribution'
                # })
                # nd_mean, nd_se, nd_mean_ripley, nd_se_ripley = self.neighbourhood_density(uniform,bin_size,area)
                # output[i, n, 0, :] = nd_mean        # Store mean
                # output[i, n, 1, :] = nd_se          # Store standard error
                # output[i, n, 2, :] = nd_mean_ripley # Store Ripley's K mean
                # output[i, n, 3, :] = nd_se_ripley   # Store Ripley's K standard error
                # entext.append('Uniform distribution')
        # print('shape of output at line 200: ', output.shape)
        # print('check1 if nd_mean is still in output:', output[0, 0, 0, :])
        # print('check2 if nd_mean is still in output:', output[0][0][0])
        # print('difference between output[0][0] and output[0,0]:', output[0][0], output[0,0])
        return output, entext        

    def clean_plotting(self, data, entext, n, plotname, bin_size, x_lim, i=None, plot_id=None, years=[], loglog=False, multi=False):       
        def plot_with_error(ax, x, y, yerr, color, label, alpha=1.0, linewidth=1, markersize=2, capsize=2):
            ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, linewidth=linewidth,
                markersize=markersize, capsize=capsize, label=label, alpha=alpha)

        def format_axis(ax, xlabel, ylabel, title, loglog=False):
            if loglog:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=12, pad=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            # no legend here → we add ordered legend manually

        species_colors = [
            ("TeBS", "goldenrod"),
            ("Fagus sylvatica", "goldenrod"),
            ("Ulmus sp.", "darkred"),
            ("Ulmus glabra", "darkred"),
            ("Fraxinus excelsior", "darkorange"),
            ("Tilia", "chocolate"),
            ("Quercus robur", "red"),

            ("IBS", "green"),
            ("Betula sp.", "green"),
            ("Betula spp.", "green"),
            ("Alnus glutinosa", "yellowgreen"),
            ("Populus tremula", "mediumspringgreen"),

            ("BNE", "blue"),
            ("Picea", "blue"),
        ]
        color_map = dict(species_colors)

        x = np.arange(0, x_lim + 2 * bin_size, bin_size)

        print('plotname:', plotname)
        print('entext:', entext)
        print('entext type:', type(entext))

        multi_year = len(years) > 1
        num_years = len(years) if multi_year else 1

        if num_years == 4:
            title_height = 0.95
            hspace = 0.3
            plotheight = 25
            bottom = 0.05
        else:
            title_height = 0.9 + 0.05 * (num_years - 1)
            hspace = 0.2 + 0.05 * (num_years - 1)
            plotheight = 5 * num_years
            bottom = 0.1 - 0.02 * (num_years - 1)

        fig = plt.figure(figsize=(15, plotheight))
        gs = fig.add_gridspec(nrows=num_years, ncols=2, hspace=hspace, wspace=0.25,
                            top=title_height, bottom=bottom, left=0.08, right=0.95)
        axes = np.ravel(gs.subplots())

        for i in range(num_years):
            idx = i if multi_year else 0
            year_label = f" ({years[i]})" if years is not None else " (simulation)"

            if data.ndim == 4:
                if multi:
                    current_data = data
                else:
                    current_data = data[idx] if years is not None else data
            else:
                current_data = data

            current_entext = entext[idx] if (multi_year and isinstance(entext[0], (list, np.ndarray))) else entext
            labels = list(current_entext)
            plot_label = f", PlotID {plot_id}" if multi and plot_id is not None else ""

            # ---- Ω(r) plot ----
            ax_density = axes[i]
            num_species = min(len(labels), current_data.shape[0])

            for j in range(num_species):
                species = labels[j]
                color = color_map.get(species, "gray")
                desat_color = sns.desaturate(color, 0.35)

                if multi:
                    m_max = current_data.shape[0]
                    # for k in range(m_max - 1):
                    #     plot_with_error(
                    #         ax_density, x,
                    #         current_data[k, j, 0, :], current_data[k, j, 1, :],
                    #         desat_color, label="_nolegend_", alpha=0.4, linewidth=0.8
                    #     )
                    plot_with_error(
                        ax_density, x,
                        current_data[m_max-1, j, 0, :], current_data[m_max-1, j, 1, :],
                        color, f"{labels[j]}"
                    )
                else:
                    plot_with_error(
                        ax_density, x,
                        current_data[j, 0, :], current_data[j, 1, :],
                        color, labels[j]
                    )

            format_axis(ax_density, 'Distance (meters)', 'Ω(r)',
                        f'Neighbourhood density Function, {plotname}{year_label}{plot_label}', loglog)

            # ordered legend
            handles, labels_ = ax_density.get_legend_handles_labels()
            handle_dict = dict(zip(labels_, handles))
            ordered_handles, ordered_labels = [], []
            for sp, col in species_colors:
                if sp in handle_dict:
                    ordered_handles.append(handle_dict[sp])
                    ordered_labels.append(sp)
            ax_density.legend(ordered_handles, ordered_labels, fontsize=8, loc="upper right")

            # ---- L(r) plot ----
            ax_ripley = axes[i + num_years]
            num_species = min(len(labels), current_data.shape[0])

            for j in range(num_species):
                species = labels[j]
                color = color_map.get(species, "gray")
                desat_color = sns.desaturate(color, 0.35)

                if multi:
                    m_max = current_data.shape[0]
                    # for k in range(m_max - 1):
                    #     plot_with_error(
                    #         ax_ripley, x,
                    #         current_data[k, j, 2, :], current_data[k, j, 3, :],
                    #         desat_color, label="_nolegend_", alpha=0.4, linewidth=0.8
                    #     )
                    plot_with_error(
                        ax_ripley, x,
                        current_data[m_max-1, j, 2, :], current_data[m_max-1, j, 3, :],
                        color, f"{labels[j]}"
                    )
                else:
                    plot_with_error(
                        ax_ripley, x,
                        current_data[j, 2, :], current_data[j, 3, :],
                        color, labels[j]
                    )

            format_axis(ax_ripley, 'Distance (meters)', 'L(r)-r',
                        f'Test function L of Ripley K-Function, {plotname}{year_label}{plot_label}', loglog)

            # ordered legend
            handles, labels_ = ax_ripley.get_legend_handles_labels()
            handle_dict = dict(zip(labels_, handles))
            ordered_handles, ordered_labels = [], []
            for sp, col in species_colors:
                if sp in handle_dict:
                    ordered_handles.append(handle_dict[sp])
                    ordered_labels.append(sp)
            ax_ripley.legend(ordered_handles, ordered_labels, fontsize=8, loc="upper right")

        plt.suptitle(f'Neighborhood Density and test function for Ripley K-Function for {plotname}',
                    fontsize=14, y=1.075)

        print(f"Saving plots for {plotname} with {n} species, bin size {bin_size}, and {num_years} years.")
        plt.savefig(f'{plotname}PPA_all_years.png', dpi=300)   
        plt.show()


    def create_color_map(self, data):
        """Create a consistent color map for the species:
        - Fixed colors for known species
        - Distinct palette colors for any unknown species
        """
        # Your fixed species color mapping
        species_colors = [
            ("TeBS", "goldenrod"),
            ("Fagus sylvatica", "goldenrod"),
            ("Ulmus sp.", "darkred"),
            ("Ulmus glabra", "darkred"),
            ("Fraxinus excelsior", "darkorange"),
            ("Tilia", "chocolate"),
            ("Quercus robur", "red"),

            ("IBS", "green"),
            ("Betula sp.", "green"),
            ("Betula spp.", "green"),
            ("Alnus glutinosa", "yellowgreen"),
            ("Populus tremula", "mediumspringgreen"),

            ("BNE", "blue"),
            ("Picea", "blue"),
        ]
        fixed_color_map = dict(species_colors)

        # Extract species in dataset
        unique_species = data['entext'].unique()

        # Palette for fallback species
        colormap = "Set3"
        palette = sns.color_palette(colormap, len(unique_species))

        def is_yellowish(color):
            """Check if a color is yellowish based on hue & brightness."""
            r, g, b = color
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            return 40/360 < h < 70/360 and l > 0.6  # yellowish range

        # Alternative palette (non-yellow)
        alt_palette = sns.color_palette("tab10", len(unique_species))

        # Final species-color mapping
        species_color_map = {}

        # Assign fixed colors first
        for sp in unique_species:
            if sp in fixed_color_map:
                species_color_map[sp] = fixed_color_map[sp]

        # Assign fallback palette for species not in fixed map
        fallback_colors = []
        for i, sp in enumerate(unique_species):
            if sp not in species_color_map:
                color = palette[i % len(palette)]
                # Replace if too yellow
                if is_yellowish(color):
                    color = alt_palette[i % len(alt_palette)]
                species_color_map[sp] = color
                fallback_colors.append((sp, color))

        # Debug print (optional)
        print("Fixed mapping applied to:", [sp for sp in species_color_map if sp in fixed_color_map])
        print("Fallback mapping applied to:", fallback_colors)

        return species_color_map




    def mapping(self, data, plotname, species_colormap, year=[]):
        plt.figure(figsize=(9,6))
        sns.scatterplot(x='x', y='y', hue='entext', data=data, s=8, palette=species_colormap)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Wrap legend labels only (30 chars per line)
        wrap = lambda s: s.replace(', ', ',\n')
        legend = plt.legend(title='Species', bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
        for text in legend.get_texts():
            text.set_text(wrap(text.get_text()))

        title = f"Tree Distribution{' in ' + str(year) if isinstance(year, int) else ''} for {plotname}"
        plt.title(title)
        plt.tight_layout()
        plt.show()


    def plot_variance_boxplots(self, aligned_output, ent, test_name, patch_area):
        """
        Creates boxplots of variances across patches for each species and function.
        
        aligned_output: shape (num_plots, num_species, 4, bins)
        ent: list of species names
        test_name: str (e.g., 'alpha test 2')
        patch_area: int (e.g., 2048)
        """
        num_plots, num_species, _, bins = aligned_output.shape
        
        # Collect variances across patches
        data = []
        for plot_idx in range(num_plots):
            for sp_idx, species in enumerate(ent):
                # variance across bins for each patch
                var_nd = np.nanvar(aligned_output[plot_idx, sp_idx, 0, :])
                var_ripley = np.nanvar(aligned_output[plot_idx, sp_idx, 2, :])
                
                data.append({"Species": species, "Function": "Ω(r)", "Variance": var_nd})
                data.append({"Species": species, "Function": "L(r)", "Variance": var_ripley})

        # Convert to DataFrame for seaborn
        import pandas as pd
        df = pd.DataFrame(data)

        # --- Plot ---
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="Species", y="Variance", hue="Function")
        sns.stripplot(data=df, x="Species", y="Variance", hue="Function",
                    dodge=True, marker="o", alpha=0.6, linewidth=0.5, palette="dark:k")

        plt.title(f"Variance of Ω(r) and L(r)\n{test_name}, Patch area={patch_area}")
        plt.ylabel("Variance across bins")
        plt.xlabel("Species")
        plt.yscale("log")  # optional if IBS blows up like 166.667
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.show()


    def main_workflow(self, data, n, plotname, years=[], map=False, bin_size=10, loglog=False, multi=False, area=None, normalize=False, plot_limit=None):
        print('area', area)
        if area is None:
            area = self.calculate_area(data[['x', 'y']])
            print(f"Calculated area: {area} m²")
        
        data['species'] = data['species'].astype(str)
        data['entext'] = data['entext'].astype(str)

        replace_map = {
            "0": "BNE",
            "1": "TeBS",
            "2": "IBS"
        }
        data['entext'] = data['entext'].replace(replace_map)

        # # Filter data for specific species (Suserup)
        # print("Species in data:")
        # print(data['entext'].value_counts())
        # replace_map = {
        #     "Betula pendula": "Betula sp.",
        #     "Betula pubescens": "Betula sp."
        # }
        # data['entext'] = data['entext'].replace(replace_map)
        # data['x'] = data['x'] - data['x'].min()
        # data['y'] = data['y'] - data['y'].min()
        # print("Species in data:")
        # print(data['entext'].value_counts())
        # # species_names = [ 'Fraxinus excelsior','Ulmus sp.', 'Fagus sylvatica', 'Quercus robur', 'Acer pseudoplatanus', 'Tilia sp.'] # TeBS
        # # species_names = ['Salix sp.', 'Prunus avium', 'Alnus glutinosa', 'Prunus cerasifera' ] # IBS
        # # species_names = [ 'Fraxinus excelsior','Ulmus sp.', 'Fagus sylvatica', 'Quercus robur', 'Acer pseudoplatanus', 'Tilia sp.', 'Salix sp.', 'Prunus avium', 'Alnus glutinosa', 'Prunus cerasifera'] # All
        # species_names = [ 'Betula sp.', 'Fagus sylvatica', 'Ulmus sp.', 'Fraxinus excelsior', 'Alnus glutinosa']
        # data = data[data['entext'].isin(species_names)]

        # # Filter data for specific species (Dalby)
        # print("Species in data:")
        # print(data['entext'].value_counts())
        # species_names = ['Fraxinus excelsior', 'Fagus sylvatica', 'Quercus robur', 'Ulmus glabra' ] # TeBS
        # species_names = ['Alnus glutinosa', 'Populus tremula' ] # IBS
        # species_names = ['Fraxinus excelsior', 'Fagus sylvatica', 'Quercus robur', 'Ulmus glabra', 'Alnus glutinosa', 'Populus tremula' ] # All
        # species_names = ['Alnus glutinosa', 'Fagus sylvatica', 'Ulmus glabra', 'Fraxinus excelsior', 'Populus tremula']
        # data = data[data['entext'].isin(species_names)]

        # Filter data for specific species (Biskopstorp)
        # print("Species in data:")
        # print(data['entext'].value_counts())
        # replace_map = {
            # "Picea albies": "Picea",
        # }
        # data['entext'] = data['entext'].replace(replace_map)
        # species_names = ['Fagus sylvatica', 'Quercus robur', 'Tilia'] #TeBS
        # species_names = ['Betula spp.'] # IBS
        # species_names = ['Picea', 'Picea albies'] # BNE
        # species_names = ['Pinus sylvestris'] # BINE
        # species_names = ['Fagus sylvatica', 'Quercus robur', 'Tilia', 'Betula spp.', 'Picea', 'Picea albies', 'Pinus sylvestris'] #All
        # species_names = ['Fagus sylvatica', 'Betula spp.', 'Tilia', 'Quercus robur', 'Picea'] # for paper
        # data = data[data['entext'].isin(species_names)]

        # # Filter data for specific species (UKR)
        # print('Species in data:', data['entext'].unique())
        # species_names = ['Fraxinus excelsior', 'Fagus sylvatica', 'Acer pseudoplatanus', 'Ulmus glabra', 'Acer platanoides'] # TeBS
        # species_names = ['Fraxinus excelsior', 'Fagus sylvatica', 'Ulmus glabra']
        # data = data[data['entext'].isin(species_names)]

        output = []
        entext = []

        if multi:
            unique_plots = data['plot_ID'].unique()
        else:
            unique_plots = [None]  # single iteration placeholder

        for i, plot_id in enumerate(unique_plots):
            if multi:
                current_data = data[data['plot_ID'] == plot_id]
                print(f"Processing plot {i + 1}/{len(unique_plots)}")
            else:
                current_data = data
            # Step 1: Automated neighbourhood density calculations
            if plot_limit is None:
                x_lim = int(0.5 * np.sqrt(2 * area))
            else:
                x_lim = plot_limit
            print(f"Using x_lim = {x_lim} meters")
            print("plot_limit:", plot_limit)
            out, ent = self.automate(current_data, n, bin_size, area, x_lim, years, normalize=normalize)
            if multi:
                output.append(out)
                entext.append(ent)
            else:
                output = out
                entext = ent

        if multi:
            output = np.array(output)  # (num_plots, 1, max_species, 4, bins)
            print('multi output shape before squeeze:', output.shape)
            output = np.squeeze(output, axis=1)  # -> (num_plots, max_species, 4, bins)

            num_plots, max_species, _, bins = output.shape

            # Step 1: Collect all unique species
            all_species = sorted(set(species for sublist in entext for species in sublist))
            species_index = {s: i for i, s in enumerate(all_species)}
            num_species = len(all_species)

            # Step 2: Create empty aligned array: (num_plots, num_species, 4, bins)
            aligned_output = np.full((num_plots, num_species, 4, bins), np.nan)

            for plot_idx in range(num_plots):
                plot_species_list = entext[plot_idx]  # list of species in this plot

                for local_idx, species_name in enumerate(plot_species_list):
                    global_idx = species_index[species_name]

                    # Insert data into aligned position
                    aligned_output[plot_idx, global_idx] = output[plot_idx, local_idx]

            # Step 3: Average across plots, ignoring NaNs for species not in all plots
            averaged_output = np.nanmean(aligned_output, axis=0)  # (num_species, 4, bins)

            # Step 4: Remove species with all-NaN data
            valid_species_mask = ~np.isnan(averaged_output).all(axis=(1, 2))
            averaged_output = averaged_output[valid_species_mask]
            ent = [s for i, s in enumerate(all_species) if valid_species_mask[i]]

            # Step 5: Plot
            if averaged_output.ndim < 4:
                averaged_output = np.expand_dims(averaged_output, axis=0)
            aligned_output = np.concatenate([aligned_output, averaged_output], axis=0)


            test_name = plotname
            patch_area = area
            np.save(f"{test_name}_aligned_output_{patch_area}.npy", aligned_output)
            with open(f"{test_name}_ent_{patch_area}.pkl", "wb") as f:
                pickle.dump(ent, f)


            self.clean_plotting(aligned_output, ent, n, plotname, bin_size, x_lim,
                                years=years, loglog=loglog, multi=multi, plot_id=None)


        else:

            # # Load output_test_20250805.pkl and entext_test_20250805.pkl      #DEBUG
            # with open('output_test_20250805.pkl', 'rb') as f:
            #     output = pickle.load(f)
            # with open('entext_test_20250805.pkl', 'rb') as f:
            #     entext = pickle.load(f)

            # print('shape of output:', output.shape)
            # print('output[0]', output[0])
            # print('output[0][0]', output[0][0])

            self.clean_plotting(output, entext, n, plotname, bin_size, x_lim,
                                years=years, loglog=loglog, multi=multi, plot_id=None)

            # Step 3: Mapping (Optional)
            if map:
                print("Creating spatial distribution maps...")
                species_colormap = self.create_color_map(current_data)

                if len(years) == 0:
                    self.mapping(current_data, plotname, species_colormap)
                else:
                    for year in years:
                        year_data = self.data_processing(current_data, year)
                        self.mapping(year_data, plotname, species_colormap, year)

        return output, entext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Pattern Analysis Tool")
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--n', type=int, required=True, help='Top n species to analyze')
    parser.add_argument('--plotname', type=str, required=True, help='Name for the plot (e.g., forest name)')
    parser.add_argument('--bin_size', type=int, default=10, help='Bin size for distance intervals (default: 10)')
    parser.add_argument('--years', type=int, nargs='*', default=[], help='List of years (space-separated)')
    parser.add_argument('--loglog', type=str, default=False, help='Option to use log-log scale in plots (default= False)')
    parser.add_argument('--map', type=lambda x: x.lower() == 'true', default=False, help='Create spatial distribution maps if True')
    parser.add_argument('--multi', type=lambda x: x.lower() == 'true', default=False, help='Process multiple plots if True')
    parser.add_argument('--area', type=int, default=None, help='Area to process (optional) (default: None)')
    parser.add_argument('--normalize', type=lambda x: x.lower() == 'true', default=False, help='Normalize data if True')
    parser.add_argument('--plot_limit', type=int, default=None, help='Limit of distance for neighbourhood density and Ripley (optional) (default: None)')

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data_path)

    # Initialise class
    analysis = PointPatternAnalysis(
        years=args.years,
        data=data,
        n=args.n,
        plotname=args.plotname,
        map=args.map,
        bin_size=args.bin_size,
        loglog=args.loglog,
        multi=args.multi,
        area=args.area,
        normalize=args.normalize,
        plot_limit=args.plot_limit
    )

    output, entext = analysis.main_workflow(
        years=args.years,
        data=data,
        n=args.n,
        plotname=args.plotname,
        map=args.map,
        bin_size=args.bin_size,
        loglog=args.loglog,
        multi=args.multi,
        area=args.area,
        normalize=args.normalize,
        plot_limit=args.plot_limit
    )

    print("Analysis complete.")
# Example usage: py PointPatternAnalysis.py --data_path suserup_data_clean.csv --years 2023 2012 2002 1992 --n 3 --plotname Suserup --map True --normalize False --bin_size 10 --plot_limit 120
# py PointPatternAnalysis.py --data_path projected_points_Sierpinski.csv  --n 1 --plotname Projected --map False
# py PointPatternAnalysis.py --data_path projected_points_Moore.csv  --n 1 --plotname Projected --map False

# py PointPatternAnalysis.py --data_path projected_points_Sierpinski_df_1000.csv --years 2022 --n 3 --plotname LPJ-GUESS --map False --multi True
# py PointPatternAnalysis.py --data_path projected_points_Sierpinski_df_test.csv --years 2022 --n 3 --plotname LPJ-GUESS_test --map False --multi True --normalize True