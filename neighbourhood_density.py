import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def data_processing(data,year):
    data = data[data['aarstal'] == year]
    data = data[['indiv_nr','x_utmetrs89','y_utmetrs89','treespecies']]
    return data

def area(coords):
    x = coords['x_utmetrs89'].values
    y = coords['y_utmetrs89'].values
    n = len(x)  # Number of vertices
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return area

def neighbourhood_density (data):
    total_tree = len(data['treespecies']) 
    total_area = area(data[['x_utmetrs89','y_utmetrs89']])
    x = np.linspace(0,790,80)
    mean_density = total_tree / total_area
    omega = np.zeros((total_tree, len(x)))
    omega_ripley = np.zeros((total_tree, len(x)))
    for i in range(total_tree):
        count = 0
        current_tree = data[['x_utmetrs89','y_utmetrs89']].iloc[i]
        dx = data['x_utmetrs89'] - current_tree['x_utmetrs89']
        dy = data['y_utmetrs89'] - current_tree['y_utmetrs89']
        dist = np.sqrt(dx**2 + dy**2)
        for j in range(len(x)):
            count = np.sum((dist >= x[j]) & (dist <= x[j]+10))
            count_ripley = np.sum(dist <= x[j]+10)
            D_x = count/(np.pi*(x[j]+10)**2-np.pi*x[j]**2)
            omega[i,j] = D_x/mean_density
            omega_ripley[i,j] = count_ripley/mean_density
    omega_mean = np.mean(omega, axis=0)
    omega_std = np.std(omega, axis=0)
    omega_se = 1.96 * omega_std / np.sqrt(total_tree)  # 95% confidence interval
    omega_mean_ripley = np.mean(omega_ripley, axis=0)
    omega_std_ripley = np.std(omega_ripley, axis=0)
    omega_se_ripley = 1.96 * omega_std_ripley / np.sqrt(total_tree)  # 95% confidence interval
    return omega_mean, omega_se, omega_mean_ripley, omega_std_ripley

def automate(data,years,treespecies):
    # Create a figure with 4x2 subplots (2 rows for each method)
    fig = plt.figure(figsize=(15, 20))  # Made figure taller for 4 rows
    gs = fig.add_gridspec(nrows=4, ncols=2, 
                         hspace=0.4,  # Vertical space between plots
                         wspace=0.3,  # Horizontal space between plots
                         top=0.95,    # Top margin for suptitle
                         bottom=0.05, # Bottom margin
                         left=0.1,    # Left margin
                         right=0.9)   # Right margin
    axes = gs.subplots()
    axes = axes.ravel()
    
    # Use simple counter for subplot index
    plot_idx = 0
    for year in years:
        print('Currently processing the year:', year)
        df = data_processing(data,year)
        df_bog = df[df['treespecies'] == treespecies[0]]
        df_elm = df[df['treespecies'] == treespecies[1]]
        df_ask = df[df['treespecies'] == treespecies[2]]
        nd_bog_mean, nd_bog_se, nd_bog_mean_ripley, nd_bog_se_ripley = neighbourhood_density(df_bog)
        nd_elm_mean, nd_elm_se, nd_elm_mean_ripley, nd_elm_se_ripley = neighbourhood_density(df_elm)
        nd_ask_mean, nd_ask_se, nd_ask_mean_ripley, nd_ask_se_ripley = neighbourhood_density(df_ask)
        x = np.linspace(10,800,80)
        
        # Plot D_x on top row
        axes[plot_idx].errorbar(x, nd_bog_mean, yerr=nd_bog_se, fmt='b-o', 
            linewidth=1.5, markersize=3, label=treespecies[0], capsize=2)
        axes[plot_idx].errorbar(x, nd_elm_mean, yerr=nd_elm_se, fmt='r-s', 
            linewidth=1.5, markersize=3, label=treespecies[1], capsize=2)
        axes[plot_idx].errorbar(x, nd_ask_mean, yerr=nd_ask_se, fmt='g-^', 
            linewidth=1.5, markersize=3, label=treespecies[2], capsize=2)
        axes[plot_idx].set_xlabel('Distance (meters)', fontsize=12)
        axes[plot_idx].set_ylabel('Ω(r)', fontsize=12)
        axes[plot_idx].set_title(f'D_x Function ({year})', fontsize=14, pad=15)
        axes[plot_idx].grid(True, linestyle='--', alpha=0.7)
        axes[plot_idx].legend(fontsize=10, loc='upper right')
        
        # Plot Ripley's K on bottom row (plot_idx + 4 to move to bottom half)
        axes[plot_idx + 4].errorbar(x, nd_bog_mean_ripley, yerr=nd_bog_se_ripley, fmt='b-o', 
            linewidth=1.5, markersize=3, label=treespecies[0], capsize=2)
        axes[plot_idx + 4].errorbar(x, nd_elm_mean_ripley, yerr=nd_elm_se_ripley, fmt='r-s', 
            linewidth=1.5, markersize=3, label=treespecies[1], capsize=2)
        axes[plot_idx + 4].errorbar(x, nd_ask_mean_ripley, yerr=nd_ask_se_ripley, fmt='g-^', 
            linewidth=1.5, markersize=3, label=treespecies[2], capsize=2)
        axes[plot_idx + 4].set_xlabel('Distance (meters)', fontsize=12)
        axes[plot_idx + 4].set_ylabel('K(r)', fontsize=12)
        axes[plot_idx + 4].set_title(f'Ripley K Function ({year})', fontsize=14, pad=15)
        axes[plot_idx + 4].grid(True, linestyle='--', alpha=0.7)
        axes[plot_idx + 4].legend(fontsize=10, loc='upper right')
        
        plot_idx += 1  # Increment counter for next subplot
    
    # Adjust title position and margins
    plt.suptitle('Neighborhood Density and Ripley K Functions Over Time', 
                fontsize=16, 
                y=0.98)
    plt.show()

path_to_data = 'trees_suserup.csv'
data = pd.read_csv(path_to_data)
years = [2023,2012,2002,1992]
species = ['BOG', 'ELM', 'ASK']
#species = ['BIR', 'BID', 'BIV']
automate(data,years, species)

#NOTES
#df_2023 = data_processing(data,2023)
#print(np.shape(data),np.shape(data_2023),np.shape(df_2023))
#print(df_2023['treespecies'].unique())
#print(df_2023['treespecies'].value_counts())
# treespecies
# BOG    4232
# ELM    3930
# ASK     930
# ER      851
# REL     715
# HAS     308
# HLD     238