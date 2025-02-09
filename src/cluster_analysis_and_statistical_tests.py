#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind

def main():
    # 1. Define the path to the Excel file
    file_path = "../databases/modularityBasedClustering.xlsx"
    
    # 2. Read the Excel file (without header) and assign column names
    try:
        # header=None indicates that the file has no header
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
    except Exception as e:
        print("Error reading the file:", e)
        return

    # Assign column names as provided:
    df.columns = [
        "id", "label", "interval", "color", "st_met", "st_teff", "st_mass", "size",
        "st_rad", "pl_orbeccen", "planet_name", "pl_dens", "pl_insol", "pl_orbper",
        "pl_orbsmax", "pl_eqt", "Modularity Class", "Clustering Coefficient",
        "Number of triangles", "Inferred Class"
    ]
    # Rename "Modularity Class" to "cluster"
    df.rename(columns={"Modularity Class": "cluster"}, inplace=True)

    # 3. Check if the necessary columns exist
    required_columns = ['cluster', 'st_mass', 'st_met']
    for col in required_columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in the dataset.")
            return

    # 4. Calculate descriptive statistics by cluster for 'st_mass' and 'st_met'
    cluster_stats = df.groupby('cluster').agg({
        'st_mass': ['mean', 'std'],
        'st_met': ['mean', 'std']
    }).reset_index()
    # Flatten the resulting multi-index and rename columns
    cluster_stats.columns = ['cluster', 'st_mass_mean', 'st_mass_std', 'st_met_mean', 'st_met_std']
    
    print("Statistics by Cluster:")
    print(cluster_stats)
    print("\n")

    # 5. ANOVA test to check differences between clusters for st_mass and st_met
    clusters = df['cluster'].unique()
    
    # Prepare data for each group for st_mass
    group_data_mass = [df[df['cluster'] == cl]['st_mass'] for cl in clusters]
    F_mass, p_mass = f_oneway(*group_data_mass)
    print(f"ANOVA for st_mass: F = {F_mass:.3f}, p = {p_mass:.3e}")
    
    # Prepare data for st_met
    group_data_met = [df[df['cluster'] == cl]['st_met'] for cl in clusters]
    F_met, p_met = f_oneway(*group_data_met)
    print(f"ANOVA for st_met: F = {F_met:.3f}, p = {p_met:.3e}")
    print("\n")

    # 6. Specific comparison with the "Earth-like" cluster
    # In this case, the Earth cluster is 40
    earth_cluster_id = 40  
    if earth_cluster_id not in clusters:
        print(f"The Earth-like cluster (id = {earth_cluster_id}) was not found in the data.")
    else:
        # Comparison for st_mass
        data_earth_mass = df[df['cluster'] == earth_cluster_id]['st_mass']
        data_other_mass = df[df['cluster'] != earth_cluster_id]['st_mass']
        t_stat_mass, p_val_mass = ttest_ind(data_earth_mass, data_other_mass, equal_var=False)
        print(f"t-test for st_mass (cluster {earth_cluster_id} vs. others): t = {t_stat_mass:.3f}, p = {p_val_mass:.3e}")

        # Comparison for st_met
        data_earth_met = df[df['cluster'] == earth_cluster_id]['st_met']
        data_other_met = df[df['cluster'] != earth_cluster_id]['st_met']
        t_stat_met, p_val_met = ttest_ind(data_earth_met, data_other_met, equal_var=False)
        print(f"t-test for st_met (cluster {earth_cluster_id} vs. others): t = {t_stat_met:.3f}, p = {p_val_met:.3e}")
    print("\n")

    # 7. Visualization: Boxplots for st_mass and st_met by cluster
    sns.set(style="whitegrid")

    # Plot for st_mass
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='st_mass', data=df)
    plt.title("Distribution of Stellar Mass by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Stellar Mass")
    plt.xticks(ticks=np.arange(0, max(clusters)+1, 5))
    plt.show()

    # Plot for st_met
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='st_met', data=df)
    plt.title("Distribution of Metallicity by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Metallicity")
    plt.xticks(ticks=np.arange(0, max(clusters)+1, 5))
    plt.show()

if __name__ == '__main__':
    main()
