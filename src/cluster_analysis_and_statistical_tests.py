#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
import os

# Configuração dos diretórios de saída
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
TXT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'txt')
CSV_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'csv')
HTML_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'html')

for directory in [TXT_OUTPUT_DIR, CSV_OUTPUT_DIR, HTML_OUTPUT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # 1. Define o caminho para o arquivo Excel
    file_path = r"C:\Users\hecla\OneDrive\Área de Trabalho\CEFET\4 periodo\AEDS II\TRABALHOS\Aplicação de Grafos\databases\modularityBasedClustering.xlsx"
    
    # 2. Lê o arquivo Excel (sem cabeçalho) e atribui os nomes das colunas
    try:
        # header=None indica que o arquivo não tem cabeçalho
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
    except Exception as e:
        print("Error reading the file:", e)
        return

    # Atribui os nomes das colunas conforme fornecido:
    df.columns = [
        "id", "label", "interval", "color", "st_met", "st_teff", "st_mass", "size",
        "st_rad", "pl_orbeccen", "planet_name", "pl_dens", "pl_insol", "pl_orbper",
        "pl_orbsmax", "pl_eqt", "Modularity Class", "Clustering Coefficient",
        "Number of triangles", "Inferred Class"
    ]
    # Renomeia "Modularity Class" para "cluster"
    df.rename(columns={"Modularity Class": "cluster"}, inplace=True)
    
    # Filtra para considerar apenas os clusters de 0 a 5
    df = df[df['cluster'].isin([0, 1, 2, 3, 4, 5])]

    # 3. Verifica se as colunas necessárias existem
    required_columns = ['cluster', 'st_mass', 'st_met']
    for col in required_columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in the dataset.")
            return

    # 4. Calcula estatísticas descritivas por cluster para 'st_mass' e 'st_met'
    cluster_stats = df.groupby('cluster').agg({
        'st_mass': ['mean', 'std'],
        'st_met': ['mean', 'std']
    }).reset_index()
    # "Achata" o multi-index e renomeia as colunas
    cluster_stats.columns = ['cluster', 'st_mass_mean', 'st_mass_std', 'st_met_mean', 'st_met_std']
    
    print("Statistics by Cluster:")
    print(cluster_stats)
    print("\n")

    # 5. Teste ANOVA para verificar diferenças entre clusters para st_mass e st_met
    clusters = sorted(df['cluster'].unique())  # Deverão ser somente 0, 1, 2, 3, 4, 5
    
    # Prepara os dados para cada grupo para st_mass
    group_data_mass = [df[df['cluster'] == cl]['st_mass'] for cl in clusters]
    F_mass, p_mass = f_oneway(*group_data_mass)
    print(f"ANOVA for st_mass: F = {F_mass:.3f}, p = {p_mass:.3e}")
    
    # Prepara os dados para st_met
    group_data_met = [df[df['cluster'] == cl]['st_met'] for cl in clusters]
    F_met, p_met = f_oneway(*group_data_met)
    print(f"ANOVA for st_met: F = {F_met:.3f}, p = {p_met:.3e}")
    print("\n")

    # 6. Comparação específica com o cluster "Earth-like"
    # Agora, considerando que os clusters vão de 0 a 5, definimos que o cluster Earth-like é o 5.
    earth_cluster_id = 5  
    if earth_cluster_id not in clusters:
        print(f"The Earth-like cluster (id = {earth_cluster_id}) was not found in the data.")
    else:
        # Comparação para st_mass
        data_earth_mass = df[df['cluster'] == earth_cluster_id]['st_mass']
        data_other_mass = df[df['cluster'] != earth_cluster_id]['st_mass']
        t_stat_mass, p_val_mass = ttest_ind(data_earth_mass, data_other_mass, equal_var=False)
        print(f"t-test for st_mass (cluster {earth_cluster_id} vs. others): t = {t_stat_mass:.3f}, p = {p_val_mass:.3e}")

        # Comparação para st_met
        data_earth_met = df[df['cluster'] == earth_cluster_id]['st_met']
        data_other_met = df[df['cluster'] != earth_cluster_id]['st_met']
        t_stat_met, p_val_met = ttest_ind(data_earth_met, data_other_met, equal_var=False)
        print(f"t-test for st_met (cluster {earth_cluster_id} vs. others): t = {t_stat_met:.3f}, p = {p_val_met:.3e}")
    print("\n")

    # 7. Visualização: Boxplots para st_mass e st_met por cluster
    sns.set(style="whitegrid")

    # Boxplot para st_mass
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='st_mass', data=df)
    plt.title("Distribution of Stellar Mass by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Stellar Mass")
    plt.xticks(ticks=range(0, 6))  # Clusters de 0 a 5
    plt.show()

    # Boxplot para st_met
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='st_met', data=df)
    plt.title("Distribution of Metallicity by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Metallicity")
    plt.xticks(ticks=range(0, 6))  # Clusters de 0 a 5
    plt.show()

if __name__ == '__main__':
    main()
