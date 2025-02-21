import numpy as np
import pandas as pd
import time
from interactive_graph_construction import construct_graph_interactive_earth_edges, construct_graph_planet_to_planet

# Importações para os novos algoritmos e comparação
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
import networkx as nx

def generate_lsh_hashes(data, num_hashes=10, seed=42):
    np.random.seed(seed)
    dimensions = data.shape[1]
    random_vectors = np.random.randn(num_hashes, dimensions)
    hashes = np.dot(data, random_vectors.T)
    return (hashes > 0).astype(int)

def main_with_earth_specific_graphs():
    # data_file = r"C:\Users\hecla\OneDrive\Desktop\CEFET\3rd period\AEDS\Graph Application\databases\PS_2025.01.12_10.15.56.csv"
    data_file = "../databases/PS_2025.01.12_10.15.56.csv"
    
    comparison_columns = [
        "st_teff", "st_rad", "st_mass", "pl_insol",
        "pl_eqt", "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_dens", "st_met"
    ]
    num_hashes = 20
    tolerance = 0.1  # (este parâmetro não está sendo usado diretamente no LSH)

    earth_reference = { 
        "st_teff": 5778,
        "st_rad": 1.0,
        "st_mass": 1.0,
        "pl_insol": 1.0,
        "pl_eqt": 288.0,
        "pl_orbper": 365.25,
        "pl_orbsmax": 1.0,
        "pl_orbeccen": 0.0167,
        "pl_dens": 5.51,
        "st_met": 0.0
    }
    similar_earth_reference = {
        "st_teff": 5778,
        "st_rad": 1.0,
        "st_mass": 1.0,
        "pl_insol": 1.02,
        "pl_eqt": 290.0,
        "pl_orbper": 365.0,
        "pl_orbsmax": 1.01,
        "pl_orbeccen": 0.02,
        "pl_dens": 5.5,
        "st_met": 0.01
    }

    # Load complete data to keep the planet_name column, if it exists
    df = pd.read_csv(data_file, comment='#')
    # If the planet_name column exists, include it in the data
    if "planet_name" in df.columns:
        columns_of_interest = comparison_columns + ["planet_name"]
    else:
        columns_of_interest = comparison_columns

    data = df[columns_of_interest].fillna(df[columns_of_interest].median())

    # Add Earth and "Similar Earth" with names
    earth_data = pd.DataFrame([earth_reference, similar_earth_reference], columns=comparison_columns)
    earth_data["planet_name"] = ["Earth", "Similar Earth"]
    data = pd.concat([data, earth_data], ignore_index=True)

    earth_index = len(data) - 2
    artificial_index = len(data) - 1

    # Aplicar log-transformação
    data_log = data.copy()
    for col in comparison_columns:
        data_log[col] = np.log(data_log[col] + 1)

    print("\n--- Gerando Hashes LSH ---")
    hashes = generate_lsh_hashes(data_log[comparison_columns], num_hashes=num_hashes)
    lsh_time = time.time() - start_lsh

    hashes_df = pd.DataFrame(hashes, columns=[f"hash_{i}" for i in range(num_hashes)])
    similarity_buckets = hashes_df.groupby(list(hashes_df.columns)).indices
    grouped_planets = {bucket: list(planets) for bucket, planets in similarity_buckets.items()}

    # Filtrar grupos que contenham a Terra
    groups_with_earth = [
        (bucket, planets) for bucket, planets in grouped_planets.items()
        if earth_index in planets
    ]

    # Gerar grafos apenas para esses grupos
    for bucket, planets in groups_with_earth:
        print(f"\nGerando grafo conectado à Terra para o bucket {bucket} com planetas {planets}")
        group_data = data.iloc[planets].copy()
        output_file = f"graph_group_with_earth_{bucket}.html"

        construct_graph_interactive_earth_edges(
            group_data,
            comparison_columns,
            earth_name="Earth",
            output_html=output_file
        )

        print(f"Gerando grafo de conexões entre planetas para o bucket {bucket}")
        output_file_planet_graph = f"planet_graph_group_{bucket}.html"  
        construct_graph_planet_to_planet(
            group_data,
            comparison_columns,
            output_html=output_file_planet_graph
        )

    print("\nGráficos gerados para todos os grupos contendo a Terra!")

if __name__ == "__main__":
    main_with_earth_specific_graphs()
