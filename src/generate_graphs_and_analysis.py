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
    # data_file = r"C:\Users\hecla\OneDrive\Área de Trabalho\CEFET\3 periodo\AEDS\Trabalhos_AEDS\Aplicação de Grafos\databases\PS_2025.01.12_10.15.56.csv"
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

    # Carregar dados completos para manter coluna planet_name, se existir
    df = pd.read_csv(data_file, comment='#')
    if "planet_name" in df.columns:
        columns_of_interest = comparison_columns + ["planet_name"]
    else:
        columns_of_interest = comparison_columns

    data = df[columns_of_interest].fillna(df[columns_of_interest].median())

    # Adicionar Terra e "Similar Earth" com nomes
    earth_data = pd.DataFrame([earth_reference, similar_earth_reference], columns=comparison_columns)
    earth_data["planet_name"] = ["Earth", "Similar Earth"]
    data = pd.concat([data, earth_data], ignore_index=True)

    earth_index = len(data) - 2
    artificial_index = len(data) - 1

    # Aplicar log-transformação
    data_log = data.copy()
    for col in comparison_columns:
        data_log[col] = np.log(data_log[col] + 1)

    # --------------------- LSH ---------------------
    print("\n--- Gerando Hashes LSH ---")
    start_lsh = time.time()
    hashes = generate_lsh_hashes(data_log[comparison_columns], num_hashes=num_hashes)
    lsh_time = time.time() - start_lsh

    hashes_df = pd.DataFrame(hashes, columns=[f"hash_{i}" for i in range(num_hashes)])
    similarity_buckets = hashes_df.groupby(list(hashes_df.columns)).indices
    grouped_planets = {bucket: list(planets) for bucket, planets in similarity_buckets.items()}

    # Criar rótulos para cada ponto com base no bucket do LSH
    unique_buckets = {bucket: idx for idx, bucket in enumerate(grouped_planets.keys())}
    lsh_labels = np.empty(data.shape[0], dtype=int)
    for bucket, indices in grouped_planets.items():
        cluster_id = unique_buckets[bucket]
        for i in indices:
            lsh_labels[i] = cluster_id

    # Filtrar grupos que contenham a Terra para geração dos grafos
    groups_with_earth = [
        (bucket, planets) for bucket, planets in grouped_planets.items()
        if earth_index in planets
    ]

    # --------------------- K-means ---------------------
    print("\n--- Aplicando K-means ---")
    start_kmeans = time.time()
    # Definindo um número fixo de clusters (por exemplo, 5)
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_log[comparison_columns])
    kmeans_time = time.time() - start_kmeans

    # --------------------- KNN (agrupamento via grafo) ---------------------
    print("\n--- Agrupamento via KNN ---")
    start_knn = time.time()
    # Utilizando os 5 vizinhos mais próximos
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(data_log[comparison_columns])
    distances, indices = knn.kneighbors(data_log[comparison_columns])
    
    # Construir grafo onde cada ponto se conecta aos seus vizinhos
    G = nx.Graph()
    n_points = data_log.shape[0]
    G.add_nodes_from(range(n_points))
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                G.add_edge(i, neighbor)
    
    # Extrair componentes conexas como clusters
    knn_labels = np.empty(n_points, dtype=int)
    for cluster_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            knn_labels[node] = cluster_id
    knn_time = time.time() - start_knn

    # --------------------- Comparação dos Agrupamentos ---------------------
    # Utilizando o Adjusted Rand Index (ARI)
    ari_lsh_kmeans = adjusted_rand_score(lsh_labels, kmeans_labels)
    ari_lsh_knn = adjusted_rand_score(lsh_labels, knn_labels)
    ari_kmeans_knn = adjusted_rand_score(kmeans_labels, knn_labels)

    # Gerar arquivo com benchmarking e similaridade
    with open("clustering_comparison.txt", "w") as f:
        f.write("Benchmarking de Métodos de Agrupamento:\n")
        f.write(f"LSH: {lsh_time:.4f} segundos\n")
        f.write(f"K-means: {kmeans_time:.4f} segundos\n")
        f.write(f"KNN (via grafo de vizinhos): {knn_time:.4f} segundos\n\n")
        f.write("Similaridade entre agrupamentos (Adjusted Rand Index):\n")
        f.write(f"LSH vs K-means: {ari_lsh_kmeans:.4f}\n")
        f.write(f"LSH vs KNN: {ari_lsh_knn:.4f}\n")
        f.write(f"K-means vs KNN: {ari_kmeans_knn:.4f}\n")
    
    print("\nComparação de agrupamentos gerada em 'clustering_comparison.txt'.")

    # --------------------- Geração de Grafos para Grupos com a Terra (LSH) ---------------------
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
