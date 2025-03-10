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
    # Caminho do arquivo (raw string para evitar problemas com barras invertidas)
    data_file = r"C:\Users\hecla\OneDrive\Área de Trabalho\CEFET\4 periodo\AEDS II\TRABALHOS\Aplicação de Grafos\databases\PS_2025.01.12_10.15.56.csv"
    
    comparison_columns = [
        "st_teff", "st_rad", "st_mass", "pl_insol",
        "pl_eqt", "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_dens", "st_met"
    ]
    num_hashes = 20

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

    # Ler o CSV (evitando warning com low_memory=False)
    df = pd.read_csv(data_file, comment='#', low_memory=False)
    
    # Converter as colunas de interesse para numérico (valores inválidos viram NaN)
    for col in comparison_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preencher NaNs com a mediana de cada coluna
    df[comparison_columns] = df[comparison_columns].fillna(df[comparison_columns].median())

    # Selecionar as colunas de interesse (incluindo 'pl_name' se disponível)
    if "pl_name" in df.columns:
        columns_of_interest = comparison_columns + ["pl_name"]
    else:
        columns_of_interest = comparison_columns

    data = df[columns_of_interest].copy()

    # Adicionar Terra e "Similar Earth" com nomes na coluna "pl_name"
    earth_data = pd.DataFrame([earth_reference, similar_earth_reference], columns=comparison_columns)
    earth_data["pl_name"] = ["Earth", "Similar Earth"]
    data = pd.concat([data, earth_data], ignore_index=True)

    earth_index = len(data) - 2
    artificial_index = len(data) - 1

    # Aplicar log-transformação usando np.log1p (log(x+1))
    data_log = data.copy()
    for col in comparison_columns:
        data_log[col] = np.where(data_log[col] > -1, data_log[col], -0.999)
        data_log[col] = np.log1p(data_log[col])
    
    data_log[comparison_columns] = data_log[comparison_columns].fillna(data_log[comparison_columns].median())

    # --------------------- Benchmarking (10 execuções) ---------------------
    lsh_times = []
    kmeans_times = []
    knn_times = []
    ari_lsh_kmeans_list = []
    ari_lsh_knn_list = []
    ari_kmeans_knn_list = []

    # Para consistência, manter a Terra com o mesmo valor de log-transformação
    earth_log = data_log.iloc[earth_index][comparison_columns].values
    n_clusters = 6

    for exec_num in range(10):
        print(f"\nExecução {exec_num+1} de 10:")

        # LSH
        start_lsh = time.time()
        hashes = generate_lsh_hashes(data_log[comparison_columns], num_hashes=num_hashes)
        current_lsh_time = time.time() - start_lsh
        lsh_times.append(current_lsh_time)

        # Processar resultados do LSH
        hashes_df = pd.DataFrame(hashes, columns=[f"hash_{i}" for i in range(num_hashes)])
        similarity_buckets = hashes_df.groupby(list(hashes_df.columns)).indices
        grouped_planets = {bucket: list(planets) for bucket, planets in similarity_buckets.items()}

        lsh_labels = np.empty(data.shape[0], dtype=int)
        unique_buckets = {bucket: idx for idx, bucket in enumerate(grouped_planets.keys())}
        for bucket, indices in grouped_planets.items():
            cluster_id = unique_buckets[bucket]
            for i in indices:
                lsh_labels[i] = cluster_id

        # K-means
        start_kmeans = time.time()
        # Selecionar aleatoriamente os demais centróides (garantindo que não seja a Terra)
        random_indices = np.random.choice(data_log.index[data_log.index != earth_index], n_clusters - 1, replace=False)
        init_centroids = np.vstack([earth_log, data_log.loc[random_indices, comparison_columns].values])
        kmeans = KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1, random_state=42)
        kmeans_labels = kmeans.fit_predict(data_log[comparison_columns])
        current_kmeans_time = time.time() - start_kmeans
        kmeans_times.append(current_kmeans_time)

        # KNN (agrupamento via grafo)
        start_knn = time.time()
        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(data_log[comparison_columns])
        distances, indices = knn.kneighbors(data_log[comparison_columns])
        G = nx.Graph()
        n_points = data_log.shape[0]
        G.add_nodes_from(range(n_points))
        for idx, neighbors in enumerate(indices):
            for neighbor in neighbors:
                if idx != neighbor:
                    G.add_edge(idx, neighbor)
        knn_labels = np.empty(n_points, dtype=int)
        for cluster_id, component in enumerate(nx.connected_components(G)):
            for node in component:
                knn_labels[node] = cluster_id
        current_knn_time = time.time() - start_knn
        knn_times.append(current_knn_time)

        # Cálculo dos ARI (Adjusted Rand Index)
        ari_lsh_kmeans = adjusted_rand_score(lsh_labels, kmeans_labels)
        ari_lsh_knn = adjusted_rand_score(lsh_labels, knn_labels)
        ari_kmeans_knn = adjusted_rand_score(kmeans_labels, knn_labels)

        ari_lsh_kmeans_list.append(ari_lsh_kmeans)
        ari_lsh_knn_list.append(ari_lsh_knn)
        ari_kmeans_knn_list.append(ari_kmeans_knn)

    # Cálculo das médias
    avg_lsh_time = np.mean(lsh_times)
    avg_kmeans_time = np.mean(kmeans_times)
    avg_knn_time = np.mean(knn_times)
    avg_ari_lsh_kmeans = np.mean(ari_lsh_kmeans_list)
    avg_ari_lsh_knn = np.mean(ari_lsh_knn_list)
    avg_ari_kmeans_knn = np.mean(ari_kmeans_knn_list)

    # Gravação dos resultados no arquivo TXT
    with open("clustering_comparison.txt", "w") as f:
        f.write("Benchmarking de Métodos de Agrupamento (média de 10 execuções):\n")
        f.write(f"LSH: {avg_lsh_time:.4f} segundos\n")
        f.write(f"K-means: {avg_kmeans_time:.4f} segundos\n")
        f.write(f"KNN (via grafo de vizinhos): {avg_knn_time:.4f} segundos\n\n")
        f.write("Média da Similaridade entre agrupamentos (Adjusted Rand Index):\n")
        f.write(f"LSH vs K-means: {avg_ari_lsh_kmeans:.4f}\n")
        f.write(f"LSH vs KNN: {avg_ari_lsh_knn:.4f}\n")
        f.write(f"K-means vs KNN: {avg_ari_kmeans_knn:.4f}\n")
    
    print("\nBenchmarking realizado e resultados salvos em 'clustering_comparison.txt'.")

    # --------------------- Geração de Grafos para Grupos com a Terra ---------------------
    # Utiliza-se o agrupamento LSH da última execução (determinístico devido à semente)
    groups_with_earth = [
        (bucket, planets) for bucket, planets in grouped_planets.items()
        if earth_index in planets
    ]

    for bucket, planets in groups_with_earth:
        group_data = data.iloc[planets].copy()
        if "pl_name" in group_data.columns:
            group_data = group_data.drop_duplicates(subset="pl_name", keep="first")
            planet_names = group_data["pl_name"].unique()
        else:
            planet_names = planets
        
        if "pl_name" in group_data.columns and "planet_name" not in group_data.columns:
            group_data["planet_name"] = group_data["pl_name"]

        print(f"\nGerando grafo conectado à Terra para o bucket {bucket} com planetas {planet_names}")
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
