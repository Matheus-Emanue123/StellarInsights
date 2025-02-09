import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
import webbrowser
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def map_density_to_color(density):
    """Map planet density to a color gradient from blue to red."""
    norm = plt.Normalize(vmin=0, vmax=10)  # Adjust range if needed
    cmap = plt.get_cmap("coolwarm")  # Blue (low density) → Red (high density)
    rgba_color = cmap(norm(density))
    hex_color = mcolors.to_hex(rgba_color)
    return hex_color

def construct_graph_interactive_earth_edges(data, comparison_columns, earth_name, output_html="graph.html"):
    G = nx.Graph()

    # Create nodes with updated colors and sizes
    for i, row in data.iterrows():
        color = map_density_to_color(row.get("pl_dens", 0))  # Use gradient mapping
        size = max(5, row.get("st_mass", 1.0) * 10)  # Avoid zero-size nodes

        # Store all attributes in node
        node_data = row.to_dict()
        node_data.update({
            "label": f"Planet {i}",
            "color": color,
            "size": size
        })
        G.add_node(i, **node_data)

    # Identify Earth indices
    earth_indices = data.index[data['planet_name'] == earth_name]

    # Create edges between Earth and selected planets (all with fixed thin width)
    for earth_idx in earth_indices:
        for i in data.index:
            if i != earth_idx:
                distance = np.linalg.norm(
                    data.loc[earth_idx, comparison_columns].values - 
                    data.loc[i, comparison_columns].values
                )
                G.add_edge(
                    earth_idx,
                    i,
                    weight=1,  # Fixed thin width
                    distance=round(distance, 2)
                )

    # Export Graph for Gephi
    export_graph_to_csv(G, "nodes.csv", "edges.csv")

    # 📌 Updated PyVis Visualization
    net = Network(height="750px", width="100%", bgcolor="white", font_color="black")
    net.force_atlas_2based(gravity=-50)  # Layout algorithm

    for node, node_data in G.nodes(data=True):
        net.add_node(
            node,
            label=node_data.get("label", str(node)),
            title="<br>".join(f"{k}: {v}" for k, v in node_data.items() if k not in ["label", "color", "size"]),
            color=node_data.get("color", "#000000"),  # Default black if missing
            size=node_data.get("size", 15)
        )

    for source, target, edge_data in G.edges(data=True):
        net.add_edge(
            source,
            target,
            title=f"Distance: {edge_data.get('distance', 1.0):.2f}",
            value=1,  # Fixed thin edges
            color="gray"
        )

    net.show_buttons(filter_=['physics'])
    net.write_html(output_html)
    webbrowser.open(f"file://{os.path.abspath(output_html)}")

    return G

def export_graph_to_csv(G, nodes_path="nodes.csv", edges_path="edges.csv"):
    # Store all node attributes
    nodes_data = []
    
    all_columns = set()
    for _, data in G.nodes(data=True):
        all_columns.update(data.keys())

    for node, data in G.nodes(data=True):
        node_info = {"id": node}
        for col in all_columns:
            node_info[col] = data.get(col, "N/A")  
        nodes_data.append(node_info)

    df_nodes = pd.DataFrame(nodes_data)
    df_nodes.to_csv(nodes_path, index=False, encoding="utf-8")

    # Store edges
    edges_data = []
    for source, target, data in G.edges(data=True):
        edges_data.append({
            "source": source,
            "target": target,
            "weight": data.get("weight", 1.0),
            "distance": data.get("distance", 1.0)
        })
    
    df_edges = pd.DataFrame(edges_data)
    df_edges.to_csv(edges_path, index=False, encoding="utf-8")

    print(f"Files exported: {nodes_path}, {edges_path}")

def construct_graph_planet_to_planet(data, comparison_columns, output_html="graph_planet_to_planet.html"):
    G = nx.Graph()

    # Criar nós
    for i, row in data.iterrows():
        color = map_density_to_color(row.get("pl_dens", 0))  
        size = max(5, row.get("st_mass", 1.0) * 10)

        node_data = row.to_dict()
        node_data.update({
            "label": f"Planet {i}",
            "color": color,
            "size": size
        })
        G.add_node(i, **node_data)

    # Calcular todas as distâncias para definir um threshold dinâmico
    distances = []
    for i in data.index:
        for j in data.index:
            if i < j:  # Evitar cálculos repetidos
                distance = np.linalg.norm(
                    data.loc[i, comparison_columns].values - 
                    data.loc[j, comparison_columns].values
                )
                distances.append(distance)

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    dynamic_threshold = mean_distance - 0.75 * std_distance  # Ajustável conforme necessário

    print(f"Threshold dinâmico definido: {dynamic_threshold}")

    # Criar arestas com base no threshold dinâmico
    for i in data.index:
        for j in data.index:
            if i < j:  
                distance = np.linalg.norm(
                    data.loc[i, comparison_columns].values - 
                    data.loc[j, comparison_columns].values
                )
                similarity = 1 / (distance + 1e-5)

                planet_i = str(data.loc[i, "planet_name"]) if "planet_name" in data.columns else f"Planet {i}"
                planet_j = str(data.loc[j, "planet_name"]) if "planet_name" in data.columns else f"Planet {j}"

# Conectar se a distância for menor que o threshold dinâmico OU se for a Terra ou Similar Earth
                if distance <= dynamic_threshold or "Earth" in planet_i or "Similar Earth" in planet_j:
                    G.add_edge(i, j, weight=similarity, distance=round(distance, 2))


    # Exportar para CSV (Gephi)
    export_graph_to_csv(G, "nodes_planet_to_planet.csv", "edges_planet_to_planet.csv")

    # Visualizar com PyVis
    net = Network(height="750px", width="100%", bgcolor="white", font_color="black")
    net.force_atlas_2based(gravity=-50)  

    for node, node_data in G.nodes(data=True):
        net.add_node(
            node,
            label=node_data.get("label", str(node)),
            title="<br>".join(f"{k}: {v}" for k, v in node_data.items() if k not in ["label", "color", "size"]),
            color=node_data.get("color", "#000000"),
            size=node_data.get("size", 15)
        )

    for source, target, edge_data in G.edges(data=True):
        net.add_edge(
            source,
            target,
            title=f"Distance: {edge_data.get('distance', 1.0):.2f}",
            value=edge_data.get("weight", 1),
            color="gray"
        )

    net.show_buttons(filter_=['physics'])
    net.write_html(output_html)
    webbrowser.open(f"file://{os.path.abspath(output_html)}")

    return G

    