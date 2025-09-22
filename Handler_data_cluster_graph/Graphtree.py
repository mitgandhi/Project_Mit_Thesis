import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

# Load Design Parameters (Ensure DS-ID is an integer)
design_params = pd.read_csv("Z:/Studenten/Mit/Thesis/PyProject/Datasets/DS/DS.csv")
design_params["DS-ID"] = design_params["DS-ID"].astype(int)

# Load Operating Conditions (Ensure DS-ID is an integer)
operating_conditions = pd.read_csv("Z:/Studenten/Mit/Thesis/PyProject/Datasets/DS/DSP1.csv")
operating_conditions["DS-ID"] = operating_conditions["DS-ID"].astype(int)

# Initialize Graph
G = nx.DiGraph()

# Add Root Node
G.add_node("Dataset", subset=0)  # Root level

# Define node categories and labels
node_colors = {}
node_labels = {}

# Add Design Parameter Nodes (First Layer)
for i, (_, design) in enumerate(design_params.iterrows()):
    design_id = f"C{design['DS-ID']}"
    G.add_node(design_id, clearance=design["Clearance"], subset=1)  # Assign subset level
    G.add_edge("Dataset", design_id)  # Connect to root

    node_labels[design_id] = f"C{design['DS-ID']}\n(Cl: {design['Clearance']})"
    node_colors[design_id] = "blue"  # Design parameter nodes

    # Add Operating Condition Nodes (Second Layer)
    for j, (_, op) in enumerate(operating_conditions[operating_conditions["DS-ID"] == design["DS-ID"]].iterrows()):
        op_id = f"OP{op['ID']}"
        G.add_node(op_id, speed=op["speed"], pressure=op["pressure"], subset=2)  # Assign subset level
        G.add_edge(design_id, op_id)  # Connect to design parameter

        node_labels[op_id] = f"OP{op['ID']}\n({op['speed']} RPM, {op['pressure']} bar)"
        node_colors[op_id] = "green"  # Operating condition nodes

# Assign color to root node
node_colors["Dataset"] = "red"  # Root node
node_labels["Dataset"] = "Dataset"

# Generate **hierarchical left-to-right layout**
pos = nx.multipartite_layout(G, subset_key="subset")

# Flip X and Y coordinates for **side view**
pos = {node: (y, -x) for node, (x, y) in pos.items()}

# Create the figure
plt.figure(figsize=(14, 6), dpi=300)  # High resolution for PPT

# Draw Nodes
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=[node_colors[n] for n in G.nodes()], edgecolors="black")

# Draw Edges
nx.draw_networkx_edges(G, pos, alpha=0.6, width=1.5, arrowstyle="-|>")

# Draw Labels
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_family="Arial", font_weight="bold")

# Title
plt.title("Hierarchical Tree View of Multi-Condition Simulation Data", fontsize=14, fontweight="bold")

# Save the figure as a high-quality PNG for PPT
plt.savefig("side_view_simulation_graph.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
