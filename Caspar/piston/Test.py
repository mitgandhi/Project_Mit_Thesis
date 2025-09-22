import numpy as np
import matplotlib.pyplot as plt


def read_inp_file(file_path):
    """
    Read an Abaqus .inp file and extract node and node set information.
    """
    nodes = {}
    node_sets = {}
    current_section = None
    current_node_set = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith('**') or not line:
                continue

            # Check for node section
            if line.startswith('*NODE'):
                current_section = 'nodes'
                current_node_set = None
                continue

            # Check for node set definition
            if line.startswith('*NSET'):
                current_section = 'node_set'
                current_node_set = line.split('=')[1].strip()
                node_sets[current_node_set] = []
                continue

            # Parse nodes
            if current_section == 'nodes':
                try:
                    parts = line.split(',')
                    node_id = int(parts[0])
                    coords = [float(p.strip()) for p in parts[1:]]
                    nodes[node_id] = coords
                except (ValueError, IndexError):
                    continue

            # Parse node sets
            elif current_section == 'node_set':
                try:
                    # Handle comma-separated node IDs potentially spread across multiple lines
                    node_ids = [int(n.strip()) for n in line.split(',') if n.strip()]
                    node_sets[current_node_set].extend(node_ids)
                except (ValueError, IndexError):
                    continue

    return {
        'nodes': nodes,
        'node_sets': node_sets
    }


def plot_node_set_distribution(file_path):
    """
    Visualize node sets in 3D space with improved scaling and view.
    """
    inp_data = read_inp_file(file_path)
    nodes = inp_data['nodes']
    node_sets = inp_data['node_sets']

    # Safely convert nodes to numpy array
    node_ids = list(nodes.keys())

    # Ensure consistent coordinate dimensions
    node_coords = np.array([nodes[node_id] for node_id in node_ids if len(nodes[node_id]) == 3])

    # Plot
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for different node sets
    colors = plt.cm.rainbow(np.linspace(0, 1, len(node_sets)))

    # Plot each node set
    for (name, set_nodes), color in zip(node_sets.items(), colors):
        # Get coordinates for this node set
        try:
            # Filter out node IDs not in the original nodes dictionary or with incorrect dimensions
            valid_nodes = [node_id for node_id in set_nodes if node_id in nodes and len(nodes[node_id]) == 3]
            set_coords = np.array([nodes[node_id] for node_id in valid_nodes])

            if set_coords.size > 0:
                ax.scatter(set_coords[:, 0], set_coords[:, 1], set_coords[:, 2],
                           label=name, alpha=0.7, color=color, s=10)
        except Exception as e:
            print(f"Error plotting node set {name}: {e}")

    # Set equal aspect ratio
    max_range = np.array([node_coords[:, 0].max() - node_coords[:, 0].min(),
                          node_coords[:, 1].max() - node_coords[:, 1].min(),
                          node_coords[:, 2].max() - node_coords[:, 2].min()]).max() / 2.0
    mid_x = (node_coords[:, 0].max() + node_coords[:, 0].min()) * 0.5
    mid_y = (node_coords[:, 1].max() + node_coords[:, 1].min()) * 0.5
    mid_z = (node_coords[:, 2].max() + node_coords[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title('Node Sets Distribution', fontsize=16)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_zlabel('Z coordinate', fontsize=12)

    # Adjust legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def analyze_node_sets(file_path):
    """
    Analyze node sets in the .inp file.
    """
    inp_data = read_inp_file(file_path)
    node_sets = inp_data['node_sets']

    return {
        'total_node_sets': len(node_sets),
        'node_set_details': {
            name: {
                'num_nodes': len(nodes),
                'min_node_id': min(nodes) if nodes else None,
                'max_node_id': max(nodes) if nodes else None
            } for name, nodes in node_sets.items()
        }
    }


def export_node_set_info(file_path, output_file='node_set_info.txt'):
    """
    Export node set information to a text file.
    """
    inp_data = read_inp_file(file_path)
    node_sets = inp_data['node_sets']

    with open(output_file, 'w') as f:
        f.write("Node Set Information:\n")
        f.write("=" * 30 + "\n")
        for name, nodes in node_sets.items():
            f.write(f"Node Set: {name}\n")
            f.write(f"Number of Nodes: {len(nodes)}\n")
            f.write(f"Node IDs: {min(nodes)} to {max(nodes)}\n")
            f.write("-" * 30 + "\n")

    print(f"Node set information exported to {output_file}")


# Example usage
if __name__ == '__main__':
    inp_file_path = 'input/IM_piston.inp'



    # Print node set analysis
    print(analyze_node_sets(inp_file_path))

    # Visualize node sets
    plot_node_set_distribution(inp_file_path)

    # Export node set information to a text file
    export_node_set_info(inp_file_path)

# Required dependencies:
# pip install numpy matplotlib