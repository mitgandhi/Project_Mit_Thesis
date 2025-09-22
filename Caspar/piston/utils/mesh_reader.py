import numpy as np
import re


class MeshReader:
    def __init__(self, filename):
        self.filename = filename
        self.nodes = {}
        self.elements = []
        self.current_section = None

    def read_file(self):
        """Read and parse the INP file"""
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('*'):
                        self.handle_section_header(line)
                    else:
                        self.handle_section_data(line)

            return self.convert_to_arrays()

        except Exception as e:
            print(f"Error reading file: {e}")
            raise

    def handle_section_header(self, line):
        """Process section headers"""
        self.current_section = line.lower()
        print(f"Processing section: {line}")

    def handle_section_data(self, line):
        """Process data lines based on current section"""
        if not self.current_section:
            return

        try:
            if 'node' in self.current_section:
                self.process_node(line)
            elif 'element' in self.current_section:
                self.process_element(line)
        except Exception as e:
            print(f"Warning: Could not process line '{line}': {e}")

    def process_node(self, line):
        """Process a node line"""
        parts = re.split(r'[,\s]+', line.strip())
        parts = [p for p in parts if p]  # Remove empty strings

        try:
            node_id = int(parts[0])
            coords = [float(x) for x in parts[1:4]]
            self.nodes[node_id] = coords
        except (ValueError, IndexError) as e:
            print(f"Warning: Invalid node data '{line}': {e}")

    def process_element(self, line):
        """Process an element line"""
        parts = re.split(r'[,\s]+', line.strip())
        try:
            element_data = [int(x) for x in parts if x]
            self.elements.append(element_data)
        except ValueError as e:
            print(f"Warning: Invalid element data '{line}': {e}")

    def convert_to_arrays(self):
        """Convert parsed data to numpy arrays"""
        print(f"\nConverting data: {len(self.nodes)} nodes, {len(self.elements)} elements")

        # Sort nodes by ID
        node_ids = sorted(self.nodes.keys())
        nodes_array = np.array([self.nodes[nid] for nid in node_ids])
        elements_array = np.array(self.elements)

        print(f"Node array shape: {nodes_array.shape}")
        print(f"Element array shape: {elements_array.shape}")

        return nodes_array, elements_array


class MeshProcessor:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.mesh_data = None

    def process(self):
        """Process the mesh data into structured format"""
        # Extract coordinates
        X = self.nodes[:, 0]
        Y = self.nodes[:, 1]
        Z = self.nodes[:, 2]

        # Calculate cylindrical coordinates
        R = np.sqrt(X ** 2 + Y ** 2)
        Theta = np.arctan2(Y, X)

        # Get mesh dimensions
        self.get_mesh_dimensions(R, Theta, Z)

        # Create structured grid
        self.create_structured_grid(X, Y, Z, R, Theta)

        return self.mesh_data

    def get_mesh_dimensions(self, R, Theta, Z):
        """Determine mesh dimensions"""
        # Get unique values with tolerance
        tol = 1e-6
        unique_r = np.unique(np.round(R / tol) * tol)
        unique_theta = np.unique(np.round(Theta / tol) * tol)
        unique_z = np.unique(np.round(Z / tol) * tol)

        self.N = len(unique_r)
        self.M = len(unique_theta)
        self.Q = len(unique_z)

        print(f"\nMesh dimensions: {self.N}x{self.M}x{self.Q}")
        print(f"R range: [{unique_r.min():.6f}, {unique_r.max():.6f}]")
        print(f"Theta range: [{unique_theta.min():.6f}, {unique_theta.max():.6f}]")
        print(f"Z range: [{unique_z.min():.6f}, {unique_z.max():.6f}]")

    def create_structured_grid(self, X, Y, Z, R, Theta):
        """Create structured grid from unstructured data"""
        try:
            # Sort points
            sort_idx = np.lexsort((Z, Theta, R))

            # Reshape arrays
            self.mesh_data = {
                'X': X[sort_idx].reshape(self.N, self.M, self.Q),
                'Y': Y[sort_idx].reshape(self.N, self.M, self.Q),
                'Z': Z[sort_idx].reshape(self.N, self.M, self.Q),
                'R': R[sort_idx].reshape(self.N, self.M, self.Q),
                'Theta': Theta[sort_idx].reshape(self.N, self.M, self.Q),
                'dimensions': (self.N, self.M, self.Q)
            }

            # Calculate area elements
            dr = np.mean(np.diff(np.unique(R)))
            dtheta = np.mean(np.diff(np.unique(Theta)))
            self.mesh_data['dAz'] = self.mesh_data['R'] * dtheta * dr

            print("\nStructured grid created successfully")

        except Exception as e:
            print(f"Error creating structured grid: {e}")
            raise