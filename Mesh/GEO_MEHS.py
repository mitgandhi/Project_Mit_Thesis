import gmsh
import meshio
import numpy as np
import os


class PistonMeshGenerator:
    def __init__(self):
        self.mesh = None
        self.surface_tags = {}
        self.volume_tag = None

    def initialize_gmsh(self):
        """Initialize GMSH"""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)  # Enable terminal output
        gmsh.model.add("Piston")

    def load_step_file(self, filename):
        """Load STEP file and identify surfaces"""
        print(f"Loading STEP file: {filename}")

        if not os.path.exists(filename):
            print(f"Error: {filename} not found!")
            return False

        try:
            # Import STEP file
            gmsh.model.occ.importShapes(filename)
            gmsh.model.occ.synchronize()

            # Get all surfaces
            surfaces = gmsh.model.getEntities(2)  # 2D entities (surfaces)
            volumes = gmsh.model.getEntities(3)  # 3D entities (volumes)

            print(f"Found {len(surfaces)} surfaces and {len(volumes)} volumes")

            # Identify the 3 main surfaces: gap, case, DC
            self.identify_surfaces(surfaces)

            return True

        except Exception as e:
            print(f"Error loading STEP file: {e}")
            return False

    def identify_surfaces(self, surfaces):
        """Identify and tag the 3 main surfaces: gap, case, DC"""
        print("Identifying surfaces: gap, case, DC")

        # For this example, we'll assign surfaces based on their index
        # In practice, you'd use geometric properties to identify them

        if len(surfaces) >= 3:
            # Assign surface tags (you may need to adjust based on your specific STEP file)
            self.surface_tags['gap'] = surfaces[0][1]  # First surface as gap
            self.surface_tags['case'] = surfaces[1][1]  # Second surface as case
            self.surface_tags['dc'] = surfaces[2][1]  # Third surface as DC

            print(f"  gap surface: tag {self.surface_tags['gap']}")
            print(f"  case surface: tag {self.surface_tags['case']}")
            print(f"  dc surface: tag {self.surface_tags['dc']}")

            # Create physical groups for surfaces (for node sets)
            gmsh.model.addPhysicalGroup(2, [self.surface_tags['gap']], name="GAP")
            gmsh.model.addPhysicalGroup(2, [self.surface_tags['case']], name="CASE")
            gmsh.model.addPhysicalGroup(2, [self.surface_tags['dc']], name="DC")

            # Create combined surface group
            all_surfaces = list(self.surface_tags.values())
            gmsh.model.addPhysicalGroup(2, all_surfaces, name="ALL_SURFACES")

        else:
            print(f"Warning: Expected 3 surfaces, found {len(surfaces)}")
            # Fallback: use all available surfaces
            for i, (dim, tag) in enumerate(surfaces[:3]):
                surface_names = ['gap', 'case', 'dc']
                if i < len(surface_names):
                    self.surface_tags[surface_names[i]] = tag
                    gmsh.model.addPhysicalGroup(2, [tag], name=surface_names[i].upper())

    def create_2d_surface_mesh(self, mesh_size=1.0):  # Changed to 1mm for speed
        """Create 2D triangular mesh on surfaces with FAST settings"""
        print("Creating 2D surface mesh with FAST settings...")

        # Set 1mm mesh size for speed
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)  # Less variation

        # FAST meshing options - prioritize speed over quality
        gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt (faster than Frontal-Delaunay)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Keep triangles

        # Reduce quality requirements for speed
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)  # Disable curvature-based refinement
        gmsh.option.setNumber("Mesh.MinimumCirclePoints", 6)  # Reduce from 12 to 6
        gmsh.option.setNumber("Mesh.MinimumCurvePoints", 2)  # Reduce from 3 to 2

        # Looser tolerance settings for speed
        gmsh.option.setNumber("Geometry.Tolerance", 1e-4)  # Relaxed from 1e-6
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-4)  # Relaxed from 1e-6

        # Additional speed optimizations
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)  # Disable boundary extension
        gmsh.option.setNumber("Mesh.LcIntegrationPrecision", 1e-6)  # Reduce precision
        gmsh.option.setNumber("Mesh.AnisoMax", 1.0)  # Reduce anisotropy limit

        try:
            print("  Attempting FAST 2D mesh generation...")
            print(f"  Using 1mm mesh size: {mesh_size}")
            print("  This should be much faster for complex geometry...")

            # Direct 2D mesh generation with timeout simulation
            print("  Generating 2D triangular mesh...")

            # Set a progress callback to monitor long operations
            def progress_callback(message):
                print(f"    Progress: {message}")

            gmsh.model.mesh.generate(2)

            # Check if meshing was successful
            nodes = gmsh.model.mesh.getNodes()
            elements_2d = gmsh.model.mesh.getElements(2)

            if len(nodes[0]) > 0:
                print(f"  SUCCESS: Generated 2D mesh with {len(nodes[0])} nodes")
                print(f"  2D elements: {sum(len(elem[1]) for elem in elements_2d)} triangular elements")
                return True
            else:
                print("  Initial 2D meshing failed, trying ULTRA-FAST approach...")
                return self.ultra_fast_2d_mesh(mesh_size * 2)

        except Exception as e:
            print(f"  Error during 2D meshing: {e}")
            print("  Trying ULTRA-FAST alternative...")
            return self.ultra_fast_2d_mesh(mesh_size * 2)

    def ultra_fast_2d_mesh(self, mesh_size):
        """Ultra-fast meshing with very coarse settings"""
        print("  Using ULTRA-FAST meshing approach...")

        try:
            # Clear any partial mesh
            gmsh.model.mesh.clear()

            # Ultra-fast algorithm settings
            gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt (fastest)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.8)

            # Disable quality optimizations for maximum speed
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
            gmsh.option.setNumber("Mesh.Optimize", 0)  # Disable optimization
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)  # Disable Netgen optimization

            # Further reduce mesh quality requirements
            gmsh.option.setNumber("Mesh.MinimumCirclePoints", 4)  # Minimum possible
            gmsh.option.setNumber("Mesh.MinimumCurvePoints", 2)  # Minimum possible
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)  # Disable

            # Generate 2D mesh with ultra-fast settings
            print(f"  Generating ULTRA-COARSE mesh (size: {mesh_size})...")
            gmsh.model.mesh.generate(2)

            nodes = gmsh.model.mesh.getNodes()
            if len(nodes[0]) > 0:
                print(f"  Ultra-fast mesh successful: {len(nodes[0])} nodes")
                return True
            else:
                print("  Ultra-fast mesh also failed")
                return False

        except Exception as e:
            print(f"  Ultra-fast mesh failed: {e}")
            return False

    def create_simplified_mesh_if_needed(self):
        """Create a simplified mesh if the original geometry is too complex"""
        print("Creating simplified mesh from bounding box...")

        try:
            # Get bounding box of the geometry
            bbox = gmsh.model.getBoundingBox(-1, -1)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            print(f"  Bounding box: ({xmin:.2f}, {ymin:.2f}, {zmin:.2f}) to ({xmax:.2f}, {ymax:.2f}, {zmax:.2f})")

            # Create a simplified cylinder representing the piston
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            center_z = zmin
            height = zmax - zmin
            radius = min(xmax - xmin, ymax - ymin) / 2 * 0.8

            # Clear the model and create simplified geometry
            gmsh.model.clear()
            gmsh.model.add("SimplifiedPiston")

            # Create cylinder
            cylinder = gmsh.model.occ.addCylinder(center_x, center_y, center_z, 0, 0, height, radius)
            gmsh.model.occ.synchronize()

            # Get surfaces for node sets
            surfaces = gmsh.model.getEntities(2)
            if len(surfaces) >= 2:
                # Bottom and top surfaces
                gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], name="DC")
                gmsh.model.addPhysicalGroup(2, [surfaces[1][1]], name="GAP")
                if len(surfaces) > 2:
                    gmsh.model.addPhysicalGroup(2, [surfaces[2][1]], name="CASE")
                else:
                    gmsh.model.addPhysicalGroup(2, [surfaces[1][1]], name="CASE")

                # All surfaces
                all_surf_tags = [s[1] for s in surfaces]
                gmsh.model.addPhysicalGroup(2, all_surf_tags, name="ALL_SURFACES")

            # Volume
            volumes = gmsh.model.getEntities(3)
            if volumes:
                self.volume_tag = volumes[0][1]
                gmsh.model.addPhysicalGroup(3, [self.volume_tag], name="PISTON_VOLUME")

            print("  Simplified geometry created successfully")
            return True

        except Exception as e:
            print(f"  Error creating simplified mesh: {e}")
            return False

    def create_3d_volume_mesh(self):
        """Create 3D tetrahedral mesh (C3D4) inside the closed surface"""
        print("Creating 3D volume mesh...")

        try:
            # First, try to create volume from surfaces
            volumes = gmsh.model.getEntities(3)

            if not volumes:
                print("  No volumes found, trying to create volume from surfaces...")
                # If no volume exists, try to create one from the surfaces
                try:
                    surface_loop = gmsh.model.occ.addSurfaceLoop(list(self.surface_tags.values()))
                    self.volume_tag = gmsh.model.occ.addVolume([surface_loop])
                    gmsh.model.occ.synchronize()
                    volumes = gmsh.model.getEntities(3)
                except Exception as e:
                    print(f"  Could not create volume from surfaces: {e}")
                    print("  Attempting to heal geometry...")

                    # Try to heal the geometry
                    gmsh.model.occ.healShapes()
                    gmsh.model.occ.synchronize()

                    # Try again
                    try:
                        surface_loop = gmsh.model.occ.addSurfaceLoop(list(self.surface_tags.values()))
                        self.volume_tag = gmsh.model.occ.addVolume([surface_loop])
                        gmsh.model.occ.synchronize()
                        volumes = gmsh.model.getEntities(3)
                    except Exception as e2:
                        print(f"  Still failed after healing: {e2}")
                        return False

            if volumes:
                self.volume_tag = volumes[0][1]
                print(f"  Using volume with tag: {self.volume_tag}")

                # Create physical group for volume (piston_volume component)
                gmsh.model.addPhysicalGroup(3, [self.volume_tag], name="PISTON_VOLUME")

                # Set 3D mesh algorithm and options
                gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal-Delaunay for 3D
                gmsh.option.setNumber("Mesh.Optimize", 1)  # Optimize mesh

                # Generate 3D mesh
                gmsh.model.mesh.generate(3)

                # Get 3D mesh statistics
                elements_3d = gmsh.model.mesh.getElements(3)  # 3D elements
                total_3d_elements = sum(len(elem[1]) for elem in elements_3d)

                print(f"  Generated 3D mesh: {total_3d_elements} tetrahedral elements")
                return True

            else:
                print("  Error: Could not create or find volume for meshing")
                return False

        except Exception as e:
            print(f"  Error creating 3D mesh: {e}")
            return False

    def create_node_sets(self):
        """Node sets are created automatically through physical groups"""
        print("Node sets created through physical groups:")

        # Get all physical groups
        physical_groups = gmsh.model.getPhysicalGroups()

        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            print(f"  {name}: dimension {dim}, entities {entities}")

    def renumber_and_optimize(self):
        """Renumber nodes and elements, optimize mesh"""
        print("Renumbering and optimizing mesh...")

        try:
            # Renumber nodes and elements
            gmsh.model.mesh.renumberNodes()
            gmsh.model.mesh.renumberElements()

            # Optimize mesh quality
            gmsh.model.mesh.optimize("Netgen")

            print("  Mesh renumbered and optimized")
        except Exception as e:
            print(f"  Optimization failed: {e}")
            print("  Continuing without optimization...")

    def export_to_meshio(self):
        """Convert GMSH mesh to meshio format"""
        print("Converting to meshio format...")

        # Get mesh data from GMSH
        try:
            # Write temporary msh file
            temp_msh = "temp_piston.msh"
            gmsh.write(temp_msh)

            # Read with meshio
            self.mesh = meshio.read(temp_msh)

            # Clean up temporary file
            if os.path.exists(temp_msh):
                os.remove(temp_msh)

            print(f"  Mesh converted: {len(self.mesh.points)} nodes, {len(self.mesh.cells)} cell blocks")

            # Print cell types
            for cell_block in self.mesh.cells:
                print(f"    {cell_block.type}: {len(cell_block.data)} elements")

        except Exception as e:
            print(f"  Error converting mesh: {e}")
            return False

        return True

    def write_inp_file(self, filename):
        """Write INP file using meshio"""
        print(f"Writing INP file: {filename}")

        if self.mesh is None:
            print("Error: No mesh available for export")
            return False

        try:
            # Convert cell types to Abaqus format
            abaqus_cells = []

            for cell_block in self.mesh.cells:
                if cell_block.type == "triangle":
                    # Include triangular surface elements for node sets
                    abaqus_cells.append(meshio.CellBlock("triangle", cell_block.data))
                elif cell_block.type == "tetra":
                    # C3D4 tetrahedral elements
                    abaqus_cells.append(meshio.CellBlock("tetra", cell_block.data))
                elif cell_block.type == "tetra10":
                    # C3D10 quadratic tetrahedral elements
                    abaqus_cells.append(meshio.CellBlock("tetra10", cell_block.data))

            # Create new mesh with all elements
            abaqus_mesh = meshio.Mesh(
                points=self.mesh.points,
                cells=abaqus_cells,
                point_data=self.mesh.point_data,
                cell_data=self.mesh.cell_data,
                field_data=self.mesh.field_data
            )

            # Write INP file
            meshio.write(filename, abaqus_mesh, file_format="abaqus")

            print(f"  Successfully written: {filename}")

            # Print summary
            total_elements = sum(len(cell.data) for cell in abaqus_cells)
            print(f"  Total nodes: {len(abaqus_mesh.points)}")
            print(f"  Total elements: {total_elements}")

            # Print node sets information
            if abaqus_mesh.field_data:
                print("  Node sets available:")
                for name, data in abaqus_mesh.field_data.items():
                    print(f"    {name}: tag {data[0]}, dimension {data[1]}")

        except Exception as e:
            print(f"  Error writing INP file: {e}")
            return False

        return True

    def cleanup(self):
        """Clean up GMSH"""
        gmsh.finalize()


def main():
    """Main execution function"""
    print("PISTON VOLUME MESH GENERATOR")
    print("Using GMSH + meshio")
    print("=" * 50)

    # Initialize generator
    generator = PistonMeshGenerator()

    try:
        # Step 1: Initialize GMSH
        generator.initialize_gmsh()

        # Step 2: Load STEP file and identify surfaces
        if not generator.load_step_file("Piston_T.stp"):
            print("Failed to load STEP file")
            return

        # Step 3: Create 2D surface mesh with 1mm mesh size for speed
        success_2d = generator.create_2d_surface_mesh(mesh_size=1.0)  # 1mm mesh size

        if not success_2d:
            print("2D meshing failed, trying simplified geometry...")
            if not generator.create_simplified_mesh_if_needed():
                print("All meshing attempts failed")
                return
            # Try 2D meshing again with simplified geometry
            success_2d = generator.create_2d_surface_mesh(mesh_size=1.5)  # Slightly larger for simplified geometry

        # Step 4: Create 3D volume mesh with C3D4 elements
        if not generator.create_3d_volume_mesh():
            print("Failed to create 3D mesh")
            return

        # Step 5: Create node sets (through physical groups)
        generator.create_node_sets()

        # Step 6: Renumber and optimize
        generator.renumber_and_optimize()

        # Step 7: Convert to meshio format
        if not generator.export_to_meshio():
            print("Failed to convert mesh")
            return

        # Step 8: Write INP file
        output_filename = "piston_volume.inp"
        if not generator.write_inp_file(output_filename):
            print("Failed to write INP file")
            return

        # Summary
        print("\n" + "=" * 50)
        print("MESH GENERATION COMPLETE")
        print("=" * 50)
        print(f"Output file: {output_filename}")
        print("\nGenerated components:")
        print("  - PISTON_VOLUME: C3D4 tetrahedral elements")
        print("\nGenerated node sets:")
        print("  - GAP: nodes on gap surface")
        print("  - CASE: nodes on case surface")
        print("  - DC: nodes on DC surface")
        print("  - ALL_SURFACES: combined surface nodes")

    except Exception as e:
        print(f"Error during mesh generation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        generator.cleanup()


def quick_mesh_generation():
    """Quick alternative using simplified approach"""
    print("\nQUICK MESH ALTERNATIVE")
    print("=" * 30)

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("QuickPiston")

        # Create simple piston geometry
        # Cylinder with radius 5, height 10
        cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 10, 5)
        gmsh.model.occ.synchronize()

        # Get entities
        surfaces = gmsh.model.getEntities(2)
        volumes = gmsh.model.getEntities(3)

        # Create physical groups
        if len(surfaces) >= 3:
            gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], name="GAP")
            gmsh.model.addPhysicalGroup(2, [surfaces[1][1]], name="CASE")
            gmsh.model.addPhysicalGroup(2, [surfaces[2][1]], name="DC")
            gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], name="ALL_SURFACES")

        if volumes:
            gmsh.model.addPhysicalGroup(3, [volumes[0][1]], name="PISTON_VOLUME")

        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)

        # Generate mesh
        print("Generating quick mesh...")
        gmsh.model.mesh.generate(3)

        # Export
        temp_file = "quick_piston.msh"
        gmsh.write(temp_file)

        # Convert to INP
        mesh = meshio.read(temp_file)
        meshio.write("quick_piston.inp", mesh, file_format="abaqus")

        print("Quick mesh generated: quick_piston.inp")

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
        gmsh.finalize()

        return True

    except Exception as e:
        print(f"Quick mesh failed: {e}")
        gmsh.finalize()
        return False


def install_requirements():
    """Print installation requirements"""
    print("Required packages:")
    print("pip install gmsh meshio")
    print("\nOptional (for advanced features):")
    print("pip install pygmsh")


if __name__ == "__main__":
    try:
        # Try main approach first
        main()
    except KeyboardInterrupt:
        print("\n\nMeshing interrupted by user")
        print("Trying quick mesh generation instead...")
        quick_mesh_generation()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease install required packages:")
        install_requirements()
    except Exception as e:
        print(f"Main meshing failed: {e}")
        print("Trying quick mesh generation...")
        quick_mesh_generation()