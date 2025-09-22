import gmsh
import meshio
import os
import time


def piston_volume_mesh(step_file, mesh_size=1.0, output_inp="piston_volume.inp"):
    print("\n🚀 PISTON VOLUME MESHING (with C3D4 and NSETS)")
    print("=" * 60)

    if not os.path.exists(step_file):
        print(f"❌ STEP file not found: {step_file}")
        return

    t0 = time.time()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("PistonModel")

    try:
        # STEP Import
        print(f"📥 Importing STEP file: {step_file}")
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()

        # Enhanced healing geometry
        print("🩹 Healing geometry...")
        gmsh.model.occ.healShapes()
        gmsh.model.occ.synchronize()

        # Remove duplicates and fix topology
        print("🔧 Removing duplicates and fixing topology...")
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # Set mesh parameters for better quality
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        # More robust mesh size setting
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)

        # Set global mesh size
        points = gmsh.model.getEntities(0)
        gmsh.model.mesh.setSize(points, mesh_size)

        # Get all surfaces and volumes
        surfaces = gmsh.model.getEntities(2)
        volumes = gmsh.model.getEntities(3)

        print(f"🔖 Found {len(surfaces)} surfaces")
        print(f"🧊 Found {len(volumes)} volumes")

        # Check if we have existing volumes
        if len(volumes) > 0:
            print("✅ Using existing volumes from STEP file")
            # Use the first volume as the main piston volume
            volume_tag = volumes[0][1]
            gmsh.model.addPhysicalGroup(3, [volume_tag], name="PISTON_VOLUME")

            # Get surfaces that bound this volume
            surface_tags = gmsh.model.getBoundary([(3, volume_tag)], combined=False, oriented=False)
            boundary_surface_tags = [abs(tag[1]) for tag in surface_tags]

        else:
            print("🔄 Creating volume from surfaces...")
            # Try to create volume from all surfaces
            try:
                # Method 1: Try to create surface loop from all surfaces
                all_surface_tags = [surf[1] for surf in surfaces]
                surf_loop = gmsh.model.occ.addSurfaceLoop(all_surface_tags)
                volume_tag = gmsh.model.occ.addVolume([surf_loop])
                gmsh.model.occ.synchronize()

                gmsh.model.addPhysicalGroup(3, [volume_tag], name="PISTON_VOLUME")
                boundary_surface_tags = all_surface_tags

            except Exception as e:
                print(f"⚠️ Method 1 failed: {e}")
                # Method 2: Try with fewer surfaces (largest ones)
                try:
                    # Calculate surface areas and select largest ones
                    surface_areas = []
                    for surf in surfaces:
                        area = gmsh.model.occ.getMass(2, surf[1])
                        surface_areas.append((surf[1], area))

                    # Sort by area and take largest surfaces
                    surface_areas.sort(key=lambda x: x[1], reverse=True)
                    selected_tags = [tag for tag, area in surface_areas[:min(len(surface_areas), 10)]]

                    surf_loop = gmsh.model.occ.addSurfaceLoop(selected_tags)
                    volume_tag = gmsh.model.occ.addVolume([surf_loop])
                    gmsh.model.occ.synchronize()

                    gmsh.model.addPhysicalGroup(3, [volume_tag], name="PISTON_VOLUME")
                    boundary_surface_tags = selected_tags

                except Exception as e2:
                    print(f"❌ Volume creation failed: {e2}")
                    return

        # Create surface groups with better naming
        print("🏷️ Creating surface groups...")
        surface_groups = {}

        # Try to identify surfaces by geometry characteristics
        for i, surf_tag in enumerate(boundary_surface_tags[:10]):  # Limit to first 10
            # Get surface properties
            try:
                center = gmsh.model.occ.getCenterOfMass(2, surf_tag)
                area = gmsh.model.occ.getMass(2, surf_tag)

                # Create meaningful names based on position and area
                if i == 0:
                    name = "TOP_SURFACE"
                elif i == 1:
                    name = "BOTTOM_SURFACE"
                elif i == 2:
                    name = "SIDE_SURFACE"
                else:
                    name = f"SURFACE_{i + 1}"

                gmsh.model.addPhysicalGroup(2, [surf_tag], name=name)
                surface_groups[name] = surf_tag

            except Exception as e:
                print(f"⚠️ Warning: Could not process surface {surf_tag}: {e}")
                continue

        # Enhanced 2D meshing with better algorithms
        print("🔺 Generating 2D triangular surface mesh...")
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 3)
        gmsh.option.setNumber("Mesh.SmoothNormals", 1)

        # Generate 2D mesh
        gmsh.model.mesh.generate(2)

        # Check for mesh quality issues
        print("🔍 Checking 2D mesh quality...")
        try:
            # Get mesh statistics
            nodes = gmsh.model.mesh.getNodes()
            print(f"   Nodes: {len(nodes[0])}")

            elements = gmsh.model.mesh.getElements(2)
            total_elements = sum(len(elem) for elem in elements[1])
            print(f"   2D Elements: {total_elements}")

        except Exception as e:
            print(f"⚠️ 2D mesh check warning: {e}")

        # Enhanced 3D meshing
        print("🧊 Generating 3D tetrahedral mesh...")
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)

        # Try 3D mesh generation with error handling
        try:
            gmsh.model.mesh.generate(3)
            print("✅ 3D mesh generation successful")

        except Exception as e:
            print(f"❌ 3D mesh generation failed: {e}")
            print("🔄 Trying with relaxed parameters...")

            # Try with relaxed parameters
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 5.0)

            try:
                gmsh.model.mesh.generate(3)
                print("✅ 3D mesh generation successful with relaxed parameters")
            except Exception as e2:
                print(f"❌ 3D mesh generation failed even with relaxed parameters: {e2}")
                return

        # Check 3D mesh
        print("🔍 Checking 3D mesh...")
        try:
            nodes_3d = gmsh.model.mesh.getNodes()
            elements_3d = gmsh.model.mesh.getElements(3)

            total_3d_elements = sum(len(elem) for elem in elements_3d[1])
            print(f"   3D Elements: {total_3d_elements}")

            if total_3d_elements == 0:
                print("❌ No 3D elements generated!")
                return

        except Exception as e:
            print(f"❌ 3D mesh check failed: {e}")
            return

        # Optimize mesh
        print("🔄 Optimizing mesh...")
        try:
            gmsh.model.mesh.optimize("Netgen")
            gmsh.model.mesh.renumberNodes()
            gmsh.model.mesh.renumberElements()
        except Exception as e:
            print(f"⚠️ Mesh optimization warning: {e}")

        # Export mesh
        temp_msh = "temp_piston.msh"
        gmsh.write(temp_msh)
        print("📦 Reading into meshio...")

        mesh = meshio.read(temp_msh)

        # Filter for tetrahedral elements
        tetra_blocks = [cb for cb in mesh.cells if cb.type == "tetra"]

        if not tetra_blocks:
            print("❌ No tetrahedral elements found!")
            return

        # Add NSETs using field_data (from physical groups)
        field_data = {}
        try:
            for dim, tag in gmsh.model.getPhysicalGroups():
                name = gmsh.model.getPhysicalName(dim, tag)
                field_data[name] = [tag, dim]
        except Exception as e:
            print(f"⚠️ Warning: Could not get physical groups: {e}")

        mesh_out = meshio.Mesh(
            points=mesh.points,
            cells=tetra_blocks,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            field_data=field_data
        )

        # Write INP file (Abaqus)
        print(f"💾 Writing INP: {output_inp}")
        meshio.write(output_inp, mesh_out, file_format="abaqus")

        print("\n✅ MESHING COMPLETE")
        print("=" * 60)
        print(f"Nodes: {len(mesh_out.points)}")
        print(f"Tetrahedrons: {len(tetra_blocks[0].data)}")
        print("NSETS:")
        for name in surface_groups.keys():
            print(f"  - {name}")
        print(f"Output File: {output_inp}")
        print(f"⏱ Time: {time.time() - t0:.2f}s")

        if os.path.exists(temp_msh):
            os.remove(temp_msh)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    piston_volume_mesh("../../../../../Docker_mesh/pythonProject/PistonMeshing/Piston_T.stp", mesh_size=1.0)