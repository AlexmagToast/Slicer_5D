import numpy as np
from stl import mesh
import numpy.linalg as la
from typing import List, Tuple, Dict
import math
import subprocess
import os
from pathlib import Path
import re


def load_stl_files(file_paths: List[str]) -> List[mesh.Mesh]:
    """
    Loads multiple STL files and returns a list of mesh objects.
    
    Args:
        file_paths: List of paths to STL files
        
    Returns:
        List[mesh.Mesh]: List of loaded mesh objects
    """
    meshes = []
    for file_path in file_paths:
        try:
            mesh_obj = mesh.Mesh.from_file(file_path)
            meshes.append(mesh_obj)
            print(f"Successfully loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    return meshes

def check_vertex_overlap(mesh1: mesh.Mesh, mesh2: mesh.Mesh, tolerance: float = 0.1) -> Tuple[bool, List[np.ndarray]]:
    """
    Checks if vertices of two meshes overlap.
    
    Args:
        mesh1: First mesh object
        mesh2: Second mesh object
        tolerance: Tolerance for overlap detection
        
    Returns:
        Tuple[bool, List[np.ndarray]]: (Overlap found, List of overlapping points)
    """
    # Extract all vertices from both meshes
    vertices1 = mesh1.vectors.reshape(-1, 3)
    vertices2 = mesh2.vectors.reshape(-1, 3)
    
    overlapping_points = []
    
    # Check each vertex of the first mesh
    for v1 in vertices1:
        # Calculate distances to all vertices of the second mesh
        distances = np.linalg.norm(vertices2 - v1, axis=1)
        
        # Find vertices within tolerance
        close_vertices = vertices2[distances < tolerance]
        
        if len(close_vertices) > 0:
            overlapping_points.extend(close_vertices)
    
    return len(overlapping_points) > 0, overlapping_points

def calculate_intersection_plane(points: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Calculates the intersection plane from a list of points.
    Uses the least squares method.
    
    Args:
        points: List of points lying on the plane
        
    Returns:
        Tuple[np.ndarray, float]: (Plane normal vector, Distance from origin)
    """
    if len(points) < 3:
        raise ValueError("At least 3 points required for plane calculation")
    
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    
    # Center the points
    centered_points = points - centroid
    
    # Calculate covariance matrix
    cov_matrix = np.dot(centered_points.T, centered_points)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = la.eigh(cov_matrix)
    
    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal = eigenvectors[:, 0]
    
    # Calculate distance from origin
    d = -np.dot(normal, centroid)
    
    return normal, d

def calculate_rotation_matrix(normal: np.ndarray) -> np.ndarray:
    """
    Calculates the rotation matrix to align the normal vector with the negative Z-axis (downward).
    
    Args:
        normal: Normal vector of the plane
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Normalize the input vector
    normal = normal / np.linalg.norm(normal)
    
    # We want to align with negative Z-axis (downward)
    target_axis = np.array([0, 0, -1])
    
    # Calculate rotation axis (cross product with negative Z-axis)
    rotation_axis = np.cross(normal, target_axis)
    
    if np.allclose(rotation_axis, 0):
        # If normal is already aligned with target axis
        if np.allclose(normal, target_axis):
            return np.eye(3)
        elif np.allclose(normal, -target_axis):
            # If normal is opposite to target axis, we need a 180-degree rotation around X-axis
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    
    # Normalize rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Calculate rotation angle
    cos_angle = np.dot(normal, target_axis)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Create rotation matrix using Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * np.dot(K, K)
    
    # Verify the result
    rotated_normal = np.dot(R, normal)
    if not np.allclose(rotated_normal, target_axis, atol=1e-6):
        # If the rotation didn't achieve the desired result, try the opposite rotation
        R = np.eye(3) + np.sin(-angle) * K + (1 - cos_angle) * np.dot(K, K)
    
    return R

def rotate_mesh(mesh_obj: mesh.Mesh, rotation_matrix: np.ndarray) -> mesh.Mesh:
    """
    Rotates a mesh using the given rotation matrix.
    
    Args:
        mesh_obj: Mesh to rotate
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        mesh.Mesh: Rotated mesh
    """
    # Create a copy of the mesh
    rotated_mesh = mesh.Mesh(mesh_obj.data.copy())
    
    # Apply rotation to all vertices
    rotated_mesh.vectors = np.dot(mesh_obj.vectors, rotation_matrix.T)
    
    return rotated_mesh

def analyze_and_rotate_stls(file_paths: List[str], tolerance: float = 0.1, output_prefix: str = "rotated_") -> Tuple[List[str], Dict[str, Tuple[np.ndarray, float]]]:
    """
    Main function to analyze STL files, detect intersections, and rotate meshes to align with XY plane
    with faces pointing downward.
    
    Args:
        file_paths: List of paths to STL files
        tolerance: Tolerance for overlap detection
        output_prefix: Prefix for output files
        
    Returns:
        Tuple[List[str], Dict[str, Tuple[np.ndarray, float]]]: (List of output files, Dictionary of rotation matrices and distances)
    """
    # Load all STL files
    meshes = load_stl_files(file_paths)
    
    if len(meshes) < 2:
        print("At least 2 STL files are required for analysis.")
        return [], {}
    
    # Store intersection planes and rotation matrices for each mesh
    intersection_planes = {}
    rotation_matrices = {}
    plane_distances = {}
    outputFiles = [file_paths[0]]
    
    # Check all mesh pairs for overlaps
    for i in range(len(meshes)):
        for j in range(i + 1, len(meshes)):
            print(f"\nAnalyzing overlaps between Mesh {i+1} and Mesh {j+1}:")
            
            has_overlap, overlap_points = check_vertex_overlap(meshes[i], meshes[j], tolerance)
            
            if has_overlap:
                print(f"Overlap found between Mesh {i+1} and Mesh {j+1}")
                print(f"Number of overlapping points: {len(overlap_points)}")
                
                try:
                    normal, d = calculate_intersection_plane(overlap_points)
                    print("\nIntersection plane:")
                    print(f"Normal vector: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                    print(f"Distance from origin: {d:.3f}")
                    
                    # Calculate additional information
                    angle_x = math.degrees(math.acos(normal[0]))
                    angle_y = math.degrees(math.acos(normal[1]))
                    angle_z = math.degrees(math.acos(normal[2]))
                    print(f"Angle to X-axis: {angle_x:.1f}°")
                    print(f"Angle to Y-axis: {angle_y:.1f}°")
                    print(f"Angle to Z-axis: {angle_z:.1f}°")
                    
                    # Store the normal vector and distance for the second mesh
                    if j not in intersection_planes:
                        intersection_planes[j] = normal
                        plane_distances[j] = d
                    
                except ValueError as e:
                    print(f"Error in plane calculation: {str(e)}")
            else:
                print(f"No overlap found between Mesh {i+1} and Mesh {j+1}")
    
    # Rotate and save meshes (except the first one)
    for i in range(1, len(meshes)):
        if i in intersection_planes:
            print(f"\nRotating Mesh {i+1} to align with XY plane (face downward)...")
            rotation_matrix = calculate_rotation_matrix(intersection_planes[i])
            rotation_matrices[file_paths[i]] = (rotation_matrix, plane_distances[i])
            rotated_mesh = rotate_mesh(meshes[i], rotation_matrix)
            
            # Verify the rotation
            normal = intersection_planes[i]
            rotated_normal = np.dot(rotation_matrix, normal)
            print(f"Original normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
            print(f"Rotated normal: [{rotated_normal[0]:.3f}, {rotated_normal[1]:.3f}, {rotated_normal[2]:.3f}]")
            
            # Save rotated mesh
            output_file = f"{output_prefix}{file_paths[i]}"
            rotated_mesh.save(output_file)
            print(f"Saved rotated mesh to: {output_file}")
            outputFiles.append(output_file)
        else:
            print(f"\nNo intersection plane found for Mesh {i+1}, skipping rotation.")
    
    return outputFiles, rotation_matrices

def transform_gcode_file(input_file: str, output_file: str, rotation_data: Tuple[np.ndarray, float]):
    """
    Transform all coordinates in a G-code file using the inverse rotation matrix and plane distance.
    
    Args:
        input_file: Path to input G-code file
        output_file: Path to output G-code file
        rotation_data: Tuple of (rotation matrix, plane distance)
    """
    print(f"\nTransforming G-code file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    rotation_matrix, plane_distance = rotation_data
    
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            current_z = 0.0  # Track current Z height
            
            for line in f_in:
                # Only transform G0 and G1 commands
                if line.startswith('G0 ') or line.startswith('G1 '):
                    # Extract current Z if present
                    z_match = re.search(r'Z(-?\d*\.?\d+)', line)
                    if z_match:
                        current_z = float(z_match.group(1))
                    
                    transformed_line = transform_gcode_coordinates(line, rotation_matrix, current_z, plane_distance)
                    f_out.write(transformed_line)
                else:
                    f_out.write(line)
        print(f"Successfully transformed G-code file")
    except Exception as e:
        print(f"Error transforming G-code file: {str(e)}")

def transform_gcode_coordinates(line: str, rotation_matrix: np.ndarray, current_z: float, plane_distance: float) -> str:
    """
    Transform G-code coordinates using the inverse rotation matrix and plane distance.
    
    Args:
        line: G-code line to transform
        rotation_matrix: Rotation matrix to apply
        current_z: Current Z height to maintain
        plane_distance: Distance of the intersection plane from origin
        
    Returns:
        str: Transformed G-code line
    """
    # Regular expression to match G0/G1 commands with coordinates
    coord_pattern = r'([XYZ])(-?\d*\.?\d+)'
    
    # Find all coordinate matches
    matches = re.findall(coord_pattern, line)
    if not matches:
        return line
    
    # Extract coordinates, default to 0 if not present
    coords = np.zeros(3)
    present_axes = set()
    for axis, value in matches:
        idx = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        coords[idx] = float(value)
        present_axes.add(axis)
    
    # If Z is not present in the command, use the current Z height
    if 'Z' not in present_axes:
        coords[2] = current_z
    
    # Apply inverse rotation
    inverse_rotation = np.linalg.inv(rotation_matrix)
    transformed_coords = np.dot(inverse_rotation, coords)
    
    # Add the plane distance to Z coordinate
    transformed_coords[2] += plane_distance
    
    # Build new line with all coordinates
    new_line = line.split(';')[0]  # Remove comments
    new_line = new_line.split()[0]  # Keep only the G0/G1 command
    
    # Add all coordinates, ensuring Z is always included
    for axis in ['X', 'Y', 'Z']:
        idx = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        new_value = transformed_coords[idx]
        new_line += f" {axis}{new_value:.3f}"
    
    # Add back any remaining parameters (like F, E, etc.)
    remaining_params = ' '.join(line.split()[1:])
    if remaining_params:
        # Remove any existing X, Y, Z parameters
        remaining_params = ' '.join([p for p in remaining_params.split() 
                                   if not p.startswith(('X', 'Y', 'Z'))])
        if remaining_params:
            new_line += f" {remaining_params}"
    
    # Add back comments if they existed
    if ';' in line:
        new_line += ';' + line.split(';', 1)[1]
    
    return new_line + '\n'

def merge_gcode_files(gcode_files: List[str], output_file: str):
    """
    Merge multiple G-code files into one, maintaining proper ordering and handling setup/teardown.
    
    Args:
        gcode_files: List of G-code files to merge
        output_file: Path to output merged G-code file
    """
    print(f"\nMerging G-code files into: {output_file}")
    
    try:
        with open(output_file, 'w') as outfile:
            # Write initial setup (from first file)
            with open(gcode_files[0], 'r') as first_file:
                for line in first_file:
                    # Write all lines until the first G1/G0 command
                    if line.startswith('G0 ') or line.startswith('G1 '):
                        break
                    outfile.write(line)
            
            # Process each file
            for i, gcode_file in enumerate(gcode_files):
                print(f"Processing file {i+1}: {gcode_file}")
                
                with open(gcode_file, 'r') as infile:
                    # Skip header for all but first file
                    if i > 0:
                        for line in infile:
                            if line.startswith('G0 ') or line.startswith('G1 '):
                                break
                    
                    # Write all remaining lines
                    for line in infile:
                        outfile.write(line)
                
                # Add a small pause between parts if not the last file
                if i < len(gcode_files) - 1:
                    outfile.write("; Pause between parts\n")
                    outfile.write("G4 P1000\n")  # 1 second pause
            
            # Write final teardown (from last file)
            with open(gcode_files[-1], 'r') as last_file:
                teardown_started = False
                for line in last_file:
                    if line.startswith('M104 S0') or line.startswith('M140 S0'):  # Start of teardown
                        teardown_started = True
                    if teardown_started:
                        outfile.write(line)
        
        print(f"Successfully merged G-code files")
    except Exception as e:
        print(f"Error merging G-code files: {str(e)}")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Example paths to STL files
    stl_files = [
        "1.stl",
        "2.stl"
    ]
    
    # Run analysis and rotation
    rotated_stl_files, rotation_matrices = analyze_and_rotate_stls(stl_files)
    print("Processed STL files:", rotated_stl_files)
    
    # PrusaSlicer configuration
    prusaslicer_path = r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"
    config_bundle = r"config.ini"
    
    # List to store all generated G-code files
    all_gcode_files = []
    
    # Process each STL file
    for stl_file in rotated_stl_files:
        # Get the base name of the STL file without extension
        base_name = Path(stl_file).stem
        
        # Create output G-code filename in the script directory
        output_gcode = script_dir / f"{base_name}.gcode"
        transformed_gcode = script_dir / f"{base_name}_original.gcode"
        
        print(f"\nProcessing {stl_file}...")
        print(f"Output will be saved to: {output_gcode}")
        
        # Run PrusaSlicer
        try:
            consoleCommand = f"{prusaslicer_path} -g --load {config_bundle} {script_dir / stl_file} --output {output_gcode} --dont-arrange --no-ensure-on-bed"
            print(f"issuing command: {consoleCommand}")
            
            subprocess.run(consoleCommand, check=True)
            print(f"Successfully created G-code: {output_gcode}")
            
            # Get original filename by removing prefix if present
            original_filename = stl_file
            if stl_file.startswith("rotated_"):
                original_filename = stl_file[8:]  # Remove "rotated_" prefix
            
            # Transform G-code back to original orientation if rotation was applied
            if original_filename in rotation_matrices:
                print(f"Transforming G-code back to original orientation: {transformed_gcode}")
                transform_gcode_file(output_gcode, transformed_gcode, rotation_matrices[original_filename])
                all_gcode_files.append(str(transformed_gcode))
            else:
                all_gcode_files.append(str(output_gcode))
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {stl_file}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error processing {stl_file}: {str(e)}")
    
    # Merge all G-code files
    if all_gcode_files:
        merged_gcode = script_dir / "merged_output.gcode"
        merge_gcode_files(all_gcode_files, str(merged_gcode))
        print(f"\nAll G-code files have been merged into: {merged_gcode}")

        

