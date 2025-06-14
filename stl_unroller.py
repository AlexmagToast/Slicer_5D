import numpy as np
from stl import mesh
import math
from mpl_toolkits import mplot3d
from matplotlib import pyplot

def unroll_stl(input_file, output_file, inner_radius, outer_radius):
    """
    Unrolls an STL file by cutting out a cylinder and flattening the resulting ring.
    
    Args:
        input_file (str): Path to input STL file
        output_file (str): Path to output STL file
        inner_radius (float): Radius of the cylinder to cut out
        outer_radius (float): Maximum radius to consider
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(input_file)
    
    # Get the vertices
    vertices = stl_mesh.vectors.reshape(-1, 3)
    
    # Convert to cylindrical coordinates
    x, y, z = vertices.T
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Filter points within the desired radius range
    mask = (r >= inner_radius) & (r <= outer_radius)
    filtered_vertices = vertices[mask]
    
    # Convert filtered points to cylindrical coordinates
    x_filt, y_filt, z_filt = filtered_vertices.T
    r_filt = np.sqrt(x_filt**2 + y_filt**2)
    theta_filt = np.arctan2(y_filt, x_filt)
    
    # Unroll the points
    # Scale the radius to make the unrolled shape more manageable
    scale_factor = 1.0
    x_unrolled = theta_filt * r_filt * scale_factor
    y_unrolled = z_filt
    z_unrolled = r_filt - inner_radius  # Offset by inner radius
    
    # Create new vertices
    unrolled_vertices = np.column_stack((x_unrolled, y_unrolled, z_unrolled))
    
    # Create a new mesh
    # Note: This is a simplified version - you might need to reconstruct the faces
    # based on your specific needs
    new_mesh = mesh.Mesh(np.zeros(len(unrolled_vertices)//3, dtype=mesh.Mesh.dtype))
    new_mesh.vectors = unrolled_vertices.reshape(-1, 3, 3)
    
    # Save the unrolled mesh
    new_mesh.save(output_file)
    
    return new_mesh

def visualize_mesh(mesh_obj, title="Mesh Visualization"):
    """
    Visualizes a mesh object with improved styling.
    
    Args:
        mesh_obj: The mesh object to visualize
        title: Title for the plot
    """
    # Create a new plot
    figure = pyplot.figure(figsize=(10, 8))
    axes = figure.add_subplot(projection='3d')

    # Create a collection of polygons
    collection = mplot3d.art3d.Poly3DCollection(mesh_obj.vectors)

    # Add some styling
    collection.set_alpha(0.8)  # Set transparency
    collection.set_facecolor('#1f77b4')  # Set face color
    collection.set_edgecolor('#000000')  # Set edge color
    collection.set_linewidth(0.5)  # Set edge line width

    # Add the collection to the plot
    axes.add_collection3d(collection)

    # Auto scale to the mesh size
    scale = mesh_obj.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Add labels
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    # Set the title
    axes.set_title(title)

    # Set the viewing angle
    axes.view_init(elev=30, azim=45)

    # Add a grid
    axes.grid(True)

    # Set the background color
    axes.set_facecolor('#f0f0f0')
    figure.set_facecolor('white')

    # Show the plot
    pyplot.show()

if __name__ == "__main__":
    # Example usage
    input_file = "saugadapter.stl"
    output_file = "unrolled.stl"
    inner_radius = 10.0  # Adjust based on your needs
    outer_radius = 20.0  # Adjust based on your needs
    
    # Process the mesh and get the result
    processed_mesh = unroll_stl(input_file, output_file, inner_radius, outer_radius)
    
    # Visualize the processed mesh
    visualize_mesh(processed_mesh, "Unrolled Mesh Visualization") 