from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np

# Create a new plot
figure = pyplot.figure(figsize=(10, 8))
axes = figure.add_subplot(projection='3d')

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('1.stl')

# Create a collection of polygons
collection = mplot3d.art3d.Poly3DCollection(your_mesh.vectors)

# Add some styling
collection.set_alpha(0.8)  # Set transparency
collection.set_facecolor('#1f77b4')  # Set face color
collection.set_edgecolor('#000000')  # Set edge color
collection.set_linewidth(0.5)  # Set edge line width

# Add the collection to the plot
axes.add_collection3d(collection)

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Add labels
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_zlabel('Z')

# Set the title
axes.set_title('3D Mesh Visualization')

# Set the viewing angle
axes.view_init(elev=30, azim=45)

# Add a grid
axes.grid(True)

# Set the background color
axes.set_facecolor('#f0f0f0')
figure.set_facecolor('white')

# Show the plot to the screen
pyplot.show()