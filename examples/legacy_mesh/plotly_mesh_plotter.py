# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:12:43 2024

@author: mgrei
"""

from sphere import Sphere
from cube import Cube
import numpy as np
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

def plot_meshes(meshes: list, show_vertices=False):
    """Plot multiple 3D meshes using Plotly in the same figure."""
    fig = go.Figure()

    for mesh in meshes:
        # Extract vertices
        x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]

        if show_vertices:
            # Create trace for vertices
            vertex_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=3, color=mesh.color),
                name=f'vertices_{id(mesh)}'
            )
            fig.add_trace(vertex_trace)

        # Create trace for edges
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in mesh.edges:
            start, end = edge
            edge_x += [mesh.vertices[start, 0], mesh.vertices[end, 0], None]
            edge_y += [mesh.vertices[start, 1], mesh.vertices[end, 1], None]
            edge_z += [mesh.vertices[start, 2], mesh.vertices[end, 2], None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=2, color=mesh.color),
            name=f'edges_{id(mesh)}'
        )
        fig.add_trace(edge_trace)

    # Set up layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.show("browser")

# Usage example
sphere = Sphere(radius=1, subdivisions=10)
cube = Cube(side_length=2)


# Apply the combined transformation
cube.rotate(25, "x")
cube.translate([2,2,1])


# Plot both the sphere and the cube in the same figure
plot_meshes([sphere, cube])
