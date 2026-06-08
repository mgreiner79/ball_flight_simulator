# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:11:03 2024

@author: mgrei
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from mesh3d import Mesh3D

class Sphere(Mesh3D):
    def __init__(self, radius: float, subdivisions: int = 10):
        self.radius = radius
        vertices, edges = self._generate_sphere_mesh(subdivisions)
        super().__init__(vertices=vertices, edges=edges)

    def _generate_sphere_mesh(self, subdivisions: int):
        """Generate vertices and edges for a 3D sphere mesh."""
        phi = np.linspace(0, np.pi, subdivisions)
        theta = np.linspace(0, 2 * np.pi, 2 * subdivisions)
        vertices = []
        
        # Generate vertices
        for p in phi:
            for t in theta:
                x = self.radius * np.sin(p) * np.cos(t)
                y = self.radius * np.sin(p) * np.sin(t)
                z = self.radius * np.cos(p)
                vertices.append([x, y, z])

        vertices = np.array(vertices)

        # Generate edges by connecting neighboring vertices
        edges = []
        for i in range(len(phi) - 1):
            for j in range(len(theta)):
                # Horizontal edges
                next_j = (j + 1) % len(theta)
                edges.append((i * len(theta) + j, i * len(theta) + next_j))
                
                # Vertical edges
                edges.append((i * len(theta) + j, (i + 1) * len(theta) + j))

        return vertices, edges


