# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:31:28 2024

@author: mgrei
"""
import json
from pathlib import Path
import pyvista as pv
import numpy as np
from ball_flight import BallPathSimulator

data_path = Path(__file__).resolve().parents[1] / "data" / "ball_positions_1.json"
if not data_path.exists():
    data_path = Path(__file__).with_name("ball_positions_1.json")

with data_path.open(encoding="utf-8") as f:
    positions = json.load(f)
    
l_pos_array = np.array([np.array([i["x"], i["y"], float(i["time"])], dtype=np.float64) for i in positions[0]["positions"]], dtype=np.float64)
r_pos_array = np.array([np.array([i["x"], i["y"], float(i["time"])], dtype=np.float64) for i in positions[1]["positions"]], dtype=np.float64)
sim = BallPathSimulator("Ball Path with Spin")

first_time = l_pos_array[0][2]

l_pos_array[:, 2] -= first_time
r_pos_array[:, 2] -= first_time

t_max = 5
t_step = 0.005
args = dict(
    v0 = 22, 
    theta = 0, 
    phi = 2, 
    spin_rate = 1800, 
    spin_elevation = 0,  
    spin_azimuth = 0, 
    x0 = -0.5, 
    y0 = 1.6, 
    z0 = 0
    )
args0 = {**args, "spin_rate": 0}
sim.initialize_simulation(**args)
sim.calculate_path(t_max, t_step)
print(sim.x.shape)
print(sim.y.shape)
print(sim.z.shape)
print(sim.time.shape)
print(sim.x[-1])
print(sim.y[-1])
print(sim.spin_rate)
sim_array = sim.get_path_array()

sim_at_time = []
for row in l_pos_array:
    matching_rows = sim_array[np.isclose(sim_array[:, 3], row[2], atol=1e-2)]
    if matching_rows.shape[0] > 0:
        sim_at_time.append(matching_rows[0])
    else:
        sim_at_time.append([0,0,0,row[2]])
sim_at_time = np.array(sim_at_time)

# Function to set camera parameters based on intrinsic and extrinsic properties
def set_camera(plotter, focal_length_mm, pixel_size_mm, image_width_px, image_height_px,
               camera_position, focal_point, view_up):
    """
    Set the camera parameters based on intrinsic and extrinsic properties.

    Parameters:
    - plotter: PyVista Plotter object.
    - focal_length_mm: Focal length in millimeters.
    - pixel_size_mm: Pixel size in millimeters.
    - image_width_px: Image width in pixels.
    - image_height_px: Image height in pixels.
    - camera_position: Tuple/List of camera (x, y, z) position.
    - focal_point: Tuple/List of camera focal point (x, y, z).
    - view_up: Tuple/List indicating the up direction (x, y, z).
    """

    # Calculate Field of View in degrees
    # Assuming vertical FOV
    fov_y = 2 * np.degrees(np.arctan((image_height_px * pixel_size_mm) / (2 * focal_length_mm)))

    # Set the camera position, focal point, and view up
    plotter.camera_position = (camera_position, focal_point, view_up)

    # Set the camera view angle based on vertical FOV
    plotter.camera.view_angle = fov_y  # PyVista uses vertical FOV by default
    
def get_2d_projection(plotter, point_3d):
    point_3d_homogeneous = np.array([*point_3d, 1.0])  # Convert to homogeneous coordinates
    projected_2d = plotter.camera.model_transform_matrix @ point_3d_homogeneous  # Apply model transform
    projected_2d /= projected_2d[3]  # Normalize by the w component to get 2D screen coordinates
    x_pixel = (projected_2d[0] + 1) * plotter.window_size[0] / 2
    y_pixel = (1 - projected_2d[1]) * plotter.window_size[1] / 2 
    return np.array([x_pixel, y_pixel])

# Create a PyVista plotter object with specified window size
image_width_px = 640   # Image width in pixels
image_height_px = 400   # Image height in pixels
plotter = pv.Plotter(window_size=[image_width_px, image_height_px])
plotter.set_background("lightgray")

distance_to_plate = 12

# Draw the strike zone
center = [0, 0.5, distance_to_plate] 
width, height = 0.44, 0.5  
normal = [0, 0, -1] 
up = [0, 1, 0]
plane = pv.Plane(center=center, direction=normal, i_size=width, j_size=height)
plotter.add_mesh(plane, color="red", opacity=0.5)

# Draw the ground
center = [0, 0, 7] 
width, height = 100, 100  
normal = [0, 1, 0] 
up = [0, 0, -1]
ground = pv.Plane(center=center, direction=normal, i_size=width, j_size=height)
plotter.add_mesh(ground, color="grey", opacity=1)

# Draw the mound
center = [0, 0.1, 0] 
width, height = 0.15, 0.44  
normal = [0, 1, 0] 
up = [0, 0, -1]
mound = pv.Plane(center=center, direction=normal, i_size=width, j_size=height)
plotter.add_mesh(mound, color="white", opacity=1)

# Draw the home plate
center = [0, 0.1, distance_to_plate] 
width, height = 0.44, 0.44  
normal = [0, 1, 0] 
up = [0, 0, -1]
plate = pv.Plane(center=center, direction=normal, i_size=width, j_size=height)
plotter.add_mesh(plate, color="white", opacity=1)

def draw_spheres_along_path(plotter, path):
    # Create the spheres for the ball at various timepoints
    pixel_coords = np.zeros((len(path), 3))
    for idx in range(0, len(path), 1):
        x = path[idx][0]
        y = path[idx][1]
        z = path[idx][2]
        sphere = pv.Sphere(radius=0.037, center=(x, y, z))
        sphere.compute_normals(inplace=True)
        plotter.add_mesh(sphere, color='white', name=f'Sphere {idx}')
        projection = get_2d_projection(plotter, [x,y,z])
        time = path[idx][3]
        pixel_coords[idx] = np.append(projection,time)
    return pixel_coords

pixel_coords = draw_spheres_along_path(plotter, sim_at_time)
# Add custom light sources for better visibility
plotter.add_light(pv.Light(position=(5, -5, 5), focal_point=(0, 0, 0), color='white'))
plotter.add_light(pv.Light(position=(-5, 5, 5), focal_point=(0, 0, 0), color='white'))

# Define camera intrinsic parameters
focal_length_mm = 1.3    # Focal length in mm
pixel_size_mm = 3 / 1000    # Pixel size in mm
image_width_px = 640    # Image width in pixels
image_height_px = 400   # Image height in pixels

# Define camera extrinsic parameters
camera_position = [-2, 0.1, -2]   
focal_point = [sim.x[-1], sim.y[-1] + 2, sim.z[-1]]        
view_up = [0, 1, 0]            

# Apply camera settings
set_camera(plotter, focal_length_mm, pixel_size_mm, image_width_px, image_height_px,
           camera_position, focal_point, view_up)

# Show the plotter window
plotter.show_axes()
plotter.show()


    
