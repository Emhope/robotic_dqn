import numpy as np


lidar_points = 200
lidar_frames = 3
lidar_freq = 2
lidar_min_angle = -np.pi/2
lidar_max_angle = np.pi/2
lidar_angle_step = (lidar_max_angle - lidar_min_angle) / lidar_points

controll_freq = 2

r = 1.2
angle_step = 0.0005
robot_z = 0.083982