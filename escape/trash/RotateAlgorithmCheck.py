import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math

direction = np.array([6,-1,-4])
unit_x = np.array([1,0,0])

direction_xy = (direction - np.array([0,0,direction[2]]))
direction_xy_norm = np.linalg.norm(direction_xy)
if direction_xy_norm != 0:
    direction_xy = direction_xy / direction_xy_norm
else:
    direction_xy = np.zeros(3)

rotation_angle_xy = np.sign(np.cross(direction_xy, unit_x)[2]) * math.acos(direction_xy[0])
rotation_matrix_z = np.array([[math.cos(rotation_angle_xy), -math.sin(rotation_angle_xy), 0],
                              [math.sin(rotation_angle_xy), math.cos(rotation_angle_xy), 0],
                              [0,0,1]
                             ])

rotated_z = np.dot(rotation_matrix_z,direction)
rotated_z = rotated_z / np.linalg.norm(rotated_z)

rotation_angle_xz = np.sign(np.cross(rotated_z, unit_x)[1]) * math.acos(rotated_z[0])
rotation_matrix_y = np.array([[math.cos(rotation_angle_xz), 0,math.sin(rotation_angle_xz)],
                              [0,1,0],
                              [-math.sin(rotation_angle_xz),0, math.cos(rotation_angle_xz)]
                             ])

rotated = np.dot(rotation_matrix_y,np.dot(rotation_matrix_z,direction))
# rotated = np.dot(rotation_matrix_z,direction)
rotated = rotated / np.linalg.norm(rotated)
print(rotated)