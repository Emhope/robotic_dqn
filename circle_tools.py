import math
from geometry_msgs.msg import Pose
import config
from gazebo_msgs.msg import ModelState
import numpy as np
import itertools


def _quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = Pose().orientation
    q.w = cy * cp * cr + sy * sp * sr
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    return q


def _create_circle_cyclogramm(r, angle_step, angle_start):
    k = angle_start / (np.pi * 2)
    angles = np.concatenate(
        (
            np.linspace(angle_start, 2*np.pi, int(2*np.pi/angle_step*(1-k)))[:-1],
            np.linspace(0, angle_start, int(2*np.pi/angle_step*k))[1:]
        )
    )
    x, y = np.cos(angles) * r, np.sin(angles) * r
    coords = np.array([x, y, angles]).transpose()
    circle_msgs = []
    for x, y, phi in coords:
        m = ModelState()
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = config.robot_z
        m.pose.orientation = _quaternion_from_euler(0, 0, (phi + np.pi/2) % (np.pi*2))
        circle_msgs.append(m)
    return circle_msgs


def create_circle_cyclogramms(obs_num, r, angle_step):
    return itertools.cycle([
        itertools.cycle(_create_circle_cyclogramm(r, angle_step, angle_start=np.pi*2/obs_num*i))
        for i in range(obs_num)
    ])


