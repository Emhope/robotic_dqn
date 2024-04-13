import numpy as np
import math


def get_purpose_angle(robot_pos, purpose_pos):
    '''
    calculating angle between x axis and vector from robot to purpose point
    '''
    angle = math.atan2(purpose_pos[1] - robot_pos[1], purpose_pos[0] - robot_pos[0]) * 180 / np.pi
    if angle < 0:
        angle += 360

    return angle


def create_vector(angle):
    return np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) - np.array([0, 0])


def angle_from_robot_to_purp(robot_pos, robot_phi, purpose_pos):
    # anticlockwise is positive

    alhpa_r = robot_phi
    alhpa_p = get_purpose_angle(robot_pos, purpose_pos)
    if abs(alhpa_p - alhpa_r) < 0.1:
        return 0

    vec_r = create_vector(alhpa_r)
    vec_p = create_vector(alhpa_p)

    ans = np.rad2deg(np.arccos(np.dot(vec_r, vec_p)/np.linalg.norm(vec_r)/np.linalg.norm(vec_p)))

    clockwise = True
    if alhpa_p - 0.1 <= (alhpa_r + ans) % 360 <= alhpa_p + 0.1:
        clockwise = False
    
    return np.deg2rad(-ans) if clockwise else np.deg2rad(ans)
    