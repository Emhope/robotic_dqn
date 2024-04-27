import numpy as np
import skimage
from matplotlib import pyplot as plt
import cv2


def cmd_vel(x0, y0, phi0, v0, w0, tv, tw, nv, nw, v, w, dt, max_v=1.0, max_w=1.0):
    '''
    x0, y0, phi0 - initial position
    v0, w0 - initial linear and angular speed
    tv, tw - time constants for linear and angular speed
    v, w - required linear and angular speed
    vn, wn - standard deviations for linear and angular speed
    dt - time step for calculation
    return iterator that calculates position of object at each dt point
    '''
    v = min(v, max_v)
    v0 = min(v0, max_v)
    w0 = min(w0, max_w)
    w = min(w, max_w)

    dv = v - v0
    dw = w - w0
    cur_x, cur_y, cur_phi = x0, y0, phi0
    t = 0
    while True:
        if v == w == v0 == w0 == 0:
            cur_v = 0
            cur_w = 0
        else:
            cur_v = v0 + dv * (1 - np.exp(-t/tv)) + np.random.normal(0, nv)
            cur_w = w0 + dw * (1 - np.exp(-t/tw)) + np.random.normal(0, nw)
            # cur_v = v0 + dv * (1 - np.exp(-t/tv)) + np.random.normal(0, nv) + 0.2 * np.exp(-t/0.5)
            # cur_w = w0 + dw * (1 - np.exp(-t/tw)) + np.random.normal(0, nw) + 0.2 * np.exp(-t/0.5)
        cur_x += dt * cur_v * np.cos(cur_phi)
        cur_y += dt * cur_v * np.sin(cur_phi)
        cur_phi += cur_w * dt
        cur_phi %= np.pi * 2
        t += dt
        yield cur_x, cur_y, cur_phi, cur_v, cur_w


def gleb_lidar_simulate(bmap, lidar_pos, mpd, lidar_range: int):
    '''
    Args:
        mpd: measures per degree
        lidar_range: in pixels
    '''
    lidar_ppr = mpd * 360  # points per revolution
    org = np.array(lidar_pos) - lidar_range
    top = np.array(lidar_pos) + lidar_range
    chunk = bmap[org[0]:top[0], org[1]:top[1]]
    print('0')
    # convert map to 2D decart point cloud
    obs = np.indices(chunk.shape)       # 2 (x, y),  height, width
    obs = obs + (org - lidar_pos).reshape(2, 1, 1) # now obstacles coords are centered at lidar
    print('1')
    # convert to 2D polar point cloud
    pmap = np.zeros(obs.shape, float)       # 2 (R^2, phi), height, width
    pmap[1] = np.arctan2(obs[1], obs[0])    # phi = atan(y, x)
    print('2')
    pmap[1] = (pmap[1] / 2 / np.pi * lidar_ppr).round()  # convert rads to points
    print('3')
    pmap[0] = (obs ** 2).sum(axis=0).astype(float)       # compute R^2
    print('4')
    # extract obstacle points
    ppc = pmap[:, chunk]
    print('6')
    ppc = ppc[:, np.lexsort(ppc)]
    print('7')
    view = np.full(lidar_ppr, np.inf)

    # group by min magic
    index = np.empty(len(ppc[1]), bool)
    index[0] = True
    index[1:] = ppc[1, 1:] != ppc[1, :-1]
    view[ppc[1, index].astype(int)] = np.sqrt(ppc[0, index])

    # treating pixels like zero-sized points results in aliasing
    # alliasing occurs when visible size of a spatial pixel is more than 2 lidar pixels
    # this only affects pixels that are closer than 0.5 * (lidar_ppr / 2pi)
    # to those pixels an antialiasing algorithm has to be applied
    return view

    fine_view = np.copy(view)

    aa_threshhold = lidar_ppr / 2 / np.pi / 2
    aa_targets = np.arange(lidar_ppr)[view < aa_threshhold]
    for phi in aa_targets:
        dist = view[phi]
        shadow = lidar_ppr / 2 / np.pi / dist
        idx = range(round(phi - shadow // 2), round(phi + shadow // 2 + shadow % 2))
        chunk = np.take(fine_view, idx, mode='wrap')
        chunk[chunk > dist] = dist
        np.put(fine_view, idx, chunk, mode='wrap')
    return fine_view


def lidar_simulate(bin_map, ppm, cx, cy, x, y, phi, min_angle, max_angle, points_num, max_dist):
    '''
    bin_map - 2d np array, that representate scene
    ppm - number of pixels in one meter
    cx, cy - pixel coordinates of (0, 0) real point on bin map
    x, y - lidar coordinates in meters
    phi - lidar orientation in radians
    min_angle, max_angle, points_num, max_dist - lidar properties
    return np array cloud of points
    '''

    angles = np.linspace(phi + min_angle, phi + max_angle, points_num)[::-1]
    x_pix, y_pix = int(x * ppm) + cx, int(y * ppm) + cy
    lidar_pos = np.array((y_pix, x_pix))
    max_pix_dist = int(max_dist * ppm)
    endpoints = np.column_stack((x_pix + np.cos(angles) * max_pix_dist, y_pix + np.sin(angles) * max_pix_dist)).astype(int)

    lines = [skimage.draw.line(y_pix, x_pix, *ep[::-1]) for ep in endpoints]
    lines_nonzeros = [np.flatnonzero(bin_map[line]) for line in lines]

    res =  np.array([
        np.linalg.norm(np.array([line[1][max(0, nonz[0]-1)], line[0][max(0, nonz[0]-1)]]) - lidar_pos[::-1])/ppm if nonz.size else max_dist
        for nonz, line in zip(lines_nonzeros, lines)
    ])
    # res =  np.array([
    #     np.linalg.norm(np.array([line[1][nonz[0]], line[0][nonz[0]]]) - lidar_pos[::-1])/ppm if nonz.size else max_dist
    #     for nonz, line in zip(lines_nonzeros, lines)
    # ])
    return res
