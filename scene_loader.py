import env_objects
import phys_calcs
import shapes
import numpy as np
from matplotlib import pyplot as plt
from types import SimpleNamespace
import json


def load_round_scene(config_name) -> env_objects.Scene:
    '''
    requires json file
    '''
    with open(config_name, 'r') as file:
        config = json.load(file)
        config = SimpleNamespace(**config)
    clockwise = np.random.choice([-1, 1])
    config.w *= clockwise
    config.v += np.random.normal(0, config.vel_noize)
    config.w += np.random.normal(0, config.vel_noize)

    rad = abs(config.v / config.w)

    scene = env_objects.Scene(dt=config.dt, map_shape=(config.box_width+15, config.box_height+15), ppm=config.ppm)

    agent = env_objects.SceneObject(
        # x=-rad*3,
        x=0,
        y=0,
        phi=np.random.uniform(0, np.pi*2),
        v0=0,
        w0=0,
        tv=0.1,
        tw=0.1,
        nv=0.01,
        nw=0.01,
        v=0,
        w=0,
        dt=config.dt,
        sprite=shapes.create_rect(w=config.width, h=config.height, ppm=config.ppm),
        name='agent'
    )
    scene.add_object(agent)

    lidar = env_objects.Lidar(
        parent=agent,
        d=0.3,
        min_angle=-np.pi/2,
        max_angle=np.pi/2,
        max_dist=config.lidar_max_dist,
        points_num=config.lidar_points_num
    )
    scene.add_lidar(lidar)

    delta_angle = np.random.uniform(0, np.pi*2)
    for i in range(config.mov_obstes):
        angle = i * (np.pi * 2 / config.mov_obstes) + delta_angle
        mov_obs = env_objects.SceneObject(
            x=rad*np.cos(angle),
            y=rad*np.sin(angle),
            phi=angle+(np.pi/2)*clockwise,
            v0=config.v,
            w0=config.w,
            tv=0.01,
            tw=0.01,
            nv=0.05,
            nw=0.05,
            v=config.v,
            w=config.w,
            dt=config.dt,
            sprite=shapes.create_rect(w=config.width, h=config.height, ppm=config.ppm),
            name=f'mov_obs{i}'
        )
        scene.add_object(mov_obs)

    # creating box around robot
    config.box_width += np.random.normal(config.box_noize)
    config.box_height += np.random.normal(config.box_noize)
    for x, y in ((-config.box_width/2, 0), (config.box_width/2, 0), (0, -config.box_height/2), (0, config.box_height/2)):
        rect = env_objects.SceneObject(
            x=x,
            y=y,
            phi=0 if x == 0 else np.pi/2,
            v=0,
            w=0,
            tv=0.1,
            tw=0.1,
            nv=0,
            nw=0,
            v0=0,
            w0=0,
            dt=config.dt,
            sprite=shapes.create_rect(h=max(config.box_width, config.box_height), w=min(config.width, config.height)-0.01, ppm=config.ppm, add_arrow=False),
            static=True,
            name=f'stat_obs{x, y}'
        )
        scene.add_object(rect)
    return scene
                