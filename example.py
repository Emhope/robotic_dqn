import env_objects
import phys_calcs
import shapes
import numpy as np


h = 0.58
w = 0.38
ppm = 200
dt = 0.01
max_v = 0.5
max_w = 0.3
rad = np.random.uniform(1, 2)
period = np.random.uniform(7, 13)

v = 2 * np.pi * rad / period
w = v / rad
v = max(v, max_v)
w = max(v, max_w)

rad = v / w
period = 2 * np.pi * rad / v

env = env_objects.Scene(dt=dt, map_shape=(50, 50), ppm=ppm)

agent = env_objects.SceneObject(
    x=0,
    y=0,
    phi=0,
    v0=0,
    w0=0,
    tv=0.01,
    tw=0.01,
    nv=0.0,
    nw=0.0,
    v=1,
    w=0.5,
    dt=0.01,
    sprite=shapes.create_rect(w=w, h=h),
    name='agent'
)
mov_obs1 = env_objects.SceneObject(
    x=0,
    y=0,
    phi=0,
    v0=0,
    w0=0,
    tv=0.01,
    tw=0.01,
    nv=0.0,
    nw=0.0,
    v=1,
    w=0.5,
    dt=0.01,
    sprite=shapes.create_rect(w=w, h=h)
)
mov_obs2 = env_objects.SceneObject(
    x=0,
    y=0,
    phi=0,
    v0=0,
    w0=0,
    tv=0.01,
    tw=0.01,
    nv=0.0,
    nw=0.0,
    v=1,
    w=0.5,
    dt=0.01,
    sprite=shapes.create_rect(w=w, h=h)
)
mov_obs3 = env_objects.SceneObject(
    x=0,
    y=0,
    phi=0,
    v0=0,
    w0=0,
    tv=0.01,
    tw=0.01,
    nv=0.0,
    nw=0.0,
    v=1,
    w=0.5,
    dt=0.01,
    sprite=shapes.create_rect(w=w, h=h)
)

stat_obs1 = env_objects.SceneObject(
            x=0,
            y=0,
            phi=0,
            v0=0,
            w0=0,
            tv=0.01,
            tw=0.01,
            nv=0.0,
            nw=0.0,
            v=1,
            w=0.5,
            dt=0.01,
            sprite=shapes.create_rect(w=w, h=h)
        )

env.add_object(agent)
env.add_object(mov_obs1)
env.add_object(mov_obs2)
env.add_object(mov_obs3)
