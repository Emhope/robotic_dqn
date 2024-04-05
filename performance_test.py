import phys_calcs
import env_objects

import numpy as np
import time
import random
from matplotlib import pyplot as plt
import shapes


def test_gleb(bmap, lidar_pos):
    # print('lidar')
    
    res = phys_calcs.gleb_lidar_simulate(
        bmap,
        lidar_pos,
        3
    )

def test_me(bmap, ppm):
    if bmap is None:
        bmap = np.zeros((50000, 50000), dtype=bool)
    res = phys_calcs.lidar_simulate(
        bmap,
        ppm,
        bmap.shape[1]//2,
        bmap.shape[0]//2,
        0,
        0,
        0,
        -np.pi/2,
        np.pi/2,
        300,
        5
    )

def main():
    objects = []
    ppm = 100
    # for i in range(4):
    #     o = env_objects.SceneObject(
    #         x=np.random.uniform(-10, 10),
    #         y=np.random.uniform(-10, 10),
    #         phi=np.random.uniform(0, np.pi*2),
    #         v0=0,
    #         w0=0,
    #         tv=0.01,
    #         tw=0.01,
    #         nv=0.0,
    #         nw=0.0,
    #         v=np.random.uniform(0.1, 1),
    #         w=np.random.uniform(0.1, 0.5),
    #         dt=0.01,
    #         sprite=np.ones((int(0.3*ppm), int(0.3*ppm)), dtype=bool),
    #         name=f'obj{i}'
    #     )
        
    #     objects.append(o)

    e = env_objects.Scene(dt=0.01, map_shape=(50, 50), ppm=ppm)
    s = shapes.create_rect(0.38, 0.58, ppm)

    r1 = env_objects.SceneObject(
        1,
        1,
        np.pi/2,
        0,
        0,
        0.1,
        0.1,
        0,
        0,
        0,
        0,
        0.02,
        s,
        name='r1'
        # shapes.create_rect(0.3, 0.5, ppm)
    )

    r2 = env_objects.SceneObject(
        -1,
        1,
        np.pi/6,
        0,
        0,
        0.1,
        0.1,
        0,
        0,
        0,
        0,
        0.02,
        s,
        name='r2'
        # shapes.create_rect(0.3, 0.5, ppm)
    )

    r3 = env_objects.SceneObject(
        0,
        1,
        np.pi/15,
        0,
        0,
        0.1,
        0.1,
        0,
        0,
        0,
        0,
        0.02,
        s,
        name='r3'
        # shapes.create_rect(0.3, 0.5, ppm)
    )

    r4 = env_objects.SceneObject(
        0,
        1,
        np.pi/15,
        0,
        0,
        0.1,
        0.1,
        0,
        0,
        0,
        0,
        0.02,
        s,
        name='r4'
        # shapes.create_rect(0.3, 0.5, ppm)
    )


    e.add_object(r1)
    e.add_object(r2)
    e.add_object(r3)
    e.add_object(r4)
    # for i in range(len(objects)):
    #     e.add_object(objects[i])

    fig, ax = plt.subplots()
    t = 20
    ss = time.perf_counter()
    x, y = [], []
    for i in range(int(t/0.01)):
        e.tick()
        # x.append(e.objects['0robot'].x)
        # y.append(e.objects['0robot'].y)
        if i % 20 == 0:
            s = time.perf_counter()
            f = e.render()
            test_me(f, ppm)
            # e.tick_lidar(0, 0, np.pi/2, -np.pi/2, np.pi/2, 300, 5.6)

            print(time.perf_counter() - s)
            print()
            
    print(t / (time.perf_counter() - ss))

    # plt.plot(x, y)
    # plt.scatter(x, y)
    # plt.gca().set_aspect('equal')
    # plt.show()
main()
