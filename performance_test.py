import phys_calcs
import env

import numpy as np
import time
import random
from matplotlib import pyplot as plt


def test_gleb(bmap, lidar_pos):
    # print('lidar')
    
    res = phys_calcs.gleb_lidar_simulate(
        bmap,
        lidar_pos,
        3
    )

def test_me(bmap=None):
    if bmap is None:
        bmap = np.zeros((50000, 50000), dtype=bool)
    res = phys_calcs.lidar_simulate(
        bmap,
        1000,
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
    for i in range(10):
        o = env.SceneObject(
            x=random.uniform(-2, 2),
            y=random.uniform(-2, 2),
            phi=0,
            v0=0,
            w0=0,
            tv=0.01,
            tw=0.01,
            nv=0.0,
            nw=0.0,
            v=1,
            w=0.5,
            dt=0.02,
            sprite=np.ones((30, 70), dtype=bool)
        )
        
        objects.append(o)

    e = env.Scene(dt=0.02, map_shape=(50, 50), ppm=1000)

    for i in range(len(objects)):
        e.add_object(f'{i}robot', objects[i])

    fig, ax = plt.subplots()
    t = 10
    s = time.perf_counter()
    for i in range(int(t/0.02)):
        # print(f'{i} it')
        # print(round(i*0.03, 3), end='r')
        e.tick()
        # print(e.objects['0robot'].x, e.objects['0robot'].y, e.objects['0robot'].phi)
        if i % 10 == 0:
            # f = e.render()
            # test_gleb(f, np.array((2500, 2500)))
            test_me()
            # ax.imshow(f)
            # plt.show(block=False)
            # plt.pause(0.01)
    print(t / (time.perf_counter() - s))
main()
