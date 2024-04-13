import numpy as np
import phys_calcs
import scipy
from matplotlib import pyplot as plt
import geom_calcs
import time
import json


class SceneObject:
    def __init__(self, x, y, phi, v, w, tv, tw, nv, nw, v0, w0, dt, sprite, name=None, static=False):
        self.x = x
        self.y = y
        self.phi = phi
        self.v = v0
        self.w = w0
        self.tv = tv
        self.tw = tw
        self.nv = nv
        self.nw = nw
        self.dt = dt
        self.sprite = sprite
        self.name = name
        self.static = static
        self.f = phys_calcs.cmd_vel(
            x0=self.x,
            y0=self.y,
            phi0=self.phi,
            v0=v0,
            w0=w0,
            tv=self.tv,
            tw=self.tw,
            nv=self.nv,
            nw=self.nw,
            v=v,
            w=w,
            dt=self.dt
        )

    def tick(self):
        if not self.static:
            res = next(self.f)
            self.x, self.y, self.phi, self.v, self.w = res
    
    def set_cmd_vel(self, v, w):
        if self.static:
            raise ValueError('static object cannot take cmds')
        self.f = phys_calcs.cmd_vel(
            x0=self.x,
            y0=self.y,
            phi0=self.phi,
            v0=self.v,
            w0=self.w,
            tv=self.tv,
            tw=self.tw,
            nv=self.nv,
            nw=self.nw,
            v=v,
            w=w,
            dt=self.dt
        )
    
    def get_rect_points(self, ppm):
        # half-width, half-height
        hw, hh = [i / ppm / 2 for i in self.sprite.shape]
        orig_rect = np.array([
            [-hh, hh, hh, -hh],
            [-hw, -hw, hw, hw]
        ])
        rot = geom_calcs.rot_matrix(self.phi)
        orig = np.dot(rot, orig_rect)
        d = np.array([[self.x], [self.y]])
        d = np.concatenate((d,)*orig.shape[1], axis=1)
        return (orig + d).T

    def get_rect_lines(self, ppm):
        points = self.get_rect_points(ppm)
        lines = [
            [points[0], points[1]],
            [points[1], points[2]],
            [points[2], points[3]],
            [points[3], points[0]]
        ]
        return np.array(lines)
    
    @property
    def pos(self):
        return [self.x, self.y]
    
    def serialize(self):
        d = dict()
        for k, v in self.__dict__.items():
            if k == 'f':
                continue
            elif k == 'sprite':
                d[k] = v.tolist()
            else:
                d[k] = v
        return d


class Lidar:
    def __init__(self, parent: SceneObject, d, min_angle, max_angle, max_dist, points_num):
        '''
        parent - the object to which the lidar is attached
        d - distance from pos of parent on the direction of parent.phi
        '''
        self.parent = parent
        self.d = d
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.max_dist = max_dist
        self.points_num = points_num
    
    @property
    def x(self):
        return self.parent.x + self.d * np.cos(self.parent.phi)
        
    @property
    def y(self):
        return self.parent.y + self.d * np.sin(self.parent.phi)
    
    def tick(self, frame, cx, cy, ppm):
        return phys_calcs.lidar_simulate(
            bin_map=frame,
            ppm=ppm,
            cx=cx,
            cy=cy,
            x=self.x,
            y=self.y,
            phi=self.parent.phi,
            min_angle=self.min_angle,
            max_angle=self.max_angle,
            points_num=self.points_num,
            max_dist=self.max_dist
        )
        

class Scene:
    def __init__(self, dt, map_shape, ppm):
        '''
        dt - time step of simulation moving objects
        map_shape - (x, y) shpe of simulated map in meters
        ppm - pixels per meter
        '''

        self.dt = dt
        self.t = 0
        self.pix_map_shape = [i*ppm for i in map_shape[::-1]]  # (pix_y, pix_x)
        self.cy, self.cx = [i//2 for i in self.pix_map_shape]  # binmap coords of (0, 0)
        self.ppm = ppm
        self.objects = dict()
        self.lidar = None

    def add_lidar(self, lidar:Lidar):
        if self.lidar is not None:
            raise Warning('lidar is already exist. The old lidar has been replaced with a new one')
        self.lidar = lidar

    def add_object(self, obj:SceneObject):
        self.objects[obj.name] = obj

    
    def tick(self):
        self.t += self.dt
        for obj_name in self.objects:
            if self.objects[obj_name].static:
                continue
            self.objects[obj_name].tick()

    def objects_lines(self):
        obj_lines = np.concatenate(
            tuple(obj.get_rect_lines(self.ppm) for obj in self.objects.values())
        )
        return obj_lines


    def geom_tick_lidar(self, x, y, phi, min_angle, max_angle, points_num, max_dist):
        # deprecated. dont use
        lidar_pos = np.array([x, y])
        angles = np.linspace(phi + min_angle, phi + max_angle, points_num)[::-1]
        endpoints = np.column_stack((x + np.cos(angles) * max_dist, y + np.sin(angles) * max_dist))
        # obj_lines = self.objects_lines()
        obj_lines = []
        for obj in self.objects.values():
            ls = obj.get_rect_lines(self.ppm)
            ps = np.concatenate((ls[0], ls[2]))
            d = ps - np.concatenate(([lidar_pos],)*4)
            m = ps[np.argmin(np.linalg.norm(d, axis=1))]
            for l in ls:
                if m in l:
                    obj_lines.append(l)
        
        dists = np.array([
                [
                np.linalg.norm(geom_calcs.seg_intersect(lidar_pos, ep, p1, p2)-lidar_pos)
                for p1, p2 in obj_lines
                ]   
            for ep in endpoints
                        ])
        
        res = np.min(dists, axis=1)
        res[res==np.inf] = max_dist
        return res
        

    def check_collides(self, robot_name):
        agent_lines = self.objects[robot_name].get_rect_lines(self.ppm)
        infp = np.array([np.inf, np.inf])
        for obj_name in self.objects:
            if obj_name == robot_name:
                continue
            obj_lines = self.objects[obj_name].get_rect_lines(self.ppm)
            for al in agent_lines:
                for ol in obj_lines:
                    p = geom_calcs.seg_intersect(*al, *ol)
                    if not all(p==infp):
                        return True
        return False

    def render(self):
        '''
        render a frame of scene
        '''
        frame = np.zeros(self.pix_map_shape, dtype=bool)
        # controll_sum = 0
        for obj_name, obj in self.objects.items():
            rot_sprite = scipy.ndimage.rotate(obj.sprite, -np.rad2deg(obj.phi), order=0, reshape=True)
            # controll_sum += np.sum(rot_sprite)
            ux = int(obj.x * self.ppm) + self.cx + rot_sprite.shape[1] // 2
            uy = int(obj.y * self.ppm) + self.cy + rot_sprite.shape[0] // 2
            
            lx = ux - rot_sprite.shape[1]
            ly = uy - rot_sprite.shape[0]
            frame[ly:uy, lx:ux] |= rot_sprite
        # return frame, controll_sum == np.sum(frame)
        return frame
    
    def tick_lidar(self):
        f = self.render()
        return self.lidar.tick(f, self.cx, self.cy, self.ppm)
