import numpy as np
import phys_calcs
import scipy
from matplotlib import pyplot as plt
import geom_calcs
import time


class SceneObject:
    def __init__(self, x, y, phi, v, w, tv, tw, nv, nw, v0, w0, dt, sprite, name=None, lidar_dist=None):
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
        res = next(self.f)
        self.x, self.y, self.phi, self.v, self.w = res
    
    def set_cmd_vel(self, v, w):
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
        self.current_state = None


    def add_object(self, obj:SceneObject):
        self.objects[obj.name] = obj

    
    def tick(self):
        self.t += self.dt
        for obj_name in self.objects:
            self.objects[obj_name].tick()

    def objects_lines(self):
        obj_lines = np.concatenate(
            tuple(obj.get_rect_lines(self.ppm) for obj in self.objects.values())
        )
        return obj_lines


    def geom_tick_lidar(self, x, y, phi, min_angle, max_angle, points_num, max_dist):
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
                    if any(p==infp):
                        return True
        return False

    def render(self):
        '''
        render a frame of scene
        return frame, and flag of collision
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
