import numpy as np
import phys_calcs
import scipy


class SceneObject:
    def __init__(self, x, y, phi, v, w, tv, tw, nv, nw, v0, w0, dt, sprite, lidar_pos=None):
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
        self.lidar_pos = lidar_pos
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


    def add_object(self, name, obj:SceneObject):
        self.objects[name] = obj

    
    def tick(self):
        self.t += self.dt
        for obj_name in self.objects:
            self.objects[obj_name].tick()


    def render(self):
        frame = np.zeros(self.pix_map_shape, dtype=bool)

        for obj_name, obj in self.objects.items():
            rot_sprite = scipy.ndimage.rotate(obj.sprite, np.deg2rad(obj.phi), order=0, reshape=True)
            
            # obj right upper corner coordinates on frame
            ux = int(obj.x * self.ppm + self.cx) + obj.sprite.shape[1]
            uy = int(obj.y * self.ppm + self.cy) + obj.sprite.shape[0]
            
            # obj left lower corner coordinates on frame
            lx = ux - obj.sprite.shape[1]
            ly = uy - obj.sprite.shape[0]

            frame[ly:uy, lx:ux] |= rot_sprite
        
        return frame
