import env_objects
from memory_buffer import MemoryBuffer
import angle_tools
import numpy as np


class RLEnv:
    def __init__(self, scene: env_objects.Scene, agent_name: str, goal, goal_radius, step_rate, lidar_rate, lidar_frames=3,):
        '''
        goal - (x, y) of goal
        goal_radius - minimum distance to goal to complete the episode successfully
        '''

        self.scene = scene
        self.agent_name = agent_name
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.step_period = 1 / step_rate
        self.step_time = 0
        self.lidar_period = 1 / lidar_rate
        self.lidar_time = 0
        self.phys_period = self.scene.dt
        self.phys_time = 0
        
        self.lidar_buffer = MemoryBuffer(size=lidar_frames)
        self.agent_states = MemoryBuffer(size=2)
        self.agent_states.push(np.array(self.scene.objects[self.agent_name].pos))

    @property
    def angle_to_goal(self):
        return angle_tools.angle_from_robot_to_purp(
            robot_pos=self.scene.objects[self.agent_name].pos,
            robot_phi=self.scene.objects[self.agent_name].phi,
            purpose_pos=self.goal
        )

    def _reward(self):
        ra, rc, rd, rg = 0, 0, 0, 0
        done = False

        # ra = abs(self.angle_to_goal) * (-1)
        
        rc = self.scene.check_collides(self.agent_name) * (-100)
        if rc:
            done = True
        
        prev_dist = np.linalg.norm(self.goal-self.agent_states.data[0])
        new_dist = np.linalg.norm(self.goal-self.agent_states.data[1])
        if new_dist < prev_dist:
            rd = 10 * (prev_dist - new_dist)
        else:
            rd = -0.1
        
        if new_dist < self.goal_radius:
            rg = 100
            done = True
        
        return ra + rc + rd + rg, done
        

    def step(self, action):
        '''
        action - (v, w)
        return reward, done
        '''
        self.scene.objects[self.agent_name].set_cmd_vel(*action)
        self.step_time += self.step_period
        while self.phys_time < self.step_time:
            self.phys_time += self.phys_period
            self.scene.tick()
            if self.lidar_time <= self.phys_time:
                self.lidar_time += self.lidar_period
                l = self.scene.tick_lidar()
                self.lidar_buffer.push(l)

        self.agent_states.push(np.array(self.scene.objects[self.agent_name].pos))
        r, done = self._reward()
        return self.lidar_buffer, r, done
