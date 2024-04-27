import env_objects
from memory_buffer import MemoryBuffer
import angle_tools
import numpy as np
import scene_loader
from memory_profiler import profile


class RLEnv:
    def __init__(self, scene_name, agent_name: str, goal, goal_radius, step_rate, check_collide_rate, lidar_rate, lidar_frames=3,):
        '''
        goal - (x, y) of goal
        goal_radius - minimum distance to goal to complete the episode successfully
        '''
        self.init_args = [scene_name, agent_name, goal, goal_radius, step_rate, check_collide_rate, lidar_rate, lidar_frames]

        self.scene = scene_loader.load_round_scene(scene_name)
        self.agent_name = agent_name
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.step_period = 1 / step_rate
        self.step_time = 0
        self.collide_period = 1 / check_collide_rate
        self.collide_time = 0
        self.lidar_period = 1 / lidar_rate
        self.lidar_time = 0
        self.phys_period = self.scene.dt
        self.phys_time = 0
        
        self.speed_buffer = MemoryBuffer(size=lidar_frames)
        self.lidar_buffer = MemoryBuffer(size=lidar_frames)
        self.agent_states = MemoryBuffer(size=2)
        self.agent_states.push(np.array(self.scene.objects[self.agent_name].pos))
        while len(self.lidar_buffer.data) < self.lidar_buffer.size:
            self.step((0, 0))
    

    # @profile
    def reset(self):
        self.__init__(*self.init_args)

    @property
    def angle_to_goal(self):
        return angle_tools.angle_from_robot_to_purp(
            robot_pos=self.scene.objects[self.agent_name].pos,
            robot_phi=self.scene.objects[self.agent_name].phi,
            purpose_pos=self.goal
        )
    
    @property
    def dist_to_goal(self):
        return np.linalg.norm(
            np.array(self.scene.objects[self.agent_name].pos) - self.goal
        )

    def _reward(self):
        # print(self.angle_to_goal)
        ra, rd = 0, 0

        ra = abs(self.angle_to_goal) * (-1)
        
        prev_dist = np.linalg.norm(self.goal-self.agent_states.data[0])
        new_dist = np.linalg.norm(self.goal-self.agent_states.data[1])
        if new_dist - prev_dist < -0.05:
            rd = 10 * (prev_dist - new_dist)
        else:
            rd = -0.1
        
        return ra + rd
        

    def _flatten_state(self, state):
        lidar = np.concatenate(state[0])
        vels = np.concatenate(state[1])
        goal_info = np.array(state[2:])
        return np.concatenate((lidar, vels, goal_info))


    # @profile
    def step(self, action):
        '''
        action - (v, w)
        return state, reward, done
        '''
        self.scene.objects[self.agent_name].set_cmd_vel(*action)
        self.step_time += self.step_period

        collide = False
        reach_goal = False
        while self.phys_time < self.step_time:
            self.phys_time += self.phys_period
            self.scene.tick()
            if self.collide_time <= self.phys_time:
                self.collide_time += self.collide_period
                if self.scene.check_collides(self.agent_name):
                    collide = True
                    break
                if np.linalg.norm(np.array(self.scene.objects[self.agent_name].pos)-self.goal) <= self.goal_radius:
                    reach_goal = True
                    break

            if self.lidar_time <= self.phys_time:
                self.lidar_time += self.lidar_period
                l = self.scene.tick_lidar()
                self.lidar_buffer.push(l)
                self.speed_buffer.push((
                    self.scene.objects[self.agent_name].v,
                    self.scene.objects[self.agent_name].w
                ))

        self.agent_states.push(np.array(self.scene.objects[self.agent_name].pos))
        r = self._reward() + 1 * reach_goal - 10 * collide
        # r = self._reward() + 100 * reach_goal - 100 * collide
        done = collide or reach_goal
        flattened_state = self._flatten_state((self.lidar_buffer.data, self.speed_buffer.data, self.angle_to_goal, self.dist_to_goal))
        return (flattened_state, r, done)
