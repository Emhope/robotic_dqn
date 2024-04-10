import env_objects


class RLEnv:
    def __init__(self, scene: env_objects.Scene, agent_name: str, goal, goal_radius, step_rate, lidar_rate, lidar_frames=3,):
        '''
        goal - (x, y) of goal
        goal_radius - minimum distance to goal to complete the episode successfully
        '''

        self.scene = scene
        self.agent_name = agent_name
        self.goal = goal
        self.goal_radius = goal_radius
        self.step_period = 1 / step_rate
        self.step_time = 0
        self.lidar_period = 1 / lidar_rate
        self.lidar_time = 0
        self.phys_period = self.scene.dt
        self.phys_time = 0
        
        # TODO add buffer for agent states and lidar frames
        self.lidar_buffer = buffer(size=lidar_frames)
        self.agent_states = buffer(size=2)
        self.agent_states.push(self.scene.objects[self.agent_name].pos)

    def step(self, action):
        '''
        return reward, done
        '''
        self.step_time += self.step_period
        while self.phys_time < self.step_time:
            self.phys_time += self.phys_period
            self.scene.tick()
            if self.lidar_time <= self.phys_time:
                # TODO call lidar
