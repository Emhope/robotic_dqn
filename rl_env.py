import env_objects


class RLEnv:
    def __init__(self, objects_fname, reward):
        '''
        objects_fname - json, that describes all objects on scene: 
            obstacles (static and dynamic);
            scene_object;
            RL agent should be called 'agent' and have the lidar_dist attribute;
            lidar params
        '''
        ...
    
    def step(self, action):
        '''
        return reward, done
        '''
        ...
        