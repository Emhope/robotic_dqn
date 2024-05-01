import tensorflow as tf
import numpy as np
import memory_buffer
import rl_env
import robot_commands
import env_objects
import q_network
from typing import List, Callable
from memory_profiler import profile


class RLAgent:
    def __init__(
            self, 
            batch_size: int, 
            mem_size:int,
            lidar_num,
            lidar_frames,
            actions: List[Callable]
            ):
        
        self.q_net = q_network.create_q_model1(
            lidar_frames_num=lidar_frames,
            lidar_num=lidar_num,
            num_actions=len(actions),
            hidden_dences=3,
            dropout_rate=0.2
        )
        self.target_net = tf.keras.models.clone_model(self.q_net)
        self.update_target_net()

        self.actions = actions
        self.batch_size = batch_size
        self.memory = memory_buffer.MemoryBuffer(size=mem_size)

        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
    

    # @profile
    def _epsilon_greedy(self, state):
        if np.random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.q_net.predict(state, verbose=False)[0])
        return np.random.randint(0, len(self.actions))

    # @profile
    def choose_action(self, state) -> int:
        return self._epsilon_greedy(state)
    
    # @profile
    def train(self, epochs=1):
        
        states, targets_f = [], []
        for e in range(epochs):
            minibatch = self.memory.sample(self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma *
                            np.amax(self.target_net.predict(next_state, verbose=False)[0]))
                target_f = self.q_net.predict(state, verbose=False)
                target_f[0][action] = target
                # Filtering out states and targets for training
                states.append(state[0])
                targets_f.append(target_f[0])
        
        h = self.q_net.fit(
            np.array(states), 
            np.array(targets_f), 
            epochs=epochs, 
            batch_size=self.batch_size,
            # verbose=0,
            )
        return h.history['loss']

    # @profile
    def add_dpoint(self, state, action, reward, next_state, done):
        self.memory.push([state, action, reward, next_state, done])

    # @profile
    def update_target_net(self):
        self.target_net.set_weights(self.q_net.get_weights())
    

    def increase_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
