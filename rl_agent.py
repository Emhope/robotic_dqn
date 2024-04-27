import tensorflow as tf
import numpy as np
import memory_buffer
import rl_env
import robot_commands
import env_objects
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
        
        self.q_net = self._create_q_model(
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
    

    def _create_q_model(
        self,
        lidar_frames_num,
        lidar_num,
        num_actions,
        hidden_dences,
        dropout_rate
        ) -> tf.keras.Model:
        inputs = tf.keras.Input((lidar_frames_num*lidar_num+2+2*(lidar_frames_num), 1))
        flatten_pos = tf.keras.layers.Flatten()(inputs[:, -2-2*(lidar_frames_num):, :])
        convs = []
        for i in range(lidar_frames_num):
            n_input = inputs[:, i*lidar_num: (i+1)*lidar_num, :]
            conv_l = tf.keras.layers.Conv1D(
                3,
                (5,),
                activation='relu',
                padding='valid',
                )(n_input)
            conv_drp = tf.keras.layers.Dropout(dropout_rate)(conv_l)
            pool = tf.keras.layers.MaxPool1D()(conv_drp)
            conv_flatten = tf.keras.layers.Flatten()(pool)
            convs.append(conv_flatten)

        merge_convs = tf.keras.layers.concatenate(convs)
        flatten_convs = tf.keras.layers.Flatten()(merge_convs)
        final_merge = tf.keras.layers.concatenate([flatten_convs, flatten_pos])
        for i in range(hidden_dences):
            if i == 0:
                d = tf.keras.layers.Dense(128, activation='relu')(final_merge)
            else:
                d = tf.keras.layers.Dense(128, activation='relu')(d)
        drop = tf.keras.layers.Dropout(dropout_rate)(d)
        out = tf.keras.layers.Dense(num_actions, activation="linear")(drop)
        m = tf.keras.Model(inputs, out)
        m.compile(
            optimizer="adam",
            loss='mse'
        )
        return m

    # @profile
    def _epsilon_greedy(self, state):
        if np.random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.q_net.predict(state, verbose=False)[0])
        return np.random.randint(0, len(self.actions))

    # @profile
    def choose_action(self, state) -> int:
        return self._epsilon_greedy(state)
    
    # @profile
    def train(self):
        minibatch = self.memory.sample(self.batch_size)
        states, targets_f = [], []
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
            epochs=1, 
            batch_size=self.batch_size,
            verbose=0,
            # callbacks=[PlotLossesKeras()]
            )
        return h.history['loss'][0]

    # @profile
    def add_dpoint(self, state, action, reward, next_state, done):
        self.memory.push([state, action, reward, next_state, done])

    # @profile
    def update_target_net(self):
        self.target_net.set_weights(self.q_net.get_weights())
    

    def increase_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
