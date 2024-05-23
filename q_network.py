import tensorflow as tf
import numpy as np

import rl_agent
import rl_env
from robot_commands import commands as COMMANDS


def create_q_model1(
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

def create_q_model2(
    lidar_frames_num,
    lidar_num,
    num_actions,
    hidden_dences,
    dropout_rate
    ) -> tf.keras.Model:
    inputs = tf.keras.Input((lidar_frames_num*lidar_num+2+2*(lidar_frames_num), 1))
    f = tf.keras.layers.Flatten()(inputs)
    first_dence = tf.keras.layers.Dense(lidar_frames_num*lidar_num+2+2*(lidar_frames_num), activation='relu')(f)
    drop_first = tf.keras.layers.Dropout(dropout_rate)(first_dence)
    for i in range(hidden_dences):
        if i == 0:
            d = tf.keras.layers.Dense(
                int((lidar_frames_num*lidar_num+2+2*(lidar_frames_num))*0.7), 
                activation='relu'
                )(drop_first)
        else:
            d = tf.keras.layers.Dense(256, activation='relu')(d)
    drop = tf.keras.layers.Dropout(dropout_rate)(d)
    out = tf.keras.layers.Dense(num_actions, activation="linear")(drop)
    m = tf.keras.Model(inputs, out)
    m.compile(
        optimizer="adam",
        loss='mse'
    )
    return m


def validate(net_filename, times=100):
    env = rl_env.RLEnv(
        scene_name='example.json',
        agent_name='agent',
        goal=(6, 0),
        goal_radius=0.3,
        step_rate=2,
        check_collide_rate=10,
        lidar_rate=2,
        lidar_frames=3
    )

    agent = rl_agent.RLAgent(
        batch_size=16,
        mem_size=1,
        lidar_num=env.scene.lidar.points_num,
        lidar_frames=3,
        actions=COMMANDS
    )
    agent.epsilon = 0
    agent.q_net = tf.keras.models.load_model(net_filename)

    success = 0
    for i in range(times):
        state, reward, done, collide = env.step((0, 0))
        done = False
        steps = 0
        while (not done) and steps < 150:
            state = state.reshape((1, state.size, 1))
            act = agent.choose_action(state)
            cmd = COMMANDS[act](env.scene.objects[env.agent_name])
            new_state, reward, done, collide = env.step(cmd)
            new_state += np.random.normal(0, 0.02, (new_state.size,))
            new_state = new_state.reshape((1, new_state.size, 1))
            state = new_state
            steps += 1
        if not collide and steps < 150:
            success += 1
        print(f'validation is {i+1}/{times}, success is {round(success/(i+1)*100, 2)}%', end='\r'.rjust(50))
        
        env.reset()
    print(f'total success is {round(success/times*100, 2)}%'.rjust(50))
    return success / times
