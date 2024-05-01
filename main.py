import rl_env
import rl_agent
from robot_commands import commands as COMMANDS
import shapes
import env_objects

import os
import time
import json
import numpy as np
from matplotlib import pyplot as plt
import sys
from memory_profiler import profile
import tensorflow as tf


dir_name = 'learnings/learning_at_' + time.asctime().replace(' ', '_').lower()
os.makedirs(dir_name)

EPISODES = 6_000
TRAIN_REPEAT = 250
TRAIN_TIMES = 20
EPSILON_UPDATE = 400
MODEL_SAVE_RATE = 50

env = rl_env.RLEnv(
    scene_name='example.json',
    agent_name='agent',
    goal=(6, 0),
    goal_radius=0.2,
    step_rate=2,
    check_collide_rate=10,
    lidar_rate=2,
    lidar_frames=3
)

agent = rl_agent.RLAgent(
    batch_size=16,
    mem_size=500,
    lidar_num=env.scene.lidar.points_num,
    lidar_frames=3,
    actions=COMMANDS
)
agent.epsilon_decay = 0.9992

eps = 0
steps = 0

best_awg_reward = -np.inf
best_model_name = ''

while eps < EPISODES:
    print(f'episode {eps}/{EPISODES}; epsilon - {agent.epsilon}')
    done = False
    for i in range(env.lidar_buffer.size+1):
        state, r, done, collide = env.step((0, 0))
        state = state.reshape((1, state.size, 1))
    
    rewards = []
    losses = []
    step_start = steps
    while (not done) and ((steps - step_start) <= 150):
        act = agent.choose_action(state)
        cmd = COMMANDS[act](env.scene.objects[env.agent_name])
        new_state, reward, done, collide = env.step(cmd)
        new_state = new_state.reshape((1, new_state.size, 1))
        agent.add_dpoint(state, act, reward, new_state, done)
        state = new_state
        rewards.append(reward)
        if steps % TRAIN_REPEAT == 0 and agent.memory.full and steps > TRAIN_REPEAT:
            agent.update_target_net()
            loss = agent.train(epochs=TRAIN_TIMES)
            losses.extend(loss) 
        
        # if steps % EPSILON_UPDATE == 0:
        #     agent.increase_epsilon()

        steps += 1
    agent.increase_epsilon()
    with open(f'{dir_name}/rewards.txt', 'a') as frews, open(f'{dir_name}/losses.txt', 'a') as floss:
        frews.write(' '.join(map(str, rewards))+'\n')
        floss.writelines(map(lambda i: str(i)+'\n', losses))

    if rewards:
        avg_reward = sum(rewards) / len(rewards)
    else:
        avg_reward = -np.inf
    if  avg_reward > best_awg_reward:
        best_awg_reward = avg_reward
        if best_model_name:
            os.remove(best_model_name)
        best_model_name = f'{dir_name}/best_{eps}episode.keras'
        agent.q_net.save(best_model_name)
    if eps % MODEL_SAVE_RATE == 0:
        agent.q_net.save(f'{dir_name}/{eps}episode.keras')

    env.reset()
    if np.random.uniform(0, 1) > 0.5:
        env.scene.objects[env.agent_name].x = np.random.uniform(-4, -6)
        env.scene.objects[env.agent_name].y = np.random.uniform(-1, 1)
    if np.random.uniform(0, 1) > 0.5:
        env.scene.objects.pop('mov_obs0')
    if np.random.uniform(0, 1) > 0.5:
        env.scene.objects.pop('mov_obs1')
    if np.random.uniform(0, 1) > 0.5:
        box = env_objects.SceneObject(
            4, 
            0, 
            0, 
            0, 
            0, 
            1, 
            1, 
            0, 
            0, 
            0, 
            0, 
            0.01,
            shapes.create_rect(np.random.uniform(1, 3), np.random.uniform(0.1, 0.3), env.scene.ppm, False),
            'new_obssss',
            True
        )
        env.scene.add_object(box)
    eps += 1
    tf.keras.backend.clear_session()

agent.q_net.save(f'{dir_name}/{eps}episode.keras')
os.system("shutdown now -h")
