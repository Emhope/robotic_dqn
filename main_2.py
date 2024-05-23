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
from q_network import validate

# dir_name = 'learnings/learning_at_' + time.asctime().replace(' ', '_').lower()
# os.makedirs(dir_name)
dir_name = 'learnings/learning_at_tue_may_14_22:43:22_2024'

EPISODES = 20_000
TRAIN_REPEAT = 1000
TRAIN_TIMES = 10
EPSILON_UPDATE = 400
MODEL_SAVE_RATE = 50
VALIDATE_RATE = 500

BATCH_SIZE = 128
MEMORY_SIZE = 2000

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
    # batch_size=32,
    batch_size=BATCH_SIZE,
    # mem_size=1000,
    mem_size=MEMORY_SIZE,
    lidar_num=env.scene.lidar.points_num,
    lidar_frames=3,
    actions=COMMANDS
)
agent.epsilon_decay = 0.9997
# agent.epsilon_delta = 0.00027
agent.epsilon_delta = 0.00005
agent.epsilon = 0.17505000000000276
# agent.epsilon = agent.epsilon_decay ** 201
agent.q_net = tf.keras.models.load_model('learnings/learning_at_tue_may_14_22:43:22_2024/15500episode.keras')
agent.update_target_net()

eps = 15_500
steps = 0

best_awg_reward = -np.inf
best_model_name = ''

while eps < EPISODES:
    # if eps == 4_000:
    #     agent.epsilon_decay = 1
    # if eps == 6_000:
    #     agent.epsilon_decay = 0.9995
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
        #     agent.increase_epsilon_lin()

        steps += 1
    if not agent.memory.full:
        print(f'memory is {len(agent.memory.data)}/{agent.memory.size} full')
        env.reset()
        continue

    agent.increase_epsilon_lin()

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
    if eps % VALIDATE_RATE == 0:
        v = validate(f'{dir_name}/{eps}episode.keras')
        with open(f'{dir_name}/validate.txt', 'a') as val_file:
            val_file.write(str(v)+'\n')

    env.reset()
    eps += 1
    tf.keras.backend.clear_session()

agent.q_net.save(f'{dir_name}/{eps}episode.keras')
# os.system("shutdown now -h")
