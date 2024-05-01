import rl_env
import rl_agent
from robot_commands import commands
import env_objects
import shapes

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def use_q_net(net_filename, env_filename, goal, goal_r, times=10):
    env = rl_env.RLEnv(
        scene_name=env_filename,
        agent_name='agent',
        goal=goal,
        goal_radius=goal_r,
        step_rate=2,
        check_collide_rate=10,
        lidar_rate=2,
        lidar_frames=3
    )

    angles = np.linspace(0, np.pi*2, 360)
    cir_x = np.cos(angles) * env.scene.ppm * goal_r + env.scene.cx + goal[0] * env.scene.ppm
    cir_y = np.sin(angles) * env.scene.ppm * goal_r + env.scene.cy + goal[1] * env.scene.ppm
    goal_circle = [cir_x, cir_y]

    agent = rl_agent.RLAgent(
        batch_size=16,
        mem_size=1,
        lidar_num=env.scene.lidar.points_num,
        lidar_frames=3,
        actions=commands
    )
    agent.epsilon = 0
    agent.q_net = tf.keras.models.load_model(net_filename)
    
    for i in range(times):
        # print(i)
        env.reset()
        env.scene.objects[env.agent_name].x = -5
        env.scene.objects.pop('mov_obs0')
        env.scene.objects.pop('mov_obs1')
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
        state, reward, done, collide = env.step((0, 0))
        r = 0
        steps = 0
        while steps < 50 and not done:
            steps += 1
            state = np.reshape(state, (1, state.size, 1))
            act = agent.choose_action(state)
            cmd = commands[act](env.scene.objects[env.agent_name])
            state, reward, done, collide = env.step(cmd)
            r += reward
            print(round(r, 5), end='\r')

            frame = env.scene.render()
            plt.gca().clear()
            plt.gcf().clf()
            plt.imshow(frame)

            plt.scatter(
                goal[0]*env.scene.ppm + env.scene.cx,
                goal[1]*env.scene.ppm + env.scene.cy,
            )
            plt.plot(*goal_circle)

            plt.show(block=False)
            plt.pause(0.0001)

        print()

if __name__ == '__main__':
    use_q_net(
        'learnings/learning_at_wed_may__1_18:49:32_2024/1000episode.keras',
        'example.json',
        goal=(6, 0),
        goal_r=0.3,
        times=100
        )
