import rl_env
import rl_agent
from robot_commands import commands
import env_objects
import shapes

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cv2


def use_q_net(net_filename, env_filename, goal, goal_r, times=10, randomize=False, save=False, show=True):
    env = rl_env.RLEnv(
        scene_name=env_filename,
        agent_name='agent',
        goal=goal,
        goal_radius=goal_r,
        step_rate=2,
        check_collide_rate=10,
        lidar_rate=2,
        lidar_frames=3,
    )

    angles = np.linspace(0, np.pi*2, 360)
    goal_pix = [
        round(goal[0]*env.scene.ppm) + env.scene.cx, 
        round(goal[1]*env.scene.ppm) + env.scene.cy
    ]
    rad_pix = round(goal_r*env.scene.ppm)
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
    frames = []
    success = 0
    for i in range(times):
        # print(i)
        env.reset()
        if randomize:
            if np.random.uniform(0, 1) > 0.5:
                env.scene.objects[env.agent_name].x = np.random.uniform(-4.9, -6)
                env.scene.objects[env.agent_name].y = np.random.uniform(-1, 1)
            if np.random.uniform(0, 1) > 0.5:
                env.scene.objects.pop('mov_obs0')
            if np.random.uniform(0, 1) > 0.5:
                env.scene.objects.pop('mov_obs1')
            if np.random.uniform(0, 1) > 0.5:
                box = env_objects.SceneObject(
                    -4, 
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
                    shapes.create_rect(np.random.uniform(2, 3), np.random.uniform(0.1, 0.3), env.scene.ppm, False),
                    'new_obssss',
                    True
                )
                env.scene.add_object(box)
        state, reward, done, collide = env.step((0, 0))
        r = 0
        steps = 0
        
        while steps < 150 and not done:
            steps += 1
            state = np.reshape(state, (1, state.size, 1))
            act = agent.choose_action(state)
            cmd = commands[act](env.scene.objects[env.agent_name])
            state, reward, done, collide = env.step(cmd)
            r += reward
            # print(round(r, 5), end='\r')
            if show:
                frame = env.scene.render().astype(np.int8) * 255
                frame = cv2.circle(frame, goal_pix, rad_pix, (255,), 10)
                plt.gca().clear()
                plt.gcf().clf()
            if save:
                frames.append([plt.imshow(~frame, cmap='gray', animated=True)])
            
                plt.imshow(~frame, cmap='gray')
                plt.show(block=False)
                plt.show()
                plt.pause(0.0001)


            # plt.scatter(
            #     goal[0]*env.scene.ppm + env.scene.cx,
            #     goal[1]*env.scene.ppm + env.scene.cy,
            # )
            # plt.scatter(*goal_circle, 0.01, color='black')

        if not collide and steps < 150:
            success += 1
        print(f'validation is {i+1}/{times}, success is {round(success/(i+1)*100, 2)}%', end='\r')
        if save:
            ani = animation.ArtistAnimation(plt.gcf(), frames, interval=200, blit=True, repeat_delay=0)
        # plt.show()
            ani.save(f'{i}movie.gif')

        # print()
    print(f'total success is {round(success/(i+1)*100, 2)}%')
    print(f'total success is {round(success/(i+1)*100, 2)}%')
    print(f'total success is {round(success/(i+1)*100, 2)}%')
    print(f'total success is {round(success/(i+1)*100, 2)}%')

if __name__ == '__main__':
    use_q_net(
        'learnings/learning_at_tue_may_14_22:43:22_2024/7500episode.keras',
        'test_example.json',
        goal=(5, 0),
        goal_r=0.3,
        times=500,
        randomize=False,
        save=False,
        show=True
        )
