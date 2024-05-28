#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

import tensorflow as tf
import numpy as np

import robot_commands
import memory_buffer
import config
import angle_tools


class ControllNode:
    def __init__(self, model_path):
        self.actions = 0
        self.max_actions = -1
        self.q_net = tf.keras.models.load_model(model_path)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.goal_subscriber = rospy.Subscriber(
            '/goal_points', 
            Float32MultiArray, 
            self.goal_update,
            queue_size=5
        )
        self.goal = None
        self.done = True
        self.speed_buffer = memory_buffer.MemoryBuffer(size=config.lidar_frames)
        for _ in range(self.speed_buffer.size):
            self.speed_buffer.push((0.0, 0.0))


    def calculate_goal_info(self, msg: Odometry) -> list:
        quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        phi = euler_from_quaternion(quat)[-1] # yaw
        pos = [
            msg.pose.pose.position.x, 
            msg.pose.pose.position.y
        ]

        angle_to_goal = angle_tools.angle_from_robot_to_purp(
            robot_pos=pos,
            robot_phi=phi,
            purpose_pos=self.goal
        )
        dist_to_goal = np.linalg.norm(
            np.array(pos)-self.goal
        )
        return [angle_to_goal, dist_to_goal]


    def goal_update(self, msg: Float32MultiArray):
        rospy.loginfo(f'got new goal at {msg.data[0]}, {msg.data[1]}')
        # goal - (x, y, max_actions)
        self.goal = msg.data[:-1]
        self.max_actions = msg.data[-1]
        self.actions = 0
        self.done = False


    def run(self):
        self.state_sub = rospy.Subscriber(
            '/prepared_lidar_buffer', 
            Float32MultiArray, 
            callback=self.q_propagate, 
            queue_size=5
        )
        rospy.spin()

    
    def flatten_buffer(self, buffer: memory_buffer.MemoryBuffer):
        r = []
        for i in buffer.data:
            r += list(i)
        return r


    def q_propagate(self, msg):
        if self.done:
            empty = Twist()
            self.cmd_pub.publish(empty)
            self.speed_buffer.push((0.0, 0.0))
            return
        if self.actions >= self.max_actions:
            self.done = True
            print('not enougth actions')
        if self.done:
            empty = Twist()
            self.cmd_pub.publish(empty)
            self.speed_buffer.push((0.0, 0.0))
            return

        self.actions += 1
        lidar_data = list(msg.data)
        if min(lidar_data) < 0.2:
            action = 1
        else:
            data = lidar_data
            data += self.flatten_buffer(self.speed_buffer)
            odom_msg = rospy.wait_for_message('/odom', Odometry)
            goal_info = self.calculate_goal_info(odom_msg)
            if goal_info[1] < 0.3:
                self.done = True
                print('goal reached!')
            data += goal_info
            rospy.loginfo(f'dist to goal {goal_info[1]}')
            state = np.array(data)
            state = np.reshape(state, (1, state.size, 1))
            action = np.argmax(self.q_net.predict(state, verbose=0))
        new_v, new_w = robot_commands.commands[action](*self.speed_buffer.data[-1])
        self.speed_buffer.push((new_v, new_w))
        cmd = robot_commands.to_twist(new_v, new_w)
        self.cmd_pub.publish(cmd)
        rospy.loginfo(f'''{robot_commands.command_names[action]}:
    old (v, w) - {round(self.speed_buffer.data[-2][0], 3)}, {round(self.speed_buffer.data[-2][1], 3)}
    new (v, w) - {round(new_v, 3)}, {round(new_w, 3)}\n''')
        

rospy.init_node('controll_qnode')
controll_node = ControllNode(
    '/home/misha/diplom_ws/src/my_youbot_world_controll/src/models/14950episode.keras',
    )
controll_node.run()
rospy.spin()



