#! /usr/bin/env python3

import numpy as np
from math import isnan

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion

import config
import memory_buffer
import angle_tools


class StateNode:
    def __init__(self):
        self.lidar_buffer = memory_buffer.MemoryBuffer(size=config.lidar_frames)
        self.state_puplisher = rospy.Publisher('/prepared_lidar_buffer', Float32MultiArray, queue_size=5)
        self.rate = rospy.Rate(config.lidar_freq)


    def run(self):
        while not rospy.is_shutdown():
            self.refresh_state()
            self.pub_state()
            self.rate.sleep()

    
    def flatten_buffer(self, buffer: memory_buffer.MemoryBuffer):
        r = []
        for i in buffer.data:
            r += i
        return r


    def compress_lidar(self, lidar_ranges, size):
        orig_step = 1 / len(lidar_ranges)
        compress_step = 1 / size
        orig_time, compress_time = 0, 0
        res = []
        for i in lidar_ranges:
            if orig_time >= compress_time:
                compress_time += compress_step
                res.append(i)
            orig_time += orig_step
        return res
    

    def crop_lidar(self, angle_min, angle_max, lidar_msg: LaserScan):
        l = (angle_min - lidar_msg.angle_min) // lidar_msg.angle_increment
        l = int(l)
        if (angle_min - lidar_msg.angle_min) % lidar_msg.angle_increment > lidar_msg.angle_increment / 2:
            l += 1
        return lidar_msg.ranges[l:-l-1]


    def prepare_lidar(self, msg: LaserScan) -> list:
        cropped = self.crop_lidar(config.lidar_min_angle, config.lidar_max_angle, msg)
        compressed = self.compress_lidar(cropped, config.lidar_points)
        for i in range(len(compressed)):
            if compressed[i] > 5.5:
                compressed[i] = 5.6
            elif isnan(compressed[i]):
                if i > 0:
                    compressed[i] = compressed[i-1]
                else:
                    compressed[i] = 0.1
        return compressed
    

    def prepare_odom(self, msg: Odometry) -> list:
        return [msg.twist.twist.linear.x, msg.twist.twist.angular.z]


    def refresh_state(self) -> list:
        lidar_msg : LaserScan = rospy.wait_for_message('/scan', LaserScan)
        odom_msg: Odometry = rospy.wait_for_message('/odom', Odometry)

        lidar_data = self.prepare_lidar(lidar_msg)
        self.lidar_buffer.push(lidar_data)
        

    def pub_state(self):
        data = self.flatten_buffer(self.lidar_buffer)
        msg = Float32MultiArray()
        msg.data = data
        self.state_puplisher.publish(msg)


rospy.init_node('q_state_publisher')
node = StateNode()
rospy.loginfo('q_state node is running')
node.run()
