#! /usr/bin/env python3

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel, DeleteModel
import rospy
import itertools
import numpy as np

import config
import circle_tools


def update_obs(msg: ModelStates, pub: rospy.Publisher, cyclogramms):
    obs_indices = [n for n, i in enumerate(msg.name) if i not in ('youbot', 'ground_plane')]
    for obs in obs_indices:
        m = next(next(cyclogramms))
        obs_m = m
        obs_m.model_name = msg.name[obs]        
        pub.publish(obs_m)


cyclogramms = circle_tools.create_circle_cyclogramms(2, config.r, config.angle_step)


rospy.init_node('obs_controll_node')
delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
try:
    delete_model('world')
except:
    ...
spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
with open('/home/misha/.gazebo/models/youbot/model.sdf', 'r') as file:
    model = file.read()

pos1 = Pose()
pos1.position.x = 5
pos2 = Pose()
pos2.position.x = -5

try:
    spawn_model('obs1', model, "", pos1, "world")
    spawn_model('obs2', model, "", pos2, "world")
except:
    ...
m = ModelState()
m.model_name = 'youbot'
m.pose.position.z = config.robot_z
pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
sub = rospy.Subscriber('/gazebo/model_states', ModelStates, callback=lambda m: update_obs(m, pub, cyclogramms))

rospy.spin()
