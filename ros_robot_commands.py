from geometry_msgs.msg import Twist

ROT_DELTA = 0.1
LIN_DELTA = 0.1
ROT_DECREASE = 0.5
LIN_DECREASE = 0.8
MAX_LIN = 1.0
MAX_ANG = 1.0

def same(v, w):
    return (v, w)

def stop(v, w):
    return (0, 0)

def forward(v, w):
    new_v = min(MAX_LIN, v + LIN_DELTA)
    new_w = w * ROT_DECREASE
    return new_v, new_w

def backward(v, w):
    new_v = max(-MAX_LIN, v - LIN_DELTA)
    new_w = w * ROT_DECREASE
    return new_v, new_w

def left(v, w):
    new_v = v * LIN_DECREASE
    new_w = min(MAX_ANG, w + ROT_DELTA)
    return new_v, new_w

def right(v, w):
    new_v = v * LIN_DECREASE
    new_w = max(-MAX_ANG, w - ROT_DELTA)
    return new_v, new_w


def to_twist(v, w):
    msg = Twist()
    msg.linear.x = v
    msg.angular.z = w
    return msg

commands = [same, stop, forward, backward, left, right]
command_names = ['same', 'stop', 'forward', 'backward', 'left', 'right']
