from geometry_msgs.msg import Twist


MAX_LIN = 0.8
MAX_ANG = 0.8


def crop_to_safe(func):
    def wrapper(v, w):
        new_v, new_w = func(v, w)
        new_v = max(-MAX_LIN, min(new_v, MAX_LIN))
        new_w = max(-MAX_ANG, min(new_w, MAX_ANG))
        return new_v, new_w
    return wrapper


def _near_zero(v, thresh=0.01):
    return v < thresh and v > -thresh


@crop_to_safe
def same(v, w):
    return (v, w)


@crop_to_safe
def stop(v, w):
    return (0, 0)


@crop_to_safe
def forward(v, w):
    new_v = v + 0.1
    new_w = 0
    if not _near_zero(w):
        new_w = w + (0.05 if w < 0 else -0.05)
    return new_v, new_w


@crop_to_safe
def backward(v, w):
    new_v = v - 0.1
    new_w = 0
    if not _near_zero(w):
        new_w = w + (0.05 if w < 0 else -0.05)
    return new_v, new_w


@crop_to_safe
def left(v, w):
    new_v = 0
    new_w = w + 0.1
    if not _near_zero(v):
        new_v = v + (0.05 if v < 0 else -0.05)
    return new_v, new_w


@crop_to_safe
def right(v, w):
    new_v = 0
    new_w = w - 0.1
    if not _near_zero(v):
        new_v = v + (0.05 if v < 0 else -0.05)
    return new_v, new_w


def to_twist(v, w):
    msg = Twist()
    msg.linear.x = v
    msg.angular.z = w
    return msg


commands = [same, stop, forward, backward, left, right]
command_names = ['same', 'stop', 'forward', 'backward', 'left', 'right']
