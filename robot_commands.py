from env_objects import SceneObject


def _near_zero(v, thresh=0.01):
    return v < thresh and v > -thresh

def same(agent: SceneObject):
    return (agent.v, agent.w)

def stop(agent: SceneObject):
    return (0, 0)

def forward(agent: SceneObject):
    new_v = agent.v + 0.1
    new_w = 0
    if not _near_zero(agent.w):
        new_w = agent.w + (0.05 if agent.w < 0 else -0.05)
    return new_v, new_w

def backward(agent: SceneObject):
    new_v = agent.v - 0.1
    new_w = 0
    if not _near_zero(agent.w):
        new_w = agent.w + (0.05 if agent.w < 0 else -0.05)
    return new_v, new_w

def left(agent: SceneObject):
    new_v = 0
    new_w = agent.w + 0.1
    if not _near_zero(agent.v):
        new_v = agent.v + (0.05 if agent.v < 0 else -0.05)
    return new_v, new_w

def right(agent: SceneObject):
    new_v = 0
    new_w = agent.w - 0.1
    if not _near_zero(agent.v):
        new_v = agent.v + (0.05 if agent.v < 0 else -0.05)
    return new_v, new_w

commands = [same, stop, forward, backward, left, right]
