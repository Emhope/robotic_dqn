import numpy as np


def rot_matrix(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [c, -s],
        [s, c],
    ])

def _perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def _on_line_seg(a1, a2, p):
    eq1 = p[0] >= min(a1[0], a2[0]) and p[0] <= max(a1[0], a2[0])
    eq2 = p[1] >= min(a1[1], a2[1]) and p[1] <= max(a1[1], a2[1])
    return eq1 and eq2

def seg_intersect(a1, a2, b1, b2):
    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    a1 = np.array(a1)
    a2 = np.array(a2)
    b1 = np.array(b1)
    b2 = np.array(b2)

    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = _perp(da)
    denom = np.dot( dap, db)
    if denom == 0:
        return np.array([np.inf, np.inf])
    num = np.dot( dap, dp )
    p =  (num / denom.astype(float))*db + b1

    if _on_line_seg(a1, a2, p) and _on_line_seg(b1, b2, p):
        return p
    return np.array([np.inf, np.inf])