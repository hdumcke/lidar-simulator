import numpy as np


def rotate_segend(segend, point, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    j = np.matrix([[c, -s], [s, c]])
    v = np.matrix([[segend[0] - point[0], segend[2] - point[0]],
                   [segend[1] - point[1], segend[3] - point[1]]])
    m = np.dot(j, v)
    p1 = np.squeeze(np.asarray(m.T[0])) + point
    p2 = np.squeeze(np.asarray(m.T[1])) + point
    return [p1[0], p2[0]], [p1[1], p2[1]]


def get_odom(cmd_vel, pose, dt):
    x = pose[0] + cmd_vel["linear_x"] * np.cos(cmd_vel["angular_z"]) * dt
    y = pose[1] + cmd_vel["linear_y"] * np.sin(cmd_vel["angular_z"]) * dt
    return [x, y, pose[2] + cmd_vel["angular_z"]]


if __name__ == "__main__":
    pose = [0.0, 0.0, 0.0]
    cmd_vel = {}
    cmd_vel["linear_x"] = 1.0
    cmd_vel["linear_y"] = 1.0
    cmd_vel["angular_z"] = 0.0
    dt = 0.1
    print(get_odom(cmd_vel, pose, dt))
