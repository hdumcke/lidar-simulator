import numpy as np
from stl import mesh
import cmath
import copy


class LidarSimulator():

    def __init__(self, env_file, min_range=2.0, max_range=12000.0, resolution=1., error=0.0):
        self.min_range = min_range
        self.max_range = max_range
        self.resolution = resolution  # in degrees
        self.load_env = self.load_environment(env_file)
        self.obstacles = []
        self.error = error

    def load_environment(self, env_file):
        env = mesh.Mesh.from_file(env_file)
        env.x = env.x - env.x.min()
        env.y = env.y - env.y.min()
        return env

    def lidar_scan(self, x, y, yaw):
        triangles = self.get_env_triangles(x, y, yaw)
        s = np.random.normal(1.0, self.error, int(360 / self.resolution))
        return self.lidar_filter(triangles) * s

    def get_lidar_points(self, x, y, yaw, theta=None, view_range=0):
        lidar_scan = self.lidar_scan(x, y, yaw)
        plot_scan = np.stack((np.arange(0, 2 * np.pi, np.radians(self.resolution)), lidar_scan), axis=1)
        plot_scan = plot_scan[plot_scan[:, 1] != np.array(None)]
        if theta is None:
            return plot_scan
        else:
            idx = np.searchsorted(plot_scan[:, 0], theta) - 1
            idx_max = int(idx + view_range / self.resolution)
            idx_min = int(idx - view_range / self.resolution)
            if idx_max <= plot_scan.shape[0] and idx_min >= 0:
                return plot_scan[idx_min:idx_max]
            if idx_min < 0:
                return np.roll(plot_scan, -idx_min, axis=0)[:2 * int(view_range / self.resolution)]
            if idx_max > plot_scan.shape[0]:
                return np.roll(plot_scan, -(idx_max - plot_scan.shape[0]), axis=0)[-2 * int(view_range / self.resolution):]

    def get_map_triangles(self):
        env = self._load_env_with_obstacles()
        subset = env.vectors[(env.normals[:, 0] == 0.0) & (env.normals[:, 1] == 0.0) & (env.normals[:, 2] < 0.0)]
        triangles = []
        for t in subset[:, :, :]:
            triangles.append(t[:, 0:2])
        return np.array(triangles)

    def get_env_triangles(self, x, y, yaw):
        env = copy.deepcopy(self._load_env_with_obstacles())
        env.x = env.x - x
        env.y = env.y - y
        env.rotate([0.0, 0.0, 1.0], yaw)
        subset = env.vectors[(env.normals[:, 0] == 0.0) & (env.normals[:, 1] == 0.0) & (env.normals[:, 2] < 0.0)]
        triangles = []
        for t in subset[:, :, :]:
            a = np.array(self._convert_array(t)).transpose()
            triangles.append(a[a[:, 0].argsort()])
        return np.array(triangles)

    def _cart2polC(self, xyz):
        x, y, z = xyz
        return (cmath.polar(complex(x, y)))  # rho, phi

    def _convert_array(self, arr):
        theta = []
        r = []
        for x in arr:
            rho, phi = self._cart2polC(x)
            theta.append(phi)
            r.append(rho)
        return theta, r

    def _get_distance(self, p1, p2, theta):
        theta_plus = theta
        if (p1[0] > np.pi) or (p2[0] > np.pi):
            # tringles are transformed in filter_triangles
            if theta < 0:
                theta_plus = theta + 2 * np.pi
        p = np.array([p1[0], p2[0]])
        if not (p.min() <= theta_plus) & (theta_plus <= p.max()):
            return self.max_range + 1.0
        r_s_1 = np.sin(p1[0]) * p1[1]
        r_c_1 = np.cos(p1[0]) * p1[1]
        r_s_2 = np.sin(p2[0]) * p2[1]
        r_c_2 = np.cos(p2[0]) * p2[1]
        if (r_c_2 - r_c_1) == 0:
            return r_c_1 / np.cos(theta_plus)
        m = (r_s_2 - r_s_1) / (r_c_2 - r_c_1)
        b = r_s_1 - m * r_c_1
        dist = b / (np.sin(theta_plus) - m * np.cos(theta_plus))
        return dist

    def _filter_triangles(self, triangles, theta):
        np.any(triangles >= 0, axis=1)[:, 0] & np.any(triangles < 0, axis=1)[:, 0]
        special_cases = triangles[np.any(triangles >= 0, axis=1)[:, 0] & np.any(triangles < 0, axis=1)[:, 0]]
        other_cases = triangles[np.invert(np.any(triangles >= 0, axis=1)[:, 0] & np.any(triangles < 0, axis=1)[:, 0])]
        # verticies on both sides
        triangles_hit = other_cases[np.any(other_cases >= theta, axis=1)[:, 0] & np.any(other_cases <= theta, axis=1)[:, 0]]
        # handle special cases:
        sc = []
        for t in special_cases:
            if ((t[:, 0].max() - t[:, 0].min()) < np.pi):
                # not so special after all
                if (theta <= t[:, 0].max()) & (t[:, 0].min() <= theta):
                    sc.append(t)
            else:
                for e in t:
                    if e[0] < 0:
                        e[0] += 2 * np.pi
                if (theta + 2 * np.pi <= t[:, 0].max()) & (t[:, 0].min() <= theta + 2 * np.pi):
                    sc.append(t)
                if (theta <= t[:, 0].max()) & (t[:, 0].min() <= theta):
                    sc.append(t)

        result = []
        for t in triangles_hit:
            result.append(t)
        for t in sc:
            result.append(t)
        return np.array(result)

    def lidar_filter(self, triangles):
        scan = []
        samples = np.arange(-np.pi, np.pi, np.radians(self.resolution))
        for sample in samples:
            #if sample == samples[181]:
            #    import pdb
            #    pdb.set_trace()
            # start with out of range
            dist = self.max_range + 1.0
            # select all triangles hit by the ray
            triangles_hit = self._filter_triangles(triangles, sample)
            for t in triangles_hit:
                dist_t = np.empty(3)
                dist_t[0] = self._get_distance(t[0], t[1], sample)
                dist_t[1] = self._get_distance(t[0], t[2], sample)
                dist_t[2] = self._get_distance(t[1], t[2], sample)
                dist = min(dist_t.min(), dist)
            scan.append(dist)
            if dist > self.max_range:
                scan[-1] = None
            if dist < self.min_range:
                scan[-1] = None
        return np.roll(np.array(scan), int(np.pi / np.radians(self.resolution)))

    def _load_env_with_obstacles(self):
        env = self.load_env
        for cube_mesh in self.obstacles:
            env = mesh.Mesh(np.concatenate([
                            env.data.copy(),
                            cube_mesh.data.copy(),
                            ]))
        return env

    def add_obstacle(self, x, y, theta, length, width):
        # Create 6 faces of a cube
        data = np.zeros(12, dtype=mesh.Mesh.dtype)
        data['vectors'][0] = np.array([[-length / 2, -width / 2, 10.0],
                                       [-length / 2, width / 2, -10.0],
                                       [-length / 2, -width / 2, -10.0]])
        data['vectors'][1] = np.array([[-length / 2, -width / 2, 10.0],
                                       [-length / 2, width / 2, 10.0],
                                       [-length / 2, width / 2, -10.0]])
        data['vectors'][2] = np.array([[length / 2, -width / 2, 10.0],
                                       [length / 2, -width / 2, -10.0],
                                       [length / 2, width / 2, -10.0]])
        data['vectors'][3] = np.array([[length / 2, -width / 2, 10.0],
                                       [length / 2, width / 2, -10.0],
                                       [length / 2, width / 2, 10.0]])
        data['vectors'][4] = np.array([[length / 2, -width / 2, 10.0],
                                       [-length / 2, -width / 2, -10.0],
                                       [length / 2, -width / 2, -10.0]])
        data['vectors'][5] = np.array([[length / 2, -width / 2, 10.0],
                                       [-length / 2, -width / 2, 10.0],
                                       [-length / 2, -width / 2, -10.0]])
        data['vectors'][6] = np.array([[length / 2, width / 2, 10.0],
                                       [length / 2, width / 2, -10.0],
                                       [-length / 2, width / 2, -10.0]])
        data['vectors'][7] = np.array([[length / 2, width / 2, 10.0],
                                       [-length / 2, width / 2, -10.0],
                                       [-length / 2, width / 2, 10.0]])
        data['vectors'][8] = np.array([[length / 2, width / 2, -10.0],
                                       [-length / 2, -width / 2, -10.0],
                                       [-length / 2, width / 2, -10.0]])
        data['vectors'][9] = np.array([[length / 2, width / 2, -10.0],
                                       [length / 2, -width / 2, -10.0],
                                       [-length / 2, -width / 2, -10.0]])
        data['vectors'][10] = np.array([[length / 2, width / 2, 10.0],
                                       [-length / 2, width / 2, 10.0],
                                       [-length / 2, -width / 2, 10.0]])
        data['vectors'][11] = np.array([[length / 2, width / 2, 10.0],
                                       [-length / 2, -width / 2, 10.0],
                                       [length / 2, -width / 2, 10.0]])

        cube_mesh = mesh.Mesh(data, remove_empty_areas=False)

        cube_mesh.rotate([0.0, 0.0, 1.0], theta)
        cube_mesh.translate([x, y, 0.0])
        self.obstacles.append(cube_mesh)

    def delete_obstacles(self):
        self.obstacles = []


if __name__ == "__main__":
    test_lidar = LidarSimulator("robocup.stl")
    #test_lidar.add_obstacle(150, 100, 0, 30, 30)
    point = [100, 100]
    yaw = np.radians(0)
    plot_scan = test_lidar.get_lidar_points(point[0], point[1], yaw)
