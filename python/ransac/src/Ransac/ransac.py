import numpy as np
import pickle


class Ransac():

    def __init__(self, num_iterations=20,
                 dist_threshold=0.005,
                 min_points_per_segment=15):
        self.num_iterations = num_iterations
        self.dist_threshold = dist_threshold
        self.min_points_per_segment = min_points_per_segment
        self.rng = np.random.default_rng()

    def extractLines(self, x, y):
        x = np.array(x)
        y = np.array(y)
        segend = []
        seglen = []
        pointIdx_a = []

        A = set()
        for i in range(len(x)):
            A.add((x[i], y[i]))

        while True:
            B, seg, num = self.extractLine(A)
            if len(B) == 0:
                break
            segend.append(seg)
            seglen.append(num)
            pointIdx_a.append([np.where(x == seg[0])[0][0],
                               np.where(x == seg[2])[0][0]])
            for val in B:
                A.remove(val)

            if len(A) < self.min_points_per_segment:
                break

        return np.array(segend), np.array(seglen), np.array(pointIdx_a)

    def extractLine(self, A):
        x = []
        y = []
        num_inliers = []
        all_inliers = []
        points = []

        for val in A:
            x.append(val[0])
            y.append(val[1])

        x = np.array(x)
        y = np.array(y)

        for i in range(self.num_iterations):
            idx = self.rng.choice(len(x), 2)
            while idx[0] == idx[1]:
                idx = self.rng.choice(len(x), 2)
            p1 = [x[idx[0]], y[idx[0]]]
            p2 = [x[idx[1]], y[idx[1]]]
            points.append([p1, p2])
            dist = self.compDistPointsToLine(x, y, p1, p2)
            inliers = dist <= self.dist_threshold
            all_inliers.append(inliers)
            num_inliers.append(inliers.sum())

        num_inliers = np.array(num_inliers)
        idx = np.unravel_index(num_inliers.argmax(), num_inliers.shape)
        inliers = all_inliers[idx[0]]

        B = set()
        x = np.array(x[inliers])
        y = np.array(y[inliers])

        if num_inliers[idx[0]] < self.min_points_per_segment:
            return B, None, None

        for i in range(num_inliers[idx[0]]):
            B.add((x[i], y[i]))

        seg = [x.min(), y[x == x.min()][0], x.max(), y[x == x.max()][0]]

        return B, seg, num_inliers[idx[0]]

    def compDistPointsToLine(self, x, y, p1, p2):
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]
        denom = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        nom = np.absolute((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))

        return nom / denom


if __name__ == "__main__":

    file = '../../../../jupyternb/data/testLineExtraction3.mat.pickle'
    with open(file, 'rb') as f:
        testdata = pickle.load(f)
    # get carthesian coordinates
    x_cart = []
    y_cart = []
    for i in range(testdata['theta'][0].shape[0]):
        r = testdata['rho'][0][i]
        theta = testdata['theta'][0][i]
        x_cart.append(r * np.cos(theta))
        y_cart.append(r * np.sin(theta))

    # x_cart = x_cart[230:238]
    # y_cart = y_cart[230:238]

    """
    from LidarSim.lidar_sim import LidarSimulator
    lidar = LidarSimulator("../../../../jupyternb/square.stl")
    point = [500, 300]
    yaw = np.radians(0)
    #plot_scan = lidar.get_lidar_points(point[0],
    #                                   point[1],
    #                                   yaw,
    #                                   theta=0,
    #                                   view_range=30)
    plot_scan = lidar.get_lidar_points(point[0], point[1], yaw)
    x_cart = []
    y_cart = []
    for alpha, r in plot_scan:
        x_cart.append(r * np.cos(alpha) + point[0])
        y_cart.append(r * np.sin(alpha) + point[1])
    #x_cart = x_cart[211:329]
    #y_cart = y_cart[211:329]
    """

    """
    angles = np.arange(0, 85, 5)
    values = [0.5197, 0.4404, 0.4850, 0.4222, 0.4132, 0.4371, 0.3912,
              0.3949, 0.3919, 0.4276, 0.4075, 0.3956, 0.4053, 0.4752,
              0.5032, 0.5273, 0.4879]
    # get carthesian coordinates
    x_cart = []
    y_cart = []
    for i in range(len(values)):
        r = values[i]
        alpha = np.radians(angles[i])
        x_cart.append(r * np.cos(alpha))
        y_cart.append(r * np.sin(alpha))
    """

    rs = Ransac()
    segend, seglen, pointIdx_a = rs.extractLines(x_cart, y_cart)
    print("%s" % (pointIdx_a))
