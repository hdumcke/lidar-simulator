import numpy as np
import pickle


class Incremental():

    def __init__(self,
                 dist_threshold=0.05,
                 min_points_per_segment=5):
        self.dist_threshold = dist_threshold
        self.min_points_per_segment = min_points_per_segment
        self.rng = np.random.default_rng()

    def fitLine(self, x, y):
        X = np.average(x)
        Y = np.average(y)
        X_d = (np.array(x) - X)
        Y_d = (np.array(y) - Y)
        X_d_2 = X_d * X_d
        m = (X_d * Y_d).sum() / X_d_2.sum()
        b = Y - m * X

        return m, b

    def extractLines(self, x, y):
        x_orig = np.array(x)
        x = np.array(x)
        y = np.array(y)
        segend = []
        seglen = []
        pointIdx_a = []
        while len(x) > 2:
            line_points_list = self.extractLine(x, y)
            idx_s = np.where(x_orig == line_points_list[0][0])[0][0]
            idx_e = np.where(x_orig == line_points_list[-1][0])[0][0]
            pointIdx_a.append([idx_s, idx_e])
            for i in range(len(line_points_list)):
                idx = np.where(x == line_points_list[i][0])[0][0]
                x = np.delete(x, idx)
                y = np.delete(y, idx)
            if len(line_points_list) >= self.min_points_per_segment:
                segend.append([line_points_list[0][0],
                               line_points_list[0][1],
                               line_points_list[-1][0],
                               line_points_list[-1][1]])
                seglen.append(len(line_points_list))

        return np.array(segend), np.array(seglen), np.array(pointIdx_a)

    def extractLine(self, x, y):
        line_points_list = []
        line_points_list.append([x[0], y[0]])
        line_points_list.append([x[1], y[1]])
        p1 = line_points_list[0]
        p2 = line_points_list[1]
        for i in range(2, len(x)):
            dist = self.compDistPointsToLine(x[:i + 1], y[:i + 1], p1, p2)
            if np.any(dist > self.dist_threshold):
                break
            else:
                line_points_list.append([x[i], y[i]])
                if i + 1 == len(x):
                    break
                m, b = self.fitLine(x[:i], y[:i])
                p1 = [x[0], m * x[0] + b]
                p2 = [x[i + 1], m * x[i + 1] + b]

        return line_points_list

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

    inc = Incremental()
    segend, seglen, pointIdx_a = inc.extractLines(x_cart, y_cart)
    print("%s" % ("%s\n%s\n%s\n" % (segend, seglen, pointIdx_a)))
