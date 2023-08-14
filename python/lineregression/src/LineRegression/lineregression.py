import numpy as np
import pickle


class LineRegression():

    def __init__(self,
                 dist_threshold=0.05,
                 min_points_per_segment=5):
        self.dist_threshold = dist_threshold
        self.min_points_per_segment = min_points_per_segment
        self.rng = np.random.default_rng()

    def extractLines(self, x, y):
        x = np.array(x)
        y = np.array(y)
        segend = []
        seglen = []
        pointIdx_a = []

        return np.array(segend), np.array(seglen), np.array(pointIdx_a)


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

    lr = LineRegression()
    segend, seglen, pointIdx_a = lr.extractLines(x_cart, y_cart)
    print("%s" % ("%s\n%s\n%s\n" % (segend, seglen, pointIdx_a)))
