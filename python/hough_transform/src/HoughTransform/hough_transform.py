import numpy as np
import pickle


class HoughTransform():

    def __init__(self, angle_step=1, rho_samples=250, acc_threshold=2, acc_r_threshold=20):
        self.rho_samples = rho_samples
        self.r_samples = [50, 150]
        self.acc_threshold = acc_threshold
        self.acc_r_threshold = acc_r_threshold
        self.thetas = np.radians(np.arange(0.0, 180.0, angle_step))
        self.thetas_r = np.radians(np.arange(0, 360))
        self.cos_t = np.cos(self.thetas)
        self.sin_t = np.sin(self.thetas)
        self.cos_t_r = np.cos(self.thetas_r)
        self.sin_t_r = np.sin(self.thetas_r)
        self.num_thetas = len(self.thetas)
        self.num_thetas_r = len(self.thetas_r)

    def extractCircles(self, x, y):
        segend = []
        seglen = []
        pointIdx_a = []
        X = np.array(x)
        X_max = X.max()
        X_min = X.min()
        Y = np.array(y)
        Y_max = Y.max()
        Y_min = Y.min()
        A = set()
        for i in range(len(x)):
            A.add((x[i], y[i]))
        while True:
            B, segend, seglen = self.extractCircle(A, segend, seglen, X_max, X_min, Y_max, Y_min)
            if len(B) == 0:
                break
            for val in B:
                A.remove(val)

        while len(A):
            A, segend, seglen = self.extractLine(A, segend, seglen)

        for i in range(len(segend)):
            if len(segend[i]) == 3:
                continue
            x1 = segend[i][0]
            x2 = segend[i][2]
            pointIdx_a.append([np.where(x == x1)[0][0], np.where(x == x2)[0][0]])

        return segend, seglen, pointIdx_a

    def extractCircle(self, A, segend, seglen, X_max, X_min, Y_max, Y_min):
        R_max = np.array(self.r_samples).max() + 1
        accumulator = np.zeros((len(self.r_samples),
                                int((np.round(X_max) - np.round(X_min)) + 2 * (R_max + 1)),
                                int((np.round(Y_max) - np.round(Y_min)) + 2 * (R_max + 1))),
                               dtype=np.uint8)
        for i in range(len(self.r_samples)):
            r = self.r_samples[i]
            # Creating a Circle Blueprint
            bprint = np.zeros((2 * (r + 1), 2 * (r + 1)), dtype=np.uint8)
            (m, n) = (r + 1, r + 1)                                                   # Finding out the center of the blueprint
            for idx in range(self.num_thetas_r):
                x = int(np.round(r * self.cos_t_r[idx]))
                y = int(np.round(r * self.sin_t_r[idx]))
                bprint[m + x, n + y] = 1
            for val in A:
                # Centering the blueprint circle over the edges
                # and updating the accumulator array
                X = [int(val[0] - X_min) - m + R_max, int(val[0] - X_min) + m + R_max]   # Computing the extreme X values
                Y = [int(val[1] - Y_min) - n + R_max, int(val[1] - Y_min) + n + R_max]   # Computing the extreme Y values
                accumulator[i, X[0]:X[1], Y[0]:Y[1]] += bprint
            accumulator[i][accumulator[i] < self.acc_r_threshold] = 0

        p, a, b = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        if accumulator[p, a, b] == 0:
            return set(), segend, seglen
        segend.append([self.r_samples[p], a + int(X_min) - m, b + int(Y_min) - n])

        B = set()
        for val in A:
            epsilon = 400
            r = (val[0] - segend[-1][1]) ** 2 + (val[1] - segend[-1][2]) ** 2
            if (segend[-1][0]**2 - epsilon <= r) and (r <= segend[-1][0]**2 + epsilon):
                B.add(val)

        seglen.append(len(B))

        return B, segend, seglen

    def extractLines(self, x, y):
        segend = []
        seglen = []
        pointIdx_a = []
        A = set()
        for i in range(len(x)):
            A.add((x[i], y[i]))
        while len(A):
            A, segend, seglen = self.extractLine(A, segend, seglen)

        for i in range(len(segend)):
            x1 = segend[i][0]
            x2 = segend[i][2]
            pointIdx_a.append([np.where(x == x1)[0][0], np.where(x == x2)[0][0]])

        return segend, seglen, pointIdx_a

    def extractLine(self, A, segend, seglen):
        accumulator = np.zeros((self.rho_samples, self.num_thetas), dtype=np.uint8)
        for val in A:
            for idx in range(self.num_thetas):
                rho = val[0] * self.cos_t[idx] + val[1] * self.sin_t[idx]
                # scale and round
                accumulator[int(round(rho * 100.0)), idx] += 1

        # extract line
        hough_max = np.unravel_index(accumulator.argmax(), accumulator.shape)
        if accumulator[hough_max] <= self.acc_threshold:
            A = set()
            return A, segend, seglen

        voting_points_x = []
        voting_points_y = []
        R = set()
        for val in A:
            rho = val[0] * self.cos_t[hough_max[1]] + val[1] * self.sin_t[hough_max[1]]
            if (int(round(rho * 100.0)) == hough_max[0]) or (int(round(rho * 100.0)) + self.rho_samples == hough_max[0]):
                voting_points_x.append(val[0])
                voting_points_y.append(val[1])
                R.add(val)

        voting_points_x = np.array(voting_points_x)
        voting_points_y = np.array(voting_points_y)

        x_min = voting_points_x.min()
        x_max = voting_points_x.max()
        y1 = voting_points_y[np.where(voting_points_x == x_min)[0][0]]
        y2 = voting_points_y[np.where(voting_points_x == x_max)[0][0]]

        segend.append([x_min, y1, x_max, y2])
        seglen.append(len(voting_points_x))

        for val in R:
            A.remove(val)

        return A, segend, seglen


if __name__ == "__main__":

    with open('../../../../jupyternb/data/testLineExtraction4.mat.pickle', 'rb') as f:
        testdata = pickle.load(f)
    # get carthesian coordinates
    x_cart = []
    y_cart = []
    for i in range(testdata['theta'][0].shape[0]):
        r = testdata['rho'][0][i]
        theta = testdata['theta'][0][i]
        x_cart.append(r * np.cos(theta))
        y_cart.append(r * np.sin(theta))

    from LidarSim.lidar_sim import LidarSimulator
    lidar = LidarSimulator("../../../../jupyternb/square.stl")
    point = [500, 300]
    yaw = np.radians(0)
    plot_scan = lidar.get_lidar_points(point[0], point[1], yaw)
    x_cart = []
    y_cart = []
    for alpha, r in plot_scan:
        x_cart.append(r * np.cos(alpha) + point[0])
        y_cart.append(r * np.sin(alpha) + point[1])

    lidar = LidarSimulator("../../../../jupyternb/racetrack.stl")
    point = [900, 50]
    yaw = np.radians(180)
    plot_scan = lidar.get_lidar_points(point[0], point[1], yaw)
    x_cart = []
    y_cart = []
    for alpha, r in plot_scan:
        x_cart.append(r * np.cos(alpha) + point[0])
        y_cart.append(r * np.sin(alpha) + point[1])

    """
    angles = np.arange(0, 85, 5)
    values = [0.5197, 0.4404, 0.4850, 0.4222, 0.4132, 0.4371, 0.3912, 0.3949, 0.3919, 0.4276, 0.4075, 0.3956, 0.4053, 0.4752, 0.5032, 0.5273, 0.4879]
    # get carthesian coordinates
    x_cart = []
    y_cart = []
    for i in range(len(values)):
        r = values[i]
        alpha = np.radians(angles[i])
        x_cart.append(r * np.cos(alpha))
        y_cart.append(r * np.sin(alpha))
    """

    # ht = HoughTransform(rho_samples=200000)
    # segend, seglen, pointIdx_a = ht.extractLines(x_cart, y_cart)
    ht = HoughTransform(rho_samples=200000)
    segend, seglen, pointIdx_a = ht.extractCircles(x_cart, y_cart)
    print("%s" % (segend))
