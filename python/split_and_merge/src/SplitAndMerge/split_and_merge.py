import numpy as np
import pickle


class SplitAndMerge():

    def __init__(self, min_seg_length=0.01, line_point_dist_threshold=0.005, min_points_per_segment=20):
        self.params = {}
        self.params['min_seg_length'] = min_seg_length
        self.params['line_point_dist_threshold'] = line_point_dist_threshold
        self.params['min_points_per_segment'] = min_points_per_segment

    def fitLine(self, x, y):
        X = np.average(x)
        Y = np.average(y)
        X_d = (np.array(x) - X)
        Y_d = (np.array(y) - Y)
        X_d_2 = X_d * X_d
        X_d_2_sum = X_d_2.sum()
        spread_x = (np.array(x).max() - np.array(x).min()) / x.shape[0]
        if np.isclose([X_d_2_sum], [0.0])[0] or np.isclose(spread_x, [0.0], atol=1e-03)[0]:
            if X >= 0:
                alpha = 0.0
                r = X
            else:
                alpha = np.pi
                r = -X
            return alpha, r
        m = (X_d * Y_d).sum() / X_d_2.sum()
        b = Y - m * X
        # caclulate perpendicular line
        x_p = -b * m / (m * m + 1)
        if np.isclose([m], [0.0])[0]:
            if Y >= 0:
                alpha = np.pi / 2
                r = Y
            else:
                alpha = 3 * np.pi / 2
                r = -Y
            return alpha, r
        y_p = -x_p / m
        alpha = np.arctan2(y_p, x_p)
        r = np.sqrt(x_p**2 + y_p**2)
        return alpha, r

    def extractLines(self, x, y):
        x = np.array(x)
        y = np.array(y)

        alpha_a, r_a, pointIdx_a = self.splitLinesRecursive(x, y, 0, x.shape[0])

        N = len(r_a)
        if N > 1:
            alpha_a, r_a, pointIdx_a = self.mergeColinearNeigbors(x, y, alpha_a, r_a, pointIdx_a)
            N = len(r_a)

        # Compute endpoints/lengths of the segments
        segend = np.zeros((N, 4))
        seglen = np.zeros((N, 1))
        for i in range(N):
            segend[i, 0] = x[pointIdx_a[i][0]]
            segend[i, 1] = y[pointIdx_a[i][0]]
            segend[i, 2] = x[pointIdx_a[i][1]]
            segend[i, 3] = y[pointIdx_a[i][1]]
            seglen[i] = np.sqrt((segend[i, 0] - segend[i, 2])**2 + (segend[i, 1] - segend[i, 3])**2)

        # Removing short segments
        goodSegIdx = np.logical_and((seglen >= self.params['min_seg_length'])[:, 0],
                                    (np.array(pointIdx_a)[:, 1] - np.array(pointIdx_a)[:, 0]) >= self.params['min_points_per_segment'])
        alpha_a = np.array(alpha_a)[goodSegIdx]
        r_a = np.array(r_a)[goodSegIdx]
        pointIdx_a = np.array(pointIdx_a)[goodSegIdx]
        segend = segend[goodSegIdx]
        seglen = seglen[goodSegIdx]

        return alpha_a, r_a, segend, seglen, pointIdx_a

    def splitLinesRecursive(self, x, y, startIdx, endIdx):
        import pdb
        pdb.set_trace()
        N = endIdx - startIdx
        alpha, r = self.fitLine(x[startIdx:endIdx], y[startIdx:endIdx])
        alpha_a = [alpha]
        r_a = [r]
        idx_a = [[startIdx, endIdx]]
        if N <= 2:
            return alpha_a, r_a, idx_a

        # Find the splitting position (if there is)
        splitPos = self.findSplitPos(x[startIdx:endIdx], y[startIdx:endIdx], alpha, r)

        if splitPos != -1:
            alpha1, r1, idx1 = self.splitLinesRecursive(x, y, startIdx, splitPos + startIdx)
            alpha2, r2, idx2 = self.splitLinesRecursive(x, y, splitPos + startIdx, endIdx)
            alpha_a = alpha1 + alpha2
            r_a = r1 + r2
            idx_a = idx1 + idx2
        else:
            idx_a = [[startIdx, endIdx]]

        return alpha_a, r_a, idx_a

    def findSplitPos(self, x, y, alpha, r):
        d = self.compDistPointsToLine(x, y, alpha, r)
        return self.findSplitPosInD(d)

    def compDistPointsToLine(self, x, y, alpha, r):
        cosA = np.cos(alpha)
        sinA = np.sin(alpha)

        xcosA = x * cosA
        ysinA = y * sinA
        return xcosA + ysinA - r

    def findSplitPosInD(self, d):
        N = d.shape[0] - 1

        farOnPositiveSideB = d > self.params['line_point_dist_threshold']
        farOnNegativeSideB = d < -self.params['line_point_dist_threshold']

        neigborsFarAwayOnTheSameSideI = (farOnPositiveSideB[:-1] & farOnPositiveSideB[1:]) | (farOnNegativeSideB[:-1] & farOnNegativeSideB[1:])

        if not np.any(neigborsFarAwayOnTheSameSideI):
            splitPos = -1
        else:
            pair_sum = np.absolute(d[1:][neigborsFarAwayOnTheSameSideI] + d[:-1][neigborsFarAwayOnTheSameSideI])
            splitPos = np.where(np.absolute(d[1:] + d[:-1]) == pair_sum.max())[0][0]
            if np.absolute(d[splitPos]) <= np.absolute(d[splitPos + 1]):
                splitPos = splitPos + 1

        # If the split position is toward either end of the segment, find otherway to split.
        if splitPos != -1:
            if splitPos < 2:
                splitPos = 2
            if splitPos > N - 2:
                splitPos = N - 2

        return splitPos

    def mergeColinearNeigbors(self, x, y, alpha_a, r_a, pointIdx_a):
        z = [alpha_a[0], r_a[0]]
        startIdx = pointIdx_a[0][0]
        lastEndIdx = pointIdx_a[0][1]

        rOut = []
        alphaOut = []
        pointIdxOut = []

        N = len(r_a)

        for i in range(N - 1):
            endIdx = pointIdx_a[i + 1][1]
            alpha, r = self.fitLine(x[startIdx:endIdx], y[startIdx:endIdx])
            splitPos = self.findSplitPos(x[startIdx:endIdx], y[startIdx:endIdx], alpha, r)
            print("%s %s %s %s %s" % (startIdx, endIdx, splitPos, alpha, r))
            #import pdb
            #pdb.set_trace()
            if splitPos == -1:
                z = [alpha, r]
            else:
                alphaOut.append(z[0])
                rOut.append(z[1])
                pointIdxOut.append([startIdx, lastEndIdx])
                z = [alpha_a[i + 1], r_a[i + 1]]
                startIdx = pointIdx_a[i + 1][0]
            lastEndIdx = endIdx

        alphaOut.append(z[0])
        rOut.append(z[1])
        pointIdxOut.append([startIdx, lastEndIdx])

        return alphaOut, rOut, pointIdxOut


if __name__ == "__main__":

    with open('../../../../jupyternb/data/testLineExtraction3.mat.pickle', 'rb') as f:
        testdata = pickle.load(f)
    # get carthesian coordinates
    x_cart = []
    y_cart = []
    for i in range(testdata['theta'][0].shape[0]):
        r = testdata['rho'][0][i]
        theta = testdata['theta'][0][i]
        x_cart.append(r * np.cos(theta))
        y_cart.append(r * np.sin(theta))

    x_cart = x_cart[230:238]
    y_cart = y_cart[230:238]

    """
    from LidarSim.lidar_sim import LidarSimulator
    lidar = LidarSimulator("../../../../jupyternb/square.stl")
    point = [500, 300]
    yaw = np.radians(0)
    #plot_scan = lidar.get_lidar_points(point[0], point[1], yaw, theta=0, view_range=30)
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

    sam = SplitAndMerge(line_point_dist_threshold=0.004, min_points_per_segment=4, min_seg_length=0.01)
    alpha_a, r_a, segend, seglen, pointIdx_a = sam.extractLines(x_cart, y_cart)
    print("%s" % (pointIdx_a))
