import numpy as np
from filterpy.kalman import KalmanFilter

P1 = 10000   # 10000
P2 = 10   # 10
R = 1   # 1
Q = 0.01    # 0.01

class TrackerBase3D:
    def __init__(self, box3d, box2d, other_info, track_id):
        self.initial_pos_box3d = box3d  # bbox3D: [x, y, z, h, w, l, ry, score]
        self.initial_pos_box2d = box2d  # bbox2D: [xx1, yy1, xx2, yy2]
        self.time_since_update = 0
        self.id = track_id
        self.hits = 1  # number of total hits including the first detection
        self.info = other_info  # other information associated, [seq, frame, cls, alpha, xx1, yy1, xx2, yy2]
        self.is_confirmed = False

    def add_hits(self):
        self.hits += 1

    def add_age(self):
        self.time_since_update += 1

    def clear_age(self):
        self.time_since_update = 0


class TrackerBase2D:
    def __init__(self, box2d, other_info, track_id):
        self.initial_pos_box2d = box2d  # bbox2D: [xx1, yy1, xx2, yy2]
        self.time_since_update = 0
        self.id = track_id
        self.hits = 1  # number of total hits including the first detection
        self.info = other_info  # other information associated, [seq, frame, cls, alpha, xx1, yy1, xx2, yy2]
        self.is_confirmed = False

    def add_hits(self):
        self.hits += 1

    def add_age(self):
        self.time_since_update += 1

    def clear_age(self):
        self.time_since_update = 0


class Tracker3D(TrackerBase3D):
    def __init__(self, box3d, box2d, other_info, track_id, min_hits):
        super().__init__(box3d, box2d, other_info, track_id)

        ######################################################
        # box3d
        ######################################################
        self.kf_box3d = KalmanFilter(dim_x=10, dim_z=7)
        # There is no need to use EKF here as the measurement and state are in the same space with linear relationship

        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
        # while all others (theta, l, w, h, dx, dy, dz) remain the same
        self.kf_box3d.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix, dim_x * dim_x
                                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        self.kf_box3d.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
        self.kf_box3d.R *= R

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.kf_box3d.P[7:, 7:] *= P1
        self.kf_box3d.P[:7, :7] *= P2

        # process uncertainty, make the constant velocity part more certain
        self.kf_box3d.Q[7:, 7:] *= Q

        # initialize data
        self.kf_box3d.x[:7] = self.initial_pos_box3d.reshape((7, 1))

        ######################################################
        # box2d
        ######################################################
        # define constant velocity model  定义匀速模型
        self.kf_box2d = KalmanFilter(dim_x=7, dim_z=4)  # 状态变量是7维，观测值是4维的，按照需要的维度构建目标
        self.kf_box2d.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 0, 1],
                                    [0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1]])
        self.kf_box2d.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0]])

        self.kf_box2d.R *= R  # measurement uncertainty
        self.kf_box2d.P[4:, 4:] *= P1  # give high uncertainty to the unobservable initial velocities 对未观测到的初始速度给出高的不确定性
        self.kf_box2d.P[:4, :4] *= P2  # 默认定义的协方差矩阵是np.eye(dim_x)，将P中的数值与10，1000相乘，赋值不确定性
        self.kf_box2d.Q[4:, 4:] *= Q

        self.kf_box2d.x[:4] = self.initial_pos_box2d

        self.min_hits = min_hits

    def update(self, box3d, box2d, other_info):
        """
        Updates the state vector with observed bounding box.
        """
        # update statistics
        self.time_since_update = 0  # reset because just updated
        self.hits += 1  # +1 because just updated

        # update 3D bounding box
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree
        box3d = Box3D.bbox2array(box3d)
        self.kf_box3d.x[3], box3d[3] = orientation_correction(self.kf_box3d.x[3], box3d[3])
        # kalman filter update with observation
        self.kf_box3d.update(box3d)
        self.kf_box3d.x[3] = within_range(self.kf_box3d.x[3])
        self.info = other_info

        # update 2D bounding box
        self.kf_box2d.update(Box2D.bbox2array(box2d))

        if self.hits >= self.min_hits:
            self.is_confirmed = True

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        # update statistics
        self.time_since_update += 1

        # propagate locations
        self.kf_box3d.predict()
        self.kf_box3d.x[3] = within_range(self.kf_box3d.x[3])

        if (self.kf_box2d.x[6] + self.kf_box2d.x[2]) <= 0:
            self.kf_box2d.x[6] *= 0.0
        self.kf_box2d.predict()

        return Box3D.array2bbox(self.kf_box3d.x.reshape((-1))[:7]), Box2D.array2bbox(self.kf_box2d.x.reshape((-1))[:4])

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return Box3D.array2bbox(self.kf_box3d.x.reshape((-1))[:7]), Box2D.array2bbox(self.kf_box2d.x.reshape((-1))[:4])


class Tracker2D(TrackerBase2D):
    def __init__(self, box2d, other_info, track_id, min_hits):
        super().__init__(box2d, other_info, track_id)

        # define constant velocity model  定义匀速模型
        self.kf_box2d = KalmanFilter(dim_x=7, dim_z=4)  # 状态变量是7维，观测值是4维的，按照需要的维度构建目标
        self.kf_box2d.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 0, 1],
                                    [0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1]])
        self.kf_box2d.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0]])

        self.kf_box2d.R *= R  # measurement uncertainty
        self.kf_box2d.P[4:, 4:] *= P1  # give high uncertainty to the unobservable initial velocities 对未观测到的初始速度给出高的不确定性
        self.kf_box2d.P[:4, :4] *= P2  # 默认定义的协方差矩阵是np.eye(dim_x)，将P中的数值与10，1000相乘，赋值不确定性
        self.kf_box2d.Q[4:, 4:] *= Q

        self.kf_box2d.x[:4] = self.initial_pos_box2d

        self.min_hits = min_hits

    def update(self, box2d, other_info):
        """
        Updates the state vector with observed bounding box.
        """
        # update statistics
        self.time_since_update = 0  # reset because just updated
        self.hits += 1  # +1 because just updated

        self.info = other_info

        # update 2D bounding box
        self.kf_box2d.update(Box2D.bbox2array(box2d))

        if self.hits >= self.min_hits:
            self.is_confirmed = True

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        # update statistics
        self.time_since_update += 1

        # propagate locations
        if (self.kf_box2d.x[6] + self.kf_box2d.x[2]) <= 0:
            self.kf_box2d.x[6] *= 0.0
        self.kf_box2d.predict()

        return Box2D.array2bbox(self.kf_box2d.x.reshape((-1))[:4])

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return Box2D.array2bbox(self.kf_box2d.x.reshape((-1))[:4])


def orientation_correction(theta_pre, theta_obs):
    # update orientation in propagated tracks and detected boxes so that they are within 90 degree

    # make the theta still in the range
    theta_pre = within_range(theta_pre)
    theta_obs = within_range(theta_obs)

    # if the angle of two theta is not acute angle, then make it acute
    if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:
        theta_pre += np.pi
        theta_pre = within_range(theta_pre)

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
        if theta_obs > 0:
            theta_pre += np.pi * 2
        else:
            theta_pre -= np.pi * 2

    return theta_pre, theta_obs


def within_range(theta):
    # make sure the orientation is within a proper range

    if theta >= np.pi: theta -= np.pi * 2  # make the theta still in the range
    if theta < -np.pi: theta += np.pi * 2

    return theta


class Box3D:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None):
        self.x = x  # center x
        self.y = y  # center y
        self.z = z  # center z
        self.h = h  # height
        self.w = w  # width
        self.l = l  # length
        self.ry = ry  # orientation

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h)

    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.ry}

    @classmethod
    def bbox2array(cls, bbox):
        return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])

    @classmethod
    def bbox2array_raw(cls, bbox):
        return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])

    @classmethod
    def array2bbox_raw(cls, data):
        # take the format of data of [h,w,l,x,y,z,theta]

        bbox = Box3D()
        bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def array2bbox(cls, data):
        # take the format of data of [x,y,z,theta,l,w,h]

        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox


class Box2D:
    def __init__(self, xx1=None, yy1=None, xx2=None, yy2=None):
        self.xx1 = xx1
        self.yy1 = yy1
        self.xx2 = xx2
        self.yy2 = yy2

    def __str__(self):
        return 'xx1: {}, yy1: {}, xx2: {}, yy2: {}'.format(self.xx1, self.yy1, self.xx2, self.yy2)

    @staticmethod
    def bbox2array(box2d):
        """
        Takes a bounding box in the form object BOX2D and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = box2d.xx2 - box2d.xx1
        h = box2d.yy2 - box2d.yy1
        if w < 0 or h < 0:
            print(box2d)
            exit()
        x = box2d.xx1 + w / 2.
        y = box2d.yy1 + h / 2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def array2bbox(data):
        """
          Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
          [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(data[2] * data[3])
        h = data[2] / w
        box2d = Box2D()
        box2d.xx1, box2d.yy1, box2d.xx2, box2d.yy2 = data[0] - w / 2., data[1] - h / 2., data[0] + w / 2., data[1] + h / 2.
        return box2d

    def average_with_other_box(self, box):
        return Box2D((self.xx1+box.xx1)/2, (self.yy1+box.yy1)/2, (self.xx2+box.xx2)/2, (self.yy2+box.yy2)/2)

    def is_not_on_the_boundary(self, img_w, img_h):
        return self.xx1 > 10 and self.xx2 < img_w - 10 and self.yy1 > 10 and self.yy2 < img_h - 10


class OtherInfo:
    def __init__(self, seq=None, frame=None, cls=None, alpha=None, score=None, feature=None):
        self.seq = seq
        self.frame = frame
        self.cls = cls
        self.alpha = alpha
        self.score = score
        self.feature = feature

    def __str__(self):
        return 'seq: {}, frame: {}, cls: {}, alpha: {}, score: {}'\
            .format(self.seq, self.frame, self.cls, self.alpha, self.score)
