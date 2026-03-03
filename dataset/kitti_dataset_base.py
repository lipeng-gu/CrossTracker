import os
import numpy as np
from PIL import Image
from numba import jit


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = self.get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        if 'Tr_imu2velo' in calib:
            self.I2V = calib['Tr_imu2velo']  # 3 x 4
            self.I2V = np.reshape(self.I2V, [3, 4])
            self.V2I = self.inverse_rigid_trans(self.I2V)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    @staticmethod
    def get_calib_from_file(calib_file):
        with open(calib_file) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        if len(lines) > 6:
            obj = lines[6].strip().split(' ')[1:]
            Tr_imu_to_velo = np.array(obj, dtype=np.float32)

            return {'P2': P2.reshape(3, 4),
                    'P3': P3.reshape(3, 4),
                    'R0': R0.reshape(3, 3),
                    'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4),
                    'Tr_imu2velo': Tr_imu_to_velo.reshape(3, 4)}
        else:
            return {'P2': P2.reshape(3, 4),
                    'P3': P3.reshape(3, 4),
                    'R0': R0.reshape(3, 3),
                    'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

    @staticmethod
    def inverse_rigid_trans(Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def velo_to_imu(self, pts_velo):
        pts_velo = self.cart_to_hom(pts_velo)  # nx4
        return np.dot(pts_velo, np.transpose(self.V2I))

    def rect_to_imu(self, pts_rect):
        pts_velo = self.rect_to_lidar(pts_rect)
        pts_imu = self.velo_to_imu(pts_velo)

        return pts_imu

    def imu_to_velo(self, pts_imu):
        pts_imu = self.cart_to_hom(pts_imu)  # nx4
        return np.dot(pts_imu, np.transpose(self.I2V))

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart_to_hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def imu_to_rect(self, pts_imu):
        pts_velo = self.imu_to_velo(pts_imu)
        pts_ref = self.project_velo_to_ref(pts_velo)
        pts_rect = self.project_ref_to_rect(pts_ref)
        return pts_rect

    def convert_boxes3d_cam_to_lidar(self, boxes3d_cam):
        """
        Args:
            boxes3d_cam: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
            calib:

        Returns:
            boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        """
        xyz_camera = boxes3d_cam[:, 0:3]
        l, h, w, r = boxes3d_cam[:, 3:4], boxes3d_cam[:, 4:5], boxes3d_cam[:, 5:6], boxes3d_cam[:, 6:7]
        xyz_lidar = self.rect_to_lidar(xyz_camera)
        xyz_lidar[:, 2] += h[:, 0] / 2
        return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

    def convert_boxes3d_cam_to_image(self, boxes3d, image_shape=None):
        """
        :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        :param calib:
        :return:
            box_2d_preds: (N, 4) [x1, y1, x2, y2]
        """
        corners3d = self.convert_boxes3d_to_corners3d(boxes3d)
        pts_img, _ = self.rect_to_img(corners3d.reshape(-1, 3))
        corners_in_image = pts_img.reshape(-1, 8, 2)

        min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
        max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
        boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
        if image_shape is not None:
            boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        return boxes2d_image

    @staticmethod
    def convert_boxes3d_to_corners3d(boxes3d, bottom_center=True):
        """
        :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
        :param bottom_center: whether y is on the bottom center of object
        :return: corners3d: (N, 8, 3)
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        """
        boxes_num = boxes3d.shape[0]
        l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
        x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
        z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
        if bottom_center:
            y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
            y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
        else:
            y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.],
                                 dtype=np.float32).T

        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                             [zeros, ones, zeros],
                             [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

        x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

        x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
        y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
        z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

        corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

        return corners.astype(np.float32)

    def egomotion_compensation_ID(self, traj_id, ego_rot_imu, ego_xyz_imu, left, right, mask=None):
        # traj_id           # N x 3
        # ego_imu can have frames less than pre+fut due to sequence boundary

        # convert trajectory data from rect to IMU for ego-motion compensation
        traj_id_imu = self.rect_to_imu(traj_id)  # less_pre x 3

        if mask is not None:
            good_index = np.where(mask == 1)[0]
            good_index = (good_index - left).tolist()
            ego_rot_imu = np.array(ego_rot_imu)
            ego_rot_imu = ego_rot_imu[good_index, :].tolist()

        # correct rotation
        for frame in range(traj_id_imu.shape[0]):
            traj_id_imu[frame, :] = np.matmul(ego_rot_imu[frame], traj_id_imu[frame, :].reshape((3, 1))).reshape((3,))

        # correct transition
        if mask is not None:
            traj_id_imu += ego_xyz_imu[good_index, :]  # les_frames x 3, TODO, need to test which is correct
        else:
            traj_id_imu += ego_xyz_imu[:traj_id_imu.shape[0], :]  # les_frames x 3

        # convert trajectory data back to rect coordinate for visualization
        traj_id_rect = self.imu_to_rect(traj_id_imu)

        return traj_id_rect


class Oxts(object):
    def __init__(self, oxts_file):
        self.imu_poses = self.load_oxts(oxts_file)  # seq_frames x 4 x 4

    def load_oxts(self, oxts_file):
        """Load OXTS data from file."""
        # https://github.com/pratikac/kitti/blob/master/pykitti/raw.py

        ext = os.path.splitext(oxts_file)[-1]
        if ext == '.json':  # loading for nuScenes-to-KITTI data
            with open(oxts_file, 'r') as file:
                imu_poses = json.load(file)
                imu_poses = np.array(imu_poses)

            return imu_poses

        # Extract the data from each OXTS packe per dataformat.txt
        from collections import namedtuple
        OxtsPacket = namedtuple('OxtsPacket',
                                'lat, lon, alt, ' +
                                'roll, pitch, yaw, ' +
                                'vn, ve, vf, vl, vu, ' +
                                'ax, ay, az, af, al, au, ' +
                                'wx, wy, wz, wf, wl, wu, ' +
                                'pos_accuracy, vel_accuracy, ' +
                                'navstat, numsats, ' +
                                'posmode, velmode, orimode')

        oxts_packets = []
        with open(oxts_file, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                data = OxtsPacket(*line)
                oxts_packets.append(data)

        # Precompute the IMU poses in the world frame
        imu_poses = self._poses_from_oxts(oxts_packets)  # seq_frames x 4 x 4

        return imu_poses


    def get_ego_traj(self, frame, pref, futf, inverse=False, only_fut=False):
        # compute the motion of the ego vehicle for ego-motion compensation
        # using the current frame as the coordinate
        # current frame means one frame prior to future, and also the last frame of the past

        # compute the start and end frame to retrieve the imu poses
        num_frames = self.imu_poses.shape[0]
        assert frame >= 0 and frame <= num_frames - 1, 'error'
        if inverse:  # pre and fut are inverse, i.e., inverse ego motion compensation
            start = min(frame + pref - 1, num_frames - 1)
            end = max(frame - futf - 1, -1)
            index = [*range(start, end, -1)]
        else:
            start = max(frame - pref + 1, 0)
            end = min(frame + futf + 1, num_frames)
            index = [*range(start, end)]

        # compute frame offset due to sequence boundary
        left = start - (frame - pref + 1)
        right = (frame + futf + 1) - end

        # compute relative transition compared to the current frame of the ego
        all_world_xyz = self.imu_poses[index, :3, 3]  # N x 3, only translation, frame = 10-19 for fut only (0-19 for all)
        cur_world_xyz = self.imu_poses[frame]  # 4 x 4, frame = 9
        T_world2imu = np.linalg.inv(cur_world_xyz)
        all_world_hom = np.concatenate((all_world_xyz, np.ones((all_world_xyz.shape[0], 1))), axis=1)  # N x 4
        all_xyz = all_world_hom.dot(T_world2imu.T)[:, :3]  # N x 3

        # compute relative rotation compared to the current frame of the ego
        all_world_rot = self.imu_poses[index, :3, :3]  # N x 3 x 3, only rotation
        cur_world_rot = self.imu_poses[frame, :3, :3]  # 3 x 3, frame = 9
        T_world2imu_rot = np.linalg.inv(cur_world_rot)
        all_rot_list = list()
        for frame in range(all_world_rot.shape[0]):
            all_rot_tmp = all_world_rot[frame].dot(T_world2imu_rot)  # 3 x 3
            all_rot_list.append(all_rot_tmp)

        if only_fut:
            fut_xyz, fut_rot_list = all_xyz[pref - left:], all_rot_list[pref - left:]
            return fut_xyz, fut_rot_list, left, right
        else:
            return all_xyz, all_rot_list, left, right

    @staticmethod
    @jit
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])

    @staticmethod
    @jit
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    @staticmethod
    @jit
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    @staticmethod
    @jit
    def transform_from_rot_trans(R, t):
        """Transforation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    @jit
    def _poses_from_oxts(self, oxts_packets):
        """Helper method to compute SE(3) pose matrices from OXTS packets."""
        # https://github.com/pratikac/kitti/blob/master/pykitti/raw.py

        er = 6378137.  # earth radius (approx.) in meters

        # compute scale from first lat value
        scale = np.cos(oxts_packets[0].lat * np.pi / 180.)

        t_0 = []  # initial position
        poses = []  # list of poses computed from oxts
        for packet in oxts_packets:
            # Use a Mercator projection to get the translation vector
            tx = scale * packet.lon * np.pi * er / 180.
            ty = scale * er * \
                 np.log(np.tan((90. + packet.lat) * np.pi / 360.))
            tz = packet.alt
            t = np.array([tx, ty, tz])

            # We want the initial position to be the origin, but keep the ENU
            # coordinate system
            if len(t_0) == 0:
                t_0 = t

            # Use the Euler angles to get the rotation matrix
            Rx = self.rotx(packet.roll)
            Ry = self.roty(packet.pitch)
            Rz = self.rotz(packet.yaw)
            R = Rz.dot(Ry.dot(Rx))

            # Combine the translation and rotation into a homogeneous transform
            poses.append(self.transform_from_rot_trans(R, t - t_0))  # store transformation matrix

        return np.stack(poses)


"""
description: read lidar data given 
input: lidar bin path "path", cam 3D to cam 2D image matrix (4,4), lidar 3D to cam 3D matrix (4,4)
output: valid points in lidar coordinates (PointsNum,4)
"""
def read_velodyne(path):

    if not os.path.exists(path):
        return np.zeros((0, 4), dtype=np.float32)
    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    return lidar

def read_image(path):
    im = Image.open(path)
    return im

def load_tracking_label(label_path, class_map):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip().split(' ')
        if line[2] not in class_map.keys():
            continue
        line[2] = class_map[line[2]]
        new_lines.append([float(x) for x in line])

    return np.array(new_lines)
