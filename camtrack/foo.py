
#! /usr/bin/env python3
from cv2.cv2 import findFundamentalMat, findHomography, RANSAC, findEssentialMat, decomposeEssentialMat, solvePnPRansac

from _corners import FrameCorners

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


W_MATRIX = np.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 1.]
])


class CameraTracker:
    def __init__(self, corner_storage: CornerStorage, intrinsic_mat: np.ndarray, triangulation_parameters: TriangulationParameters):
        max_id = np.amax(np.concatenate([corners.ids for corners in corner_storage]))

        self.__id_to_position = [None for _ in range(max_id + 1)]
        self.__number_of_times_deleted = [0 for _ in range(max_id + 1)]
        self.__frame_matrix = [None for _ in corner_storage]

        self.__corner_storage = corner_storage
        self.__intrinsic_mat = intrinsic_mat

        self.__triangulation_parameters = triangulation_parameters

        self.__failed = False

        self.__track_initialization()
        self.__track_frames()

        for i in range(len(self.__frame_matrix)):
            if self.__frame_matrix[i] is None:
                self.__frame_matrix[i] = eye3x4()

    def __calculate_pose(self, frame_corners1: FrameCorners, frame_corners2: FrameCorners):
        correspondences = build_correspondences(frame_corners1, frame_corners2)

        if correspondences.points_1.shape[0] <= 5:
            return None, None

        E, mask = findEssentialMat(correspondences.points_1, correspondences.points_2, self.__intrinsic_mat)  # todo: have parameters
        mask = mask.reshape(-1)
        filtered_correspondences = build_correspondences(frame_corners1, frame_corners2, np.nonzero(1 - mask)[0])

        fundamental_inliers = np.count_nonzero(mask)

        H, mask = findHomography(correspondences.points_1, correspondences.points_2, RANSAC)  # todo: configure
        mask = mask.reshape(-1)

        homography_inliers = np.count_nonzero(mask)

        if fundamental_inliers / homography_inliers < 1.1: # 1.5
            return None, None

        R1, R2, t1 = decomposeEssentialMat(E)
        t1.reshape(-1)
        t2 = -t1

        possible_poses = [Pose(R1, t1), Pose(R1, t2), Pose(R2, t1), Pose(R2, t2)]

        pose_cloud_size = []
        for pose in possible_poses:
            positions, ids = triangulate_correspondences(
                filtered_correspondences,
                eye3x4(),
                pose_to_view_mat3x4(pose),
                self.__intrinsic_mat,
                self.__triangulation_parameters
            )

            pose_cloud_size.append(ids.shape[0])

        index = np.argmax(pose_cloud_size)
        if pose_cloud_size[index] == 0:
            return None, None

        pose = possible_poses[index]

        return pose, pose_cloud_size[index]

    def __track_frames(self):
        frames_left = len(self.__frame_matrix) - 2
        for _ in range(frames_left):
        #for _ in range(0): # 10
            mask = np.ones(len(self.__frame_matrix))
            frame_id = self.__best_frame_to_estimate(mask)
            while not self.__add_frame(frame_id):
                if self.__failed:
                    return

                mask[frame_id] = False
                frame_id = self.__best_frame_to_estimate(mask)

            for id, matrix in enumerate(self.__frame_matrix):
                if matrix is None or id == frame_id:
                    continue

                self.__update_cloud(id, frame_id)

            frames_left -= 1
            print('Frames left: {}'.format(frames_left))

    def __track_initialization(self):
        poses, qualities = zip(*[self.__calculate_pose(self.__corner_storage[0], i) for i in self.__corner_storage])
        index = np.nanargmax(np.array(qualities, dtype=np.float32))

        self.__frame_matrix[0] = eye3x4()
        self.__frame_matrix[index] = pose_to_view_mat3x4(poses[index])
        self.__update_cloud(0, index)

    def __update_cloud(self, frame_id1, frame_id2):
        correspondences = build_correspondences(self.__corner_storage[frame_id1], self.__corner_storage[frame_id2])

        positions, ids = triangulate_correspondences(
            correspondences,
            self.__frame_matrix[frame_id1],
            self.__frame_matrix[frame_id2],
            self.__intrinsic_mat,
            self.__triangulation_parameters
        )

        points_added = 0
        for pos, id in zip(positions, ids):
            if self.__id_to_position[id] is None:
                self.__id_to_position[id] = pos
                points_added += 1

        if points_added > 0:
            print('Frame {} and {}: {} 3d points added'.format(frame_id1, frame_id2, points_added))

    def __points_on_frame(self, frame_corners: FrameCorners):
        mask = np.ones(frame_corners.ids.shape[0], dtype=np.bool)

        for i in range(mask.shape[0]):
            if self.__id_to_position[frame_corners.ids[i][0]] is None:
                mask[i] = 0

        return frame_corners.ids[mask, 0], frame_corners.points[mask]

    def __best_frame_to_estimate(self, mask):
        max_num_of_points = 0
        best_index = -1

        for index, matrix in enumerate(self.__frame_matrix):
            if matrix is not None or not mask[index]:
                continue

            num_of_points_on_frame = len(self.__points_on_frame(self.__corner_storage[index])[0])
            if num_of_points_on_frame > max_num_of_points:
                max_num_of_points = num_of_points_on_frame
                best_index = index

        return best_index

    def __add_frame(self, index):
        ids, image_points = self.__points_on_frame(self.__corner_storage[index])

        if ids.shape[0] < 6:
            self.__failed = True
            return False

        object_points = []
        for id in ids:
            object_points.append(self.__id_to_position[id])

        object_points = np.array(object_points)

        retval, rvec, tvec, inliers = solvePnPRansac(object_points, image_points, self.__intrinsic_mat, None)
        # todo: configure

        if not retval:
            return False

        inliers = inliers.reshape(-1)
        mask = np.ones(ids.shape[0], dtype=np.bool)
        mask[inliers] = 0

        outliers_num = 0
        for outlier_id in ids[mask]:
            self.__id_to_position[outlier_id] = None
            outliers_num += 1

        if outliers_num > 0:
            print('PnP excluded {} outliers'.format(outliers_num))

        self.__frame_matrix[index] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        print('Restored camera on frame #{}'.format(index))
        return True


    def get_frame_matrices(self) -> List[np.ndarray]:
        return self.__frame_matrix

    def get_cloud_builder(self):
        ids = np.array(list(map(lambda x: x[0], filter(lambda x: x[1] is not None, enumerate(self.__id_to_position)))))
        positions = np.array([self.__id_to_position[i] for i in ids])

        return PointCloudBuilder(ids, positions)

    def is_failed(self):
        return self.__failed


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    max_reprojection_error = 1.
    tracker = CameraTracker(corner_storage, intrinsic_mat,
                            TriangulationParameters(max_reprojection_error=max_reprojection_error,
                                                    min_triangulation_angle_deg=4., min_depth=0.1))

    return tracker.get_frame_matrices(), tracker.get_cloud_builder()


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()