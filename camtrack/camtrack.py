#! /usr/bin/env python3
import cv2

from _camtrack import _remove_correspondences_with_ids

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


triangulation_parameters = TriangulationParameters(
    max_reprojection_error=1.0,
    min_triangulation_angle_deg=5.0,
    min_depth=0.1
)


def _initialize_with_two_frames(corners_1, corners_2, intrinsic_mat):
    correspondences = build_correspondences(corners_1, corners_2)
    if correspondences.points_1.shape[0] <= 5:
        return 0, None, None, None
    E, mask_E = cv2.findEssentialMat(
        correspondences.points_1,
        correspondences.points_2,
        intrinsic_mat,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    fundamental_inliers = np.sum(mask_E)
    H, mask_H = cv2.findHomography(
        correspondences.points_1,
        correspondences.points_2,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999
    )
    homography_inliers = np.sum(mask_H)
    if fundamental_inliers / homography_inliers < 1:
        return 0, None, None, None
    correspondences = _remove_correspondences_with_ids(correspondences, np.where(mask_E == 0)[0])
    best_points, best_ids, best_pose = None, None, None
    R1, R2, t_d = cv2.decomposeEssentialMat(E)
    for pose in [Pose(R1.T, R1.T @ t_d), Pose(R2.T, R2.T @ t_d), Pose(R1.T, R1.T @ -t_d), Pose(R2.T, R2.T @ -t_d)]:
        points, ids = triangulate_correspondences(
            correspondences,
            eye3x4(),
            pose_to_view_mat3x4(pose),
            intrinsic_mat,
            triangulation_parameters,
        )
        if best_points is None or points.size > best_points.size:
            best_points, best_ids, best_pose = points, ids, pose
    return len(best_points), best_points, best_ids, best_pose


def _initialize_with_storage(corner_storage, intrinsic_mat):
    print("init...")
    best_size = 0
    best_index = 1
    for i in range(1, len(corner_storage)):
        print("init 0 and", i)
        size, _, _, _ = _initialize_with_two_frames(corner_storage[0], corner_storage[i], intrinsic_mat)
        if size > best_size:
            best_index = i

    init_size, init_points, init_ids, init_pose = \
        _initialize_with_two_frames(corner_storage[0], corner_storage[best_index], intrinsic_mat)
    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder.add_points(init_ids, init_points)
    return best_index, init_size, init_points, init_ids, init_pose, point_cloud_builder


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    view_mats = [eye3x4()] * len(corner_storage)
    init_index, init_size, init_points, init_ids, init_pose, point_cloud_builder = \
        _initialize_with_storage(corner_storage, intrinsic_mat)

    for i in range(1, len(corner_storage)):
        corners = corner_storage[i]
        print("frame", i, "/", len(corner_storage))
        if i == init_index:
            view_mats[i] = pose_to_view_mat3x4(init_pose)
        else:
            ids = []
            object_points = []
            image_points = []
            for id, point in zip(corners.ids, corners.points):
                indices, _ = np.nonzero(point_cloud_builder.ids == id)
                if len(indices) == 0:
                    continue
                ids.append(id)
                object_points.append(point_cloud_builder.points[indices[0]])
                image_points.append(point)
            if len(object_points) < 4:
                continue
            solve_result, R, t, inliers = cv2.solvePnPRansac(
                np.array(object_points, dtype=np.float64).reshape((len(object_points), 1, 3)),
                np.array(image_points, dtype=np.float64).reshape((len(object_points), 1, 2)),
                cameraMatrix=intrinsic_mat,
                distCoeffs=None
            )
            if not solve_result:
                continue
            print(inliers.tolist(), "inliers")
            view_mats[i] = rodrigues_and_translation_to_view_mat3x4(R, t)
        new_points = 0
        for j in range(i):
            correspondences = build_correspondences(
                corner_storage[j],
                corner_storage[i],
                ids_to_remove=point_cloud_builder.ids
            )
            points, ids = triangulate_correspondences(
                correspondences,
                view_mats[j],
                view_mats[i],
                intrinsic_mat,
                triangulation_parameters
            )
            point_cloud_builder.add_points(ids, points)
            new_points += len(points)
        print(new_points, "new points")
    return view_mats, point_cloud_builder


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
