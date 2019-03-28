#! /usr/bin/env python3
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

from _corners import FrameCorners
import cv2

def solvePnPRansac(points3d: np.ndarray, points2d: np.ndarray, intrinsic_mat: np.ndarray, reprojectionError, iters):
    assert (points3d.shape[0] == points2d.shape[0] >= 4 and
            points3d.shape[1] == 3 and
            points2d.shape[1] == 2 and
            iters > 0)

    n = points3d.shape[0]

    ids = np.array(list(range(n)))

    r_vec_opt, t_vec_opt, inliers_opt = -1, -1, np.empty(0)

    for _ in range(iters):
        sample = np.random.choice(ids, 4)

        _, r_vec, t_vec = cv2.solvePnP(points3d[sample],
                                       points2d[sample].reshape(-1, 1, 2),
                                       intrinsic_mat, None, flags=cv2.SOLVEPNP_EPNP)
        view = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        inliers = calc_inlier_indices(points3d, points2d, intrinsic_mat @ view, reprojectionError)
        if inliers.shape[0] > inliers_opt.shape[0]:
            inliers_opt, r_vec_opt, t_vec_opt = inliers, r_vec, t_vec

    return None, r_vec_opt, t_vec_opt, None if inliers_opt.shape[0] == 0 else inliers_opt


class Tracker:
    def __init__(self, corners: CornerStorage, intrinsic_mat: np.ndarray):
        self._frames = corners
        self._intrinsic_mat = intrinsic_mat
        self._triangulation_parameters = TriangulationParameters(0.7, 6, 0.1)
        self._cloudBuilder = PointCloudBuilder()
        self.HOM_INLIERS_THRESHOLD = 1.3
        self.FRAMES_TO_INITIALIZE_DENSITY = 100

        self._n_of_frames = len(corners)
        self._max_id = corners.max_corner_id()

        self._poses = {}
        self._corner3ds = {}
        self._corner3ds_being_found_counter = np.array([0] * (self._max_id + 1))

    def _initial_pose(self, frame1: FrameCorners, frame2: FrameCorners) -> Tuple[Pose, float]:
        frame1._points = np.array(frame1.points, dtype=np.float32)
        corresps = build_correspondences(frame1, frame2)

        E, mask_essential = cv2.findEssentialMat(corresps.points_1, corresps.points_2, self._intrinsic_mat)
        mask_essential = mask_essential.flatten()
        essential_inliers = np.sum(mask_essential)

        _, mask_homography = cv2.findHomography(corresps.points_1, corresps.points_2, cv2.RANSAC)
        mask_homography = mask_homography.flatten()
        homography_inliners = np.sum(mask_homography)

        if homography_inliners / essential_inliers > self.HOM_INLIERS_THRESHOLD:
            return None, 0

        corresps = _remove_correspondences_with_ids(corresps, corresps.ids[mask_essential == 0])

        R1, R2, t = cv2.decomposeEssentialMat(E)
        poses = [Pose(R1.T, R1.T.dot(t)), Pose(R1.T, R1.T.dot(-t)), Pose(R2.T, R2.T.dot(t)), Pose(R2.T, R2.T.dot(-t))]
        triangulated_in_pose = []
        for pose in poses:
            points, ids, re_median, cos_median = triangulate_correspondences(corresps, eye3x4(),
                                                                             pose_to_view_mat3x4(pose),
                                                                             self._intrinsic_mat,
                                                                             self._triangulation_parameters)
            triangulated_in_pose.append(points.shape[0])

        pose_id = np.array(triangulated_in_pose).argmax()
        return poses[pose_id], max(triangulated_in_pose)

    def _initialize(self):

        print("Initializing...")

        optimal_pose = None
        optimal_frame1, optimal_frame2, optimal_metric = 0, 0, 0

        step = max(1, self._n_of_frames // self.FRAMES_TO_INITIALIZE_DENSITY)

        i = 0
        for j in range(i + 1, self._n_of_frames, step):
            pose, metric = self._initial_pose(self._frames[i], self._frames[j])
            if metric > optimal_metric:
                print(i, j, metric)
                optimal_metric = metric
                optimal_frame1, optimal_frame2, optimal_pose = i, j, pose

        self._poses[optimal_frame1] = view_mat3x4_to_pose(eye3x4())
        self._poses[optimal_frame2] = optimal_pose
        self._add_cloud_points(optimal_frame1, optimal_frame2)

    def _add_cloud_points(self, frame1: int, frame2: int):
        corresps = build_correspondences(self._frames[frame1], self._frames[frame2])
        points, ids, _, _ = triangulate_correspondences(corresps,
                                                        pose_to_view_mat3x4(self._poses[frame1]),
                                                        pose_to_view_mat3x4(self._poses[frame2]),
                                                        self._intrinsic_mat, self._triangulation_parameters)
        for point, id in zip(points, ids):
            if id not in self._corner3ds:
                self._corner3ds[id] = point
                self._corner3ds_being_found_counter[id] += 1
            else:  # update point to be average
                self._corner3ds[id] *= self._corner3ds_being_found_counter[id]
                self._corner3ds[id] += point
                self._corner3ds_being_found_counter[id] += 1
                self._corner3ds[id] /= self._corner3ds_being_found_counter[id]

    def _find_best_frame(self, mask: np.ndarray) -> Tuple[int, int]:
        optimal_frame, optimal_metric = -1, 0
        for frame in range(self._n_of_frames):
            if mask[frame] == 0 and frame not in self._poses:
                metric = len(self._get_estimated_corners_on_frame(frame)[0])
                if metric > optimal_metric:
                    optimal_metric = metric
                    optimal_frame = frame
        return optimal_frame, optimal_metric

    def _get_estimated_corners_on_frame(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.array(list(set(self._frames[frame].ids.flatten()) & set(self._corner3ds.keys())))
        id2point2d = dict(zip(self._frames[frame].ids.flatten(), self._frames[frame].points))
        return ids, np.array([id2point2d[id] for id in ids])

    def _estimate_camera_on_frame(self, frame: int):
        found_corners_id, found_corners2d = self._get_estimated_corners_on_frame(frame)

        if len(found_corners_id) < 4:
            return
        found_corners3d = np.array([self._corner3ds[i] for i in found_corners_id])

        # _, rvec, tvec, inliers = cv2.solvePnPRansac(found_corners3d, found_corners2d, self._intrinsic_mat, None,
        #                                             reprojectionError=self._triangulation_parameters.max_reprojection_error, useExtrinsicGuess=True)

        _, rvec, tvec, inliers = solvePnPRansac(found_corners3d, found_corners2d, self._intrinsic_mat,
                                                reprojectionError=self._triangulation_parameters.max_reprojection_error,
                                                iters=100)

        if inliers is None:
            return

        outlier_ids = np.delete(np.array(list(found_corners_id)), inliers.flatten())

        print("{} outliers has been removed".format(outlier_ids.shape[0]))

        # print(frame, found_corners_id[50], found_corners2d[50])

        for outlier in outlier_ids:
            self._corner3ds.pop(outlier)

        self._poses[frame] = view_mat3x4_to_pose(rodrigues_and_translation_to_view_mat3x4(rvec, tvec))

    def _frame_metric(self, frame1, frame2, pose1, pose2) -> int:
        corresps = build_correspondences(self._frames[frame1], self._frames[frame2])
        _, ids, _, _ = triangulate_correspondences(corresps, pose_to_view_mat3x4(pose1), pose_to_view_mat3x4(pose2),
                                                   self._intrinsic_mat, self._triangulation_parameters)
        return ids.shape[0]

    def track(self):
        print(self._n_of_frames)
        self._initialize()
        frames_estimated = 2
        mask = np.zeros(self._n_of_frames)
        first_frame = 0
        while frames_estimated < self._n_of_frames:
            frame, metric = self._find_best_frame(mask)
            self._estimate_camera_on_frame(frame)
            if frame not in self._poses:
                mask[frame] = 1
                continue

            if frames_estimated == 2:
                first_frame = frame

            frames_estimated += 1

            for frame2 in range(self._n_of_frames):
                if frame2 == frame or frame2 not in self._poses:
                    continue
                self._add_cloud_points(frame, frame2)

            print("Estimating position on the frame #{}\n"
                  "  Cloud size = {}\n"
                  "  Number of estimated corners on the frame = {}\n"
                  "  {} frames out of {}\n\n".format(frame, len(self._corner3ds), metric, frames_estimated,
                                                     self._n_of_frames))

            mask.fill(0)

    def get_track(self) -> List[np.ndarray]:
        return [pose_to_view_mat3x4(self._poses[i]) for i in range(self._n_of_frames)]

    def get_point_cloud(self) -> PointCloudBuilder:
        return PointCloudBuilder(*map(np.array, zip(*self._corner3ds.items())))


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    tracker = Tracker(corner_storage, intrinsic_mat)
    tracker.track()
    return tracker.get_track(), tracker.get_point_cloud()


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
