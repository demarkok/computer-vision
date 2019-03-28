#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
import _corners

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _to256(img):
    return np.array(img * 256, dtype=np.uint8)


class _CornerTracker:

    def __init__(self):
        self.n_corners = 800
        self.circle_size = 3
        self.safe_area_radius = 6
        self.image = None
        self.corners = None
        self.safe_area_delta = [(i, j) for i in range(-self.safe_area_radius, self.safe_area_radius + 1) for j in
                                range(-self.safe_area_radius, self.safe_area_radius + 1)
                                if i ** 2 + j ** 2 < self.safe_area_radius ** 2]
        self.mask = None
        self.flow_params = dict(winSize=(15, 15),
                                maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3),
                                minEigThreshold=0.005)
        self.corners_params = dict(maxCorners=self.n_corners,
                                   qualityLevel=0.05,
                                   minDistance=self.safe_area_radius,
                                   blockSize=10,
                                   gradientSize=1,
                                   mask=None)

    def update_mask(self):
        def is_bounded(x, y):
            return 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]
        self.mask = np.ones((self.image.shape[1], self.image.shape[0]), dtype=np.uint8)
        for point in np.array(self.corners.points, dtype=np.int32):
            for d in self.safe_area_delta:
                neighbour = point + d
                if is_bounded(*neighbour):
                    self.mask[tuple(neighbour)] = 0

    def add_new_corners(self, new_corner_points):
        if new_corner_points is None:
            return
        new_corner_points = np.array(new_corner_points, dtype=np.int32).reshape(-1, 2)
        if self.corners is None:
            self.corners = FrameCorners(np.array(range(new_corner_points.shape[0])),
                                        new_corner_points,
                                        np.array([self.circle_size] * new_corner_points.shape[0]))
            return
        self.corners.add_corners(new_corner_points, self.circle_size)

    def update_image(self, new_image):
        if self.image is None:
            self.image = new_image
            self.add_new_corners(cv2.goodFeaturesToTrack(new_image, **self.corners_params))
        else:
            if self.corners.ids.shape[0] != 0:
                updated_corner_points, status, _ = cv2.calcOpticalFlowPyrLK(_to256(self.image),
                                                                            _to256(new_image),
                                                                            np.array(self.corners.points,
                                                                                     dtype=np.float32).reshape(-1, 2),
                                                                            None, **self.flow_params)
                status = np.array(status, dtype=np.bool).reshape(-1)
                self.corners = _corners.filter_frame_corners(self.corners, status)  # remove non-tracked points
                self.corners._points = np.array(updated_corner_points, dtype=np.float32)[status]  # update coordinates
                self.corners_params['maxCorners'] = self.n_corners - self.corners.points.shape[0]
                if self.corners_params['maxCorners'] > 0:
                    self.update_mask()
                    self.corners_params['mask'] = self.mask.transpose()
                    self.add_new_corners(cv2.goodFeaturesToTrack(new_image, **self.corners_params))
        self.image = new_image
        return self.corners


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    tracker = _CornerTracker()
    for frame, image in enumerate(frame_sequence):
        corners = tracker.update_image(image)
        builder.set_corners_at_frame(frame, corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
