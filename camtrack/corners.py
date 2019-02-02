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
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3),
                              minEigThreshold=0.005)

        self.feature_params = dict(maxCorners=1000,
                                   qualityLevel=0.02,
                                   minDistance=5,
                                   blockSize=10)
        self.circle_size = 3

        self.image = None
        self.corners = None

    def add_new_corners(self, new_corner_points):
        new_corner_points = np.array(new_corner_points, dtype=np.int32).reshape(-1, 2)
        if self.corners is None:
            self.corners = FrameCorners(np.array(range(new_corner_points.shape[0])),
                                        new_corner_points,
                                        np.array([self.circle_size] * new_corner_points.shape[0]))
            return

        marked = np.zeros_like(self.image, dtype=np.bool).transpose()
        for point in np.array(self.corners.points.round(), dtype=np.int32):
            marked[point] = True

        def empty_window(position):
            return not marked[tuple(position)]

        points = []
        for point in new_corner_points:
            if empty_window(point):
                points.append(point)
        self.corners.add_corners(points, self.circle_size)

    def update_image(self, new_image):
        if self.image is None:
            self.image = new_image
            self.add_new_corners(cv2.goodFeaturesToTrack(new_image, **self.feature_params))
        else:
            if self.corners.ids.shape[0] != 0:
                refreshed_corners, status, _ = cv2.calcOpticalFlowPyrLK(_to256(self.image),
                                                                        _to256(new_image),
                                                                        np.array(self.corners.points, dtype=np.float32).round().reshape(-1, 2),
                                                                        None, **self.lk_params)
                status = np.array(status, dtype=np.bool).reshape(-1)
                self.corners = _corners.filter_frame_corners(self.corners, status)
                self.corners._points = np.array(refreshed_corners)[status]
                self.add_new_corners(cv2.goodFeaturesToTrack(new_image, **self.feature_params))

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
