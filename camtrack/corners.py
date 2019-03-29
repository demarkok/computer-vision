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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    max_corners = 1500
    quality_level = 0.05
    min_distance = 6
    corner_coordinates = cv2.goodFeaturesToTrack(
        image=image_0,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )
    prev_last_id = last_id = len(corner_coordinates)
    corners = FrameCorners(
        ids=np.array(range(last_id)),
        points=corner_coordinates,
        sizes=np.full(last_id, min_distance)
    )
    builder.set_corners_at_frame(0, corners)
    print('builing corners', end=' ')
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        print(frame, end=' ')
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prevImg=np.uint8(image_0 / image_0.max() * 255.0),
            nextImg=np.uint8(image_1 / image_1.max() * 255.0),
            prevPts=corner_coordinates,
            nextPts=None,
        )
        #ids = np.argsort(err.flatten())[:int(len(err) * 0.95)]
        ids = np.where(status == 1)[0]
        corner_coordinates = next_pts[ids]
        additional_corner_coordinates = []
        if len(corner_coordinates) < max_corners:
            potential_corner_coordinates = cv2.goodFeaturesToTrack(
                image=image_1,
                maxCorners=max_corners,
                qualityLevel=quality_level,
                minDistance=min_distance,
            )
            for potential_corner_coordinate in potential_corner_coordinates:
                distances = np.linalg.norm(potential_corner_coordinate - corner_coordinates, axis=1)
                if np.linalg.norm(distances, axis=1).min() >= min_distance:
                    additional_corner_coordinates.append(potential_corner_coordinate)
                    last_id += 1
                if len(corner_coordinates) + len(additional_corner_coordinates) >= max_corners:
                    break
        if len(additional_corner_coordinates) != 0:
            corner_coordinates = np.concatenate([corner_coordinates, additional_corner_coordinates])
            ids = np.concatenate([ids, np.array(range(prev_last_id, last_id))])
            prev_last_id = last_id
        corners = FrameCorners(
            ids=ids,
            points=corner_coordinates,
            sizes=np.full(len(corner_coordinates), min_distance),
        )
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1
    print()

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
