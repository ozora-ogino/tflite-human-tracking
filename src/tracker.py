#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from sort import Sort
from utils import check_direction, is_intersect


class Tracker(object):
    def __init__(
        self,
        border: List[Tuple[int]],
        directions: Tuple[bool],
        count_callback: Optional[Callable] = None,
    ):
        """Constructor of Tracker.

        Args:
            border (List[Tuple[int]]): Border to detect count.
            count_callback (Optional[Callable], optional): Callback function which will be called when the counter is up.
                                                           Take counter(int) for arguments.
        """
        self.tracker = Sort()
        self.border = border
        self.count_callback = count_callback
        self.memory = {}
        self.counter = {key: 0 for key in directions.keys()}
        self.directions = directions

        np.random.seed(2021)
        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    def _is_count(
        self,
        center: Tuple[int],
        center_prev: Tuple[int],
        border: List[Tuple[int]],
        key: str,
    ) -> bool:
        """Check whether count or not.

        1. check_direction: Check the direction of human movement.
                            If direction is not specified, return True.
        2. is_intersect: Check whether the border and the human movement intersect.

        Args:
            center(Tuple[int]): Current center position.
            center_prev(Tuple[int]): Previous center position.
            border(List[Tuple[int]]): Border.
            key(str): "inside", "outside" or "total".
        """

        return check_direction(center, center_prev, self.directions[key]) and is_intersect(
            center, center_prev, border[0], border[1]
        )

    def update(self, frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """Update tracker and draw bounding box in a frame.

        Args:
            frame (np.ndarray): Target frame.
            dets (np.ndarray): Array like [xyxy + score].

        Returns:
            np.ndarray: Frame with bounding box and count.
        """
        # Update Sort.
        tracks = self.tracker.update(dets)

        boxes = []
        index_ids = []
        previous = self.memory.copy()

        for track in tracks.astype(int):
            boxes.append([track[0], track[1], track[2], track[3]])
            index_ids.append(track[4])
            # Add index id and box to memory.
            self.memory[index_ids[-1]] = boxes[-1]

        if len(boxes) == 0:
            return frame

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box

            color = [int(c) for c in self.COLORS[index_ids[i] % len(self.COLORS)]]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            if index_ids[i] in previous:
                previous_box = previous[index_ids[i]]
                xmin2, ymin2, wmax2, ymax2 = previous_box

                # Calculate the center of the bounding box.
                center = (int(xmin + (xmax - xmin) / 2), int(ymin + (ymax - ymin) / 2))
                center_prev = (int(xmin2 + (wmax2 - xmin2) / 2), int(ymin2 + (ymax2 - ymin2) / 2))

                # Draw a motion of bounding box.
                cv2.line(frame, center, center_prev, color, 3)

                callback = False
                for key in self.directions.keys():
                    if self._is_count(center, center_prev, self.border, key):
                        self.counter[key] += 1
                        callback = True

                # Execute callback.
                if self.count_callback and callback:
                    self.count_callback(self.counter)

            # Put ID on the box.
            cv2.putText(
                frame,
                str(index_ids[i]),
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw border.
        cv2.line(frame, self.border[0], self.border[1], (10, 255, 0), 3)
        # Put counter in the top left corner.
        for i, (key, count) in enumerate(self.counter.items()):
            cv2.putText(
                frame,
                f"{key}: {count}",
                (30, 30 + 80 * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (10, 255, 0),
                5,
            )
        return frame
