#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from sort import Sort
from utils import is_intersect


class Tracker(object):
    def __init__(
        self,
        border: List[Tuple[int]],
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
        self.counter = 0

        self.H, self.H = None, None

        np.random.seed(2021)
        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    def _is_count(self, motion, border):
        """Return true if motion and border intersect"""
        return is_intersect(motion[0], motion[1], border[0], border[1])

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

                if self._is_count([center, center_prev], self.border):
                    self.counter += 1
                    # Execute callback.
                    if self.count_callback:
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
        # Put counter in top right corner.
        cv2.putText(
            frame,
            str(self.counter),
            (100, 200),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            5.0,
            (10, 255, 0),
            3,
        )
        return frame
