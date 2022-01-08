#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

from typing import Tuple

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class Detect(object):
    """YOLOv5 tflite detect model."""

    def __init__(
        self,
        model_file: str,
        conf_thr: float,
    ):
        # Load model to memory.
        self.interpreter = Interpreter(model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter
        _, self.width, self.height, _ = self.interpreter.get_input_details()[0]["shape"]
        self.output_details = self.interpreter.get_output_details()
        self.conf_thr = conf_thr

    def detect(self, img: np.ndarray, box_type="xywh") -> Tuple[np.ndarray]:
        """Detect objects.
        Returns:
           Tuple[np.ndarray]: The shape of each element is (25500, 4) (25500,) (25500,).
        """
        img = self.preprocess(img)
        output_data = self._detect(img)
        boxes, scores, class_idx = self.postprocess(output_data, box_type)
        return boxes, scores, class_idx

    def _detect(self, img: np.ndarray):
        """Inference."""
        self.interpreter.set_tensor(0, img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])  # get tensor  x(1, 25200, 7)
        return output_data

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess."""
        # Resize
        img = cv2.resize(img, (self.height, self.width))
        # BGR -> RGB
        img = img[:, :, [2, 1, 0]]
        # Normalize
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def postprocess(self, output_data, box_type: str) -> Tuple[np.ndarray]:
        """Postprocess."""
        output_data = output_data[0]
        # xywh
        boxes = output_data[..., :4]
        conf = output_data[..., 4:5]
        cls = np.argmax(output_data[..., 5:], axis=1).astype(np.float32).reshape(-1, 1)

        conf = np.squeeze(conf, axis=1)
        cls = np.squeeze(cls, axis=1)

        if box_type == "xyxy":
            # xywh -> xyxyx
            boxes = self.to_xyxy(boxes)

        # Filter by confidence threshold.
        idxs = np.where(conf > self.conf_thr)
        boxes = boxes[idxs]
        cls = cls[idxs]
        conf = conf[idxs]

        return boxes, conf, cls

    def to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Covert xywh to xyxy."""
        # (x, y) is cordinate fo the center of the box.
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        boxes = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        # [4, n] -> [n, 4]
        boxes = boxes.transpose((1, 0))
        return boxes
