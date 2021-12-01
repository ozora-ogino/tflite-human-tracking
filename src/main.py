#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

import argparse
import os
import time

import cv2
import numpy as np

from detect import Detect
from streams import VideoStream
from tracker import Tracker
from utils import direction_config


def _detect_person(
    detect: Detect,
    frame: np.ndarray,
    confidence: float,
    iou_threshold: float,
) -> np.ndarray:
    """Detect person objects in a frame.

    Returns:
        np.ndarray: Array like [xyxy, score].
    """

    # Detect objects in the frame.
    boxes, scores, class_idx = detect.detect(frame)

    # NMS
    idx = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou_threshold)
    boxes = boxes[idx]
    scores = scores[idx]
    class_idx = class_idx[idx]

    # Filter only person object (class index = 0).
    person_idx = np.where(class_idx == 0)[0]
    boxes = boxes[person_idx]
    scores = scores[person_idx]

    # Scale boxes by frame size.
    H, W = frame.shape[:2]
    boxes = detect.to_xyxy(boxes) * np.array([W, H, W, H])

    # dets:  [xmin, ymin, xmax, ymax, score]
    dets = np.concatenate([boxes.astype(int), scores.reshape(-1, 1)], axis=1)
    return dets


def main(
    src: str,
    dest: str,
    model: str,
    video_fmt: str,
    confidence: float,
    iou_threshold: float,
    direction: str,
):
    """Track human objects and count the number of human.

    Args:
        src (str): Source video.
        dest (str): Directory to save results.
        model (str): Path to tflite weight.
        confidence (float): Confidence threshold.
        iou_threshold (float): IoU threshold for NMS.
    """
    if not os.path.exists(dest):
        os.mkdir(dest)

    # The line to count.
    border = [(0, 500), (1920, 500)]
    direction = direction_config.get(direction)
    tracker = Tracker(border, direction)
    detect = Detect(model, confidence)
    stream = VideoStream(src)
    writer = None

    total_frames = len(stream)
    if total_frames:
        print(f"Total frames: {len(stream)}")

    while True:
        # Read the next frame from stream.
        is_finish, frame = stream.next()

        if not is_finish:
            break

        start = time.time()
        dets = _detect_person(detect, frame, confidence, iou_threshold)
        end = time.time()

        # Update tracker and draw bounding boxes in frame.
        frame = tracker.update(frame, dets)

        # Executed only first time.
        if writer is None:
            # Initialize video writer.
            model_name = os.path.basename(model).split(".")[0]
            video_name = os.path.basename(src).split(".")[0]
            codecs = {"mp4": "MP4V", "avi": "MJPG"}
            basename = f"{video_name}_{model_name}"
            output_video = os.path.join(dest, f"{basename}.{video_fmt}")
            fourcc = cv2.VideoWriter_fourcc(*codecs[video_fmt])
            writer = cv2.VideoWriter(output_video, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            # Estimate total time.
            second_per_frame = end - start
            print(f"Computation time per a frame: {second_per_frame:.4f} seconds")
            print(f"Estimated total time: {second_per_frame * total_frames:.4f}")

        # Save frame as an image and video.
        cv2.imwrite(os.path.join(dest, f"{basename}.jpg"), frame)
        writer.write(frame)

    writer.release()
    stream.release()
    print("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Path to video source.", default="./data/TownCentreXVID.mp4")
    parser.add_argument("--dest", help="Path to output directory", default="./outputs/")
    parser.add_argument("--model", help="Path to YOLOv5 tflite file", default="./models/yolov5n6-fp16.tflite")
    parser.add_argument("--video-fmt", help="Format of output video file.", choices=["mp4", "avi"], default="mp4")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="IoU threshold for NMS.")
    parser.add_argument("--direction", default=None, choices=direction_config.keys())
    args = vars(parser.parse_args())
    main(**args)
