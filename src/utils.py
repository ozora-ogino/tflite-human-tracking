#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

from typing import List, Tuple

# Direction configuration.
# 'direction_key': (x, y)
direction_config = {
    "right": (True, None),
    "top": (None, True),
    "left": (False, None),
    "bottom": (None, False),
    "righttop": (True, True),
    "rightbottom": (True, False),
    "lefttop": (False, True),
    "leftbottom": (False, False),
}


def check_direction(
    current_center: List[int],
    prev_center: List[int],
    direction: Tuple[bool] = None,
):
    """If direction is specified, check whether the direction of target is correct.

    Args:
        prev_center(List[int]):
        current_center(List[int]):
        direction(Tuple[bool]):
    """
    if direction is None:
        return True

    # True if the direction is right.
    direction_x = (current_center[0] - prev_center[0]) > 0
    # True if the direction is top.
    direction_y = (current_center[1] - prev_center[1]) < 0

    x_is_true = direction[0] is None or direction_x is direction[0]
    y_is_true = direction[1] is None or direction_y is direction[1]

    return x_is_true and y_is_true


def is_intersect(A, B, C, D):
    """Return true if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    """Counter clock wise."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
