#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino


def is_intersect(A, B, C, D):
    """Return true if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    """Counter clock wise."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
