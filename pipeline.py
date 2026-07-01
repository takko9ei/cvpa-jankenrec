"""
pipeline.py — 传统 CV 猜拳识别核心流水线（Step 0 脚手架）

只用 OpenCV + NumPy。禁止任何机器学习 / 预训练模型 / cv2.convexityDefects。
本文件目前只有常量区 + 8 个空函数骨架，尚未实现任何算法逻辑。

Pipeline order:
    skin_mask -> split_hands -> palm_center -> crop_forearm
    -> count_fingers -> classify -> judge -> draw
"""

import cv2
import numpy as np


# =============================================================================
# CONSTANTS — 所有可调参数集中在这里（占位值，后续各阶段再调）
# All tunable parameters live here. Change values ONLY in this block.
# =============================================================================

# --- Skin segmentation (YCrCb thresholding) -------------------------------
# Lower / upper bounds for the Cr and Cb channels used to threshold skin.
# (Y is usually left wide open; skin clusters tightly in Cr/Cb.)
CR_MIN = 133          # lower bound of Cr channel
CR_MAX = 173          # upper bound of Cr channel
CB_MIN = 77           # lower bound of Cb channel
CB_MAX = 127          # upper bound of Cb channel

# --- Morphology (clean up the raw skin mask) ------------------------------
# Close = fill small holes inside the hand; Open = remove small speckle noise.
MORPH_CLOSE_KSIZE = 7   # kernel size for morphological closing (odd number)
MORPH_OPEN_KSIZE = 5    # kernel size for morphological opening (odd number)

# --- Hand region area filter (in pixels) ----------------------------------
# Connected components smaller/larger than this are not treated as a hand.
HAND_AREA_MIN = 3000     # ignore blobs smaller than this (noise)
HAND_AREA_MAX = 200000   # ignore blobs larger than this (background bleed)

# --- Finger counting: concentric ring sampling ----------------------------
# We sample circles at these multiples of the palm inscribed-circle radius.
# Each factor > 1.0 pushes the ring outward past the palm into the fingers.
RING_RADIUS_COEFFS = [1.4, 1.7, 2.0, 2.3]

# --- Classification thresholds (finger count -> gesture) ------------------
# 0-1 -> Rock, 2-3 -> Scissors, >=4 -> Paper
ROCK_MAX_FINGERS = 1        # <= this many fingers => Rock
SCISSORS_MAX_FINGERS = 3    # <= this many fingers (and > ROCK_MAX) => Scissors
                            # anything greater => Paper


# =============================================================================
# PIPELINE FUNCTIONS — 8 个阶段（当前仅骨架）
# =============================================================================

def skin_mask(frame):
    """Stage 1 — Skin segmentation.

    Convert the BGR frame to a skin/non-skin binary mask using YCrCb
    thresholding, then clean it up with morphology.

    Args:
        frame: BGR image (H, W, 3) uint8.

    Returns:
        mask: binary image (H, W) uint8, 255 = skin, 0 = background.
    """
    raise NotImplementedError


def split_hands(mask):
    """Stage 2 — Split the mask into one sub-mask per hand.

    Find connected components in the skin mask and keep those whose area
    falls within [HAND_AREA_MIN, HAND_AREA_MAX]. Each kept component becomes
    its own isolated hand mask.

    Args:
        mask: binary skin mask (H, W) uint8 from skin_mask().

    Returns:
        list[hand_mask]: list of binary masks (H, W) uint8, one per hand
        (0, 1, or 2 entries in practice).
    """
    raise NotImplementedError


def palm_center(hand_mask):
    """Stage 3 — Locate the palm center and its inscribed-circle radius.

    Find the point deepest inside the hand blob (largest distance to the
    boundary). That point is the palm center; its distance to the boundary
    is the radius of the largest circle that fits inside the palm.

    Args:
        hand_mask: binary mask (H, W) uint8 of a single hand.

    Returns:
        (center_xy, radius): center_xy is an (x, y) int tuple, radius is a
        float (pixels).
    """
    raise NotImplementedError


def crop_forearm(hand_mask, center_xy, radius):
    """Stage 4 — Cut off the wrist / forearm below the palm.

    Using the palm center and radius, remove the arm portion so only the
    hand (palm + fingers) remains for finger counting.

    Args:
        hand_mask: binary mask (H, W) uint8 of a single hand.
        center_xy: (x, y) palm center from palm_center().
        radius: palm inscribed-circle radius from palm_center().

    Returns:
        hand_mask: binary mask (H, W) uint8 with the forearm removed.
    """
    raise NotImplementedError


def count_fingers(hand_mask, center_xy, radius):
    """Stage 5 — Count extended fingers via concentric-ring sampling.

    Sample circles around the palm center at RING_RADIUS_COEFFS multiples of
    the palm radius. On each ring, count how many separate skin arcs it
    crosses (each extended finger pierces the ring once). Aggregate the rings
    into a single finger count. Must be hand-written geometry — no
    cv2.convexityDefects.

    Args:
        hand_mask: binary mask (H, W) uint8 (forearm already cropped).
        center_xy: (x, y) palm center.
        radius: palm inscribed-circle radius.

    Returns:
        int: number of extended fingers.
    """
    raise NotImplementedError


def classify(finger_count):
    """Stage 6 — Map a finger count to a gesture name.

    0-1 -> "Rock", 2-3 -> "Scissors", >=4 -> "Paper"
    (see ROCK_MAX_FINGERS / SCISSORS_MAX_FINGERS).

    Args:
        finger_count: int from count_fingers().

    Returns:
        str: one of "Rock", "Scissors", "Paper".
    """
    raise NotImplementedError


def judge(shapes):
    """Stage 7 — Decide the winner(s) from the detected gestures.

    Rules: Rock > Scissors, Scissors > Paper, Paper > Rock.

    Args:
        shapes: list of gesture-name strings (typically 2 hands).

    Returns:
        list[result]: per-hand outcome (e.g. "Win" / "Lose" / "Draw"),
        aligned with the input order.
    """
    raise NotImplementedError


def draw(frame, records):
    """Stage 8 — Draw results back onto the original frame.

    Overlay per-hand annotations (palm center, gesture name, win/lose, etc.)
    onto a copy of the frame for display.

    Args:
        frame: original BGR frame (H, W, 3) uint8.
        records: per-hand info collected through the pipeline (centers,
            gestures, results, ...).

    Returns:
        frame: BGR image (H, W, 3) uint8 with annotations drawn on.
    """
    raise NotImplementedError
