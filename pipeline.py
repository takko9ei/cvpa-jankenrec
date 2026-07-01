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
# (Y luma is left wide open; skin clusters tightly in Cr/Cb regardless of
# brightness.) TUNE PER REAL CAPTURE: a window too wide leaks background,
# too narrow drops parts of the hand.
# NOTE: tightened from the textbook 133-173 / 77-127 box after testing on
# data/paper.png, whose warm cream wall sits near skin in chroma (wall Cr~135
# Cb~116, palm Cr~153 Cb~111). Skin here reads as HIGHER Cr + LOWER Cb than
# the wall, so we raise CR_MIN and lower CB_MAX to reject it. On a background
# that clear
# ly differs from skin, these can be loosened back toward defaults.
CR_MIN = 135      # lower bound of Cr channel (raise -> stricter/redder)
CR_MAX = 173          # upper bound of Cr channel (lower -> stricter)
CB_MIN = 77      # lower bound of Cb channel (raise -> stricter)
CB_MAX = 120          # upper bound of Cb channel (lower -> stricter/less blue)

# --- Morphology (clean up the raw skin mask) ------------------------------
# Close = fill small holes inside the hand; Open = remove small speckle noise.
# Bigger kernel = stronger effect (fills/removes larger regions) but blurs
# the silhouette; too big can merge fingers or eat thin finger tips.
MORPH_CLOSE_KSIZE = 7   # kernel size for morphological closing (odd number)
MORPH_OPEN_KSIZE = 5    # kernel size for morphological opening (odd number)

# --- Hand region area filter (in pixels) ----------------------------------
# split_hands() keeps only connected components whose pixel area is in
# [HAND_AREA_MIN, HAND_AREA_MAX]; smaller = noise, larger = background bleed.
# TUNE PER REAL CAPTURE -- these are ABSOLUTE pixel counts, so they scale with
# resolution. Values below are sized for the high-res debug photos in data/
# (~2.5-2.9 M px total, where one hand is ~0.6-1.0 M px). A 640x480 webcam
# frame is ~10x smaller, so lower BOTH by roughly that factor for live use.
HAND_AREA_MIN = 50000     # ignore blobs smaller than this (noise / desk strips)
HAND_AREA_MAX = 1300000   # ignore blobs larger than this (background bleed)

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

def skin_mask(frame, return_stages=False):
    """Stage 1 — Skin segmentation.

    Convert the BGR frame to a skin/non-skin binary mask using YCrCb
    thresholding, then clean it up with morphology.

    Args:
        frame: BGR image (H, W, 3) uint8.
        return_stages: debug flag. When True, also return the pre-cleanup
            mask so debug.py can compare raw vs. clean side by side. Normal
            callers (main.py) leave this False and get a single mask.

    Returns:
        mask: binary image (H, W) uint8, 255 = skin, 0 = background.
        If return_stages is True: (raw_mask, clean_mask) tuple instead.
    """
    # 1. BGR -> YCrCb. Skin tone forms a compact cluster in the Cr/Cb plane,
    #    which makes it far easier to threshold than in RGB/BGR.
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # 2. Threshold ONLY Cr and Cb. Y (luma/brightness) is left wide open
    #    (0..255) so lighting changes don't knock skin pixels out of range.
    #    cv2.inRange needs per-channel bounds; channel order here is Y,Cr,Cb.
    lower = np.array([0, CR_MIN, CB_MIN], dtype=np.uint8)
    upper = np.array([255, CR_MAX, CB_MAX], dtype=np.uint8)
    raw = cv2.inRange(ycrcb, lower, upper)   # 255 = skin, 0 = background

    # 3. Morphological cleanup with elliptical (rounded) kernels so we don't
    #    carve blocky corners into the hand silhouette.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE))
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_KSIZE, MORPH_OPEN_KSIZE))
    # Close first: dilate-then-erode fills small black holes inside the hand.
    clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
    # Then open: erode-then-dilate removes small white speckles in background.
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, open_kernel)

    if return_stages:
        return raw, clean
    return clean


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
    # 1. Label 8-connected blobs. connectedComponentsWithStats returns four
    #    things: the label count, the per-pixel label image, a per-label stats
    #    table (bbox + area), and per-label centroids. We only need the count,
    #    the label image, and the area column here.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    hands = []
    # 2. Label 0 is ALWAYS the background, so start from 1.
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # 3. Keep only blobs whose area looks like a hand; drop noise (too
        #    small) and background bleed (too large).
        if area < HAND_AREA_MIN or area > HAND_AREA_MAX:
            continue
        # 4. Build a fresh binary mask holding ONLY this component: pixels whose
        #    label == i become 255, everything else 0. No morphology here --
        #    cleanup already happened in skin_mask(); this stage only splits.
        hand_mask = np.uint8(labels == i) * 255
        hands.append(hand_mask)

    # 5. len(hands) is the number of hands found -- not assumed to be fixed.
    return hands


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
    # 1. Distance transform: replace every foreground (hand) pixel with its
    #    distance to the NEAREST background pixel, i.e. to the hand's edge.
    #    DIST_L2 = true Euclidean distance; 5 = the distance-mask size.
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)

    # 2. The pixel with the LARGEST distance-to-edge is the point sitting
    #    deepest inside the hand -> the palm center. minMaxLoc returns the min
    #    value, max value, and their (x, y) locations; we want the max.
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)

    # 3. That max distance IS the radius of the biggest circle that fits fully
    #    inside the hand (the palm's inscribed circle).
    center_xy = (int(max_loc[0]), int(max_loc[1]))
    radius = float(max_val)
    return center_xy, radius


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
