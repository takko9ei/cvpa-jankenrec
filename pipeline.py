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
CB_MAX = 130          # upper bound of Cb channel (lower -> stricter/less blue)

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

# --- Palm center search band (fist-vs-forearm fix) ------------------------
# palm_center() picks the deepest interior point via distance transform. For an
# OPEN hand the palm is the widest interior region, so the transform peaks there.
# But a closed FIST is nearly as thick as the forearm below it, so the global
# maximum can slide DOWN onto the wrist and mislocate the palm (which then breaks
# crop_forearm). Fix with the frontal-hand assumption (fingers/fist UP, forearm
# DOWN): only search inside a horizontal band that starts at the hand's TOPMOST
# white pixel and runs down this many times the hand's bounding-box WIDTH.
# Because it scales with hand width, the band self-adjusts: an OPEN hand (spread
# fingers -> wider bbox) gets a deeper band to reach its lower palm, a FIST
# (narrow bbox) gets a shallow band that stops before the forearm widens.
# 0.7 is a fairly tight window on the test images: RAISE it and a fist's center
# can drop onto the wrist; LOWER it and an open hand's center rides up toward the
# finger bases. Retune if hands sit very differently in the frame.
TOP_SEARCH_SCALE = 0.78

# Palm center = the CENTROID of the deepest region (every in-band pixel whose
# distance transform is >= this fraction of the in-band maximum), NOT the single
# brightest pixel. Averaging the high-value plateau makes the center robust to a
# sharp secondary distance-transform spike at a finger base, which otherwise
# yanks the argmax onto a knuckle (the scissors mis-localization). Lower = wider
# plateau (steadier but can drift toward a neighbouring blob); higher = tighter
# (closer to the raw argmax).
CENTROID_FRAC = 0.7

# --- Forearm cropping -----------------------------------------------------
# Frontal hand assumption: fingers point up, wrist/forearm points down. Any
# hand pixel more than this many palm-radii BELOW the palm center is treated
# as forearm and blacked out. Larger = keep more (safer for the palm heel),
# smaller = cut more aggressively (risk clipping the palm).
CUT_BELOW_SCALE = 1.2

# --- Finger counting: concentric ring sampling ----------------------------
# count_fingers() samples concentric rings at these multiples of the palm
# inscribed-circle radius. Each scale > 1.0 pushes the ring OUTWARD past the
# palm so it crosses the extended fingers. We sample SEVERAL rings and take the
# MODE of their per-ring finger counts, so a single badly-placed ring can't
# decide the result: too small (< ~1.2) and it still sits inside the palm;
# too big (> ~2.3) and it can shoot past short fingertips. Add/shift scales if
# fingers of very different lengths get missed.
RING_RADIUS_COEFFS = [1.7, 1.9, 2.1, 2.3, 2.5]

# Number of points sampled evenly around EACH ring (360 = one per ~1 degree).
# More points = finer angular resolution (less chance of skipping a thin finger
# or a narrow gap between fingers) but a little slower. The sampled 0/1 sequence
# is treated as CIRCULAR: index 359 wraps back to index 0.
RING_NUM_SAMPLES = 360

# Finger-width gate (in DEGREES of arc). Every extended finger crosses a ring as
# one contiguous skin arc, but so do non-fingers: sampling noise makes tiny
# arcs, and a closed FIST (or the palm heel) makes WIDE arcs because the ring
# runs along its rounded edge for a long angular span. A real finger is a thin
# protrusion, so its arc width is bounded. We therefore only count an arc as a
# finger when its angular width is within [MIN, MAX]. Measured on the test set:
# finger arcs span ~9-35 deg, fist/palm bulk ~43-82 deg, noise <7 deg -- so this
# band cleanly separates them. Because it is measured in DEGREES it is
# scale-invariant (works for big and small hands alike). RAISE MAX if long/thick
# fingers get dropped; LOWER it if a fist still leaks arcs; RAISE MIN to kill
# more speckle.
FINGER_ARC_MIN_DEG = 8
FINGER_ARC_MAX_DEG = 40

# --- Classification thresholds (finger count -> gesture) ------------------
# 0-1 -> Rock, 2-3 -> Scissors, >=4 -> Paper
ROCK_MAX_FINGERS = 1        # <= this many fingers => Rock
SCISSORS_MAX_FINGERS = 3    # <= this many fingers (and > ROCK_MAX) => Scissors
                            # anything greater => Paper

# --- Result overlay style (draw) ------------------------------------------
# Text size / stroke for the gesture+result label draw() writes near each palm.
# RESOLUTION-DEPENDENT (like the area filter): these are sized for the high-res
# data/ photos (~2500 px wide); for a ~640 px live webcam frame divide both by
# roughly 3-4. Result -> BGR color: Win green, Lose red, Draw/Pending gray.
DRAW_FONT_SCALE = 2.0
DRAW_THICKNESS = 4
RESULT_COLORS = {
    "Win":     (0, 255, 0),      # green
    "Lose":    (0, 0, 255),      # red
    "Draw":    (150, 150, 150),  # gray
    "Pending": (150, 150, 150),  # gray (single hand, no opponent yet)
}


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

    The search is restricted to an upper band of the hand (see
    TOP_SEARCH_SCALE): a closed fist is nearly as thick as the forearm, so an
    unconstrained maximum can slide down onto the wrist. Frontal-hand
    assumption: fingers/fist point up, forearm points down.

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

    # 2. Constrain the palm search to the UPPER band of the hand. Find the
    #    topmost white pixel (smallest y) and the hand's bounding-box width;
    #    the band runs from top_y down TOP_SEARCH_SCALE * width. This keeps the
    #    center on the palm/fist instead of letting it slide onto the forearm
    #    (a fist is nearly as thick as the wrist, so the raw max can drop down).
    ys, xs = np.where(hand_mask > 0)
    top_y = int(ys.min())                     # highest white pixel (smallest y)
    hand_w = int(xs.max() - xs.min() + 1)     # hand bounding-box width (px)
    band_bottom = top_y + int(TOP_SEARCH_SCALE * hand_w)

    # 3. Blank out the distance transform BELOW the band so minMaxLoc can only
    #    pick a candidate inside [top_y, band_bottom). A band_bottom past the
    #    image edge just leaves the whole hand in play (empty slice, no-op).
    dist[band_bottom:, :] = 0

    # 4. Palm center = CENTROID of the deepest region, not the single argmax
    #    pixel. Take the in-band maximum distance, then average the (x, y) of
    #    every pixel whose distance is >= CENTROID_FRAC of it. That plateau is
    #    dominated by the broad palm blob, so a sharp secondary spike at a
    #    finger base can't hijack the center the way the raw argmax could.
    max_val = float(dist.max())
    ys_hi, xs_hi = np.where(dist >= CENTROID_FRAC * max_val)
    center_xy = (int(xs_hi.mean()), int(ys_hi.mean()))

    # 5. Radius = that in-band maximum distance = the palm's inscribed-circle
    #    radius (the biggest circle that fits inside the hand at the palm).
    radius = max_val
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
    # Fingers point up, the arm points down (frontal-hand assumption). So every
    # hand pixel far enough BELOW the palm center is wrist/forearm. Only the
    # center's y matters for a horizontal cut line; everything from this row
    # downward gets blacked out.
    _, cy = center_xy
    cut_y = int(cy + radius * CUT_BELOW_SCALE)
    result = hand_mask.copy()
    result[cut_y:, :] = 0     # rows below the cut line -> background (0)
    return result


def count_fingers(hand_mask, center_xy, radius):
    """Stage 5 — Count extended fingers via concentric-ring sampling.

    Sample circles around the palm center at RING_RADIUS_COEFFS multiples of the palm
    radius. On each ring, walk the circle once and record inside-hand (1) vs
    background (0) at RING_NUM_SAMPLES points. Every extended finger shows up as
    one contiguous arc of 1s, so counting those arcs (0->1 rising edges) on that
    (circular) sequence counts the fingers the ring crosses — but only arcs of
    finger-like width are kept, so a closed fist's wide edge-arcs and sampling
    noise don't inflate the count (see _count_ring_arcs). Take the MODE of the
    per-ring counts as the answer. Hand-written geometry only — no
    cv2.convexityDefects.

    Args:
        hand_mask: binary mask (H, W) uint8 (forearm already cropped).
        center_xy: (x, y) palm center.
        radius: palm inscribed-circle radius.

    Returns:
        int: number of extended fingers. (debug.py calls the shared helper
        _count_fingers_rings directly to also get the per-ring detail.)
    """
    counts, _ = _count_fingers_rings(hand_mask, center_xy, radius)

    # Aggregate the per-ring counts into one number. The MODE (most frequent
    # value) is robust: rings that land badly (still inside the palm -> 0, or
    # past the fingertips -> 0) are outvoted by the rings that sit across the
    # fingers. If two values tie for most-frequent, fall back to the rounded
    # median of the raw counts (a safe middle value).
    if not counts:
        return 0
    values, freqs = np.unique(counts, return_counts=True)
    top = freqs.max()
    modes = values[freqs == top]
    if len(modes) == 1:
        return int(modes[0])
    return int(np.median(counts))


def _count_fingers_rings(hand_mask, center_xy, radius):
    """Shared core for count_fingers: return (per_ring_counts, per_ring_seqs).

    per_ring_counts[i] is the finger count for RING_RADIUS_COEFFS[i]; per_ring_seqs[i]
    is that ring's raw 0/1 circular sequence (kept so debug.py can visualize
    exactly what was sampled). Factored out so the live path and the debug path
    sample identically.
    """
    cx, cy = center_xy
    h, w = hand_mask.shape[:2]

    # One angle per sample, evenly spaced over the full circle. endpoint=False
    # so we DON'T sample 360 deg (a duplicate of 0 deg); the wrap is handled by
    # treating the sequence as circular below.
    angles = np.linspace(0.0, 2.0 * np.pi, RING_NUM_SAMPLES, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    counts, seqs = [], []
    for scale in RING_RADIUS_COEFFS:
        ring_r = scale * radius
        # Pixel coordinates of the sample points around this ring.
        xs = np.round(cx + ring_r * cos_a).astype(int)
        ys = np.round(cy + ring_r * sin_a).astype(int)

        # Look up the mask at each point. Points that fall OUTSIDE the image are
        # treated as background (0) so the ring can safely poke past the frame.
        seq = np.zeros(RING_NUM_SAMPLES, dtype=np.uint8)
        on = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        seq[on] = (hand_mask[ys[on], xs[on]] > 0).astype(np.uint8)

        # Turn the circular 0/1 sequence into a finger count: one arc (run of
        # 1s) per finger, keeping only finger-width arcs (see _count_ring_arcs).
        counts.append(_count_ring_arcs(seq))
        seqs.append(seq)

    return counts, seqs


def _count_ring_arcs(seq):
    """Count finger-width skin arcs on one ring's circular 0/1 sequence.

    Each run of consecutive 1s is one skin arc the ring crossed (its start is a
    0->1 rising edge). We keep only arcs whose angular width lies in
    [FINGER_ARC_MIN_DEG, FINGER_ARC_MAX_DEG]: narrower ones are sampling noise,
    wider ones are palm/fist bulk, neither is an extended finger.

    Args:
        seq: (RING_NUM_SAMPLES,) uint8 array of 0/1 around the ring. The array
            is CIRCULAR — index n-1 is adjacent to index 0.

    Returns:
        int: number of finger-width arcs.
    """
    n = len(seq)
    min_len = FINGER_ARC_MIN_DEG / 360.0 * n   # arc-width band, in samples
    max_len = FINGER_ARC_MAX_DEG / 360.0 * n
    if not seq.any():
        return 0    # ring entirely in background (radius overshot the hand)
    if seq.all():
        return 0    # ring entirely inside the hand (still one big 360 deg arc)

    # Rotate so the sequence STARTS on a background (0) sample. After this no run
    # of 1s straddles the wrap seam, so a single left-to-right scan sees every
    # arc exactly once — this is how the circular first/last connection is
    # handled without double-counting or missing the arc crossing 0 deg.
    shift = int(np.flatnonzero(seq == 0)[0])
    rolled = np.roll(seq, -shift)

    fingers = 0
    run = 0                       # length of the current run of 1s
    for v in rolled:
        if v:
            run += 1
        elif run:                 # a run just ended -> test its width
            if min_len <= run <= max_len:
                fingers += 1
            run = 0
    if run and min_len <= run <= max_len:   # trailing run ending at the array end
        fingers += 1
    return fingers


def classify(finger_count):
    """Stage 6 — Map a finger count to a gesture name.

    0-1 -> "Rock", 2-3 -> "Scissors", >=4 -> "Paper"
    (see ROCK_MAX_FINGERS / SCISSORS_MAX_FINGERS).

    Args:
        finger_count: int from count_fingers().

    Returns:
        str: one of "Rock", "Scissors", "Paper".
    """
    # Thresholds live in the constants block (ROCK_MAX_FINGERS = 1,
    # SCISSORS_MAX_FINGERS = 3). Boundaries: 1 -> Rock, 3 -> Scissors, 4 ->
    # Paper. A closed fist reads 0-1 fingers (Rock); two/three extended fingers
    # are Scissors; a fully open hand is Paper.
    if finger_count <= ROCK_MAX_FINGERS:
        return "Rock"
    elif finger_count <= SCISSORS_MAX_FINGERS:
        return "Scissors"
    else:
        return "Paper"


def judge(shapes):
    """Stage 7 — Decide the winner(s) from the detected gestures.

    Rules: Rock > Scissors, Scissors > Paper, Paper > Rock.

    Args:
        shapes: list of gesture-name strings (typically 2 hands).

    Returns:
        list[result]: per-hand outcome ("Win" / "Lose" / "Draw" / "Pending"),
        aligned with the input order.
    """
    # "X beats Y": the classic cycle Rock > Scissors > Paper > Rock. A plain
    # dict keeps the rule readable — no clever arithmetic.
    beats = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}

    # A single hand has no opponent, so its result is undecided. We return
    # "Pending" (rather than "Draw") to make that state explicit for the live
    # loop / overlay.
    if len(shapes) == 1:
        return ["Pending"]

    # How many DISTINCT gestures are on the table decides everything.
    kinds = set(shapes)

    # All hands identical (1 kind) OR all three gestures present at once -> no
    # single winner, everyone draws. (Why 3-kinds is a draw: see the writeup.)
    if len(kinds) == 1 or len(kinds) == 3:
        return ["Draw"] * len(shapes)

    # Exactly 2 distinct gestures: one of them beats the other. Pick the winning
    # gesture, then label every hand by whether it holds it.
    a, b = kinds
    winner = a if beats[a] == b else b
    return ["Win" if s == winner else "Lose" for s in shapes]


def draw(frame, records):
    """Stage 8 — Draw results back onto the original frame.

    Overlay per-hand annotations (palm center, gesture name, win/lose, etc.)
    onto a copy of the frame for display.

    Args:
        frame: original BGR frame (H, W, 3) uint8.
        records: list of per-hand dicts, each with keys "center" ((x, y) palm
            center), "gesture" (str), "result" (str from judge()), and optional
            "radius" (float, palm circle to draw / place the label above).

    Returns:
        frame: BGR image (H, W, 3) uint8 with annotations drawn on.
    """
    out = frame.copy()
    for rec in records:
        cx, cy = rec["center"]
        gesture, result = rec["gesture"], rec["result"]
        # One color carries the outcome everywhere: circle + text share it.
        color = RESULT_COLORS.get(result, (150, 150, 150))
        radius = int(rec.get("radius", 0))

        # Optional palm circle in the result color, so each hand is easy to spot.
        if radius > 0:
            cv2.circle(out, (cx, cy), radius, color, DRAW_THICKNESS)

        # Label: "<gesture>: <result>", centered horizontally on the palm and
        # placed just above the circle. getTextSize lets us center it; clamp the
        # y so the text never runs off the top of the frame.
        label = "%s: %s" % (gesture, result)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      DRAW_FONT_SCALE, DRAW_THICKNESS)
        tx = cx - tw // 2
        ty = max(th + 5, cy - radius - 15)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    DRAW_FONT_SCALE, color, DRAW_THICKNESS, cv2.LINE_AA)
    return out
