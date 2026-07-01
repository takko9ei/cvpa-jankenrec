"""
main.py — 实时主程序（Step 8 + 时序平滑 + 调试叠加层）

打开摄像头，对每一帧跑完整的传统 CV 流水线：
    skin_mask -> split_hands -> (每只手) palm_center -> crop_forearm
    -> count_fingers -> classify   ==> judge ==> draw

单帧分类会抖动，所以加了时序平滑：跨帧按掌心就近匹配"同一只手"，为每只手维护
最近 N 帧的手势历史，显示时用多数投票的手势（见 update_tracks）。

按 'd' 可切换调试叠加层（draw_debug_overlay）：把每一级流水线中间量都画出来，
用于诊断掌心/半径误判。调试层里的所有中间量都用与 pipeline 相同的公式/常量在
main.py 里重新计算，只做可视化，绝不改动 pipeline 的算法。按 'q' 退出。
"""

import math
import time
from collections import deque, Counter

import cv2
import numpy as np

import pipeline


# --- Camera capture config ------------------------------------------------
# Which camera to open. On macOS an iPhone "Continuity Camera" often grabs
# index 0, pushing the Mac's built-in camera to index 1 (verified with
# camera_test.py: index 0 = iPhone, index 1 = built-in). Change this one number
# if your setup differs, or turn Continuity Camera off on the iPhone.
CAM_INDEX = 1

# Ask the camera for a higher resolution. pipeline's area/size constants
# (HAND_AREA_MIN=50000, DRAW_FONT_SCALE=2.0, ...) were tuned on the ~2500 px
# data/ photos. At a typical default 640x480 a hand's pixel area falls well
# below HAND_AREA_MIN and split_hands would drop it; 1280x720 gives the hand
# enough pixels for those constants to have a chance. (The camera may ignore an
# unsupported size and hand back its own; tune the pipeline constants to match
# whatever resolution you actually run at.)
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# --- Temporal smoothing ---------------------------------------------------
# Per-frame classification jitters (one miscounted finger flips Paper<->Scissors
# for a frame). Instead of the raw current-frame gesture we keep each hand's
# last SMOOTH_WINDOW gestures and DISPLAY the majority vote, so a single bad
# frame is outvoted. Bigger = steadier but slower to react to a real change.
SMOOTH_WINDOW = 15

# Frame-to-frame hand tracking, so each hand keeps its OWN history. A detection
# is matched to the previous hand whose palm center is within this many pixels.
# RESOLUTION-DEPENDENT (sized for 1280x720): raise if a fast-moving hand loses
# its history, lower if two nearby hands get swapped.
MATCH_MAX_DIST = 300

# Keep a hand's track alive through this many frames of NON-detection before
# forgetting it, so a brief 1-2 frame skin-mask dropout doesn't wipe its history.
TRACK_MAX_MISSES = 5

# FPS text style.
FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_SCALE = 0.8
FPS_COLOR = (0, 255, 0)   # BGR green
FPS_THICK = 2

# --- Debug overlay style (press 'd' to toggle) ----------------------------
# Colors/alphas for the diagnostic layer. Everything drawn by draw_debug_overlay
# is recomputed from pipeline's own constants purely for visualization.
DEBUG_MASK_COLOR = (255, 0, 0)        # blue tint over skin-mask pixels
DEBUG_MASK_ALPHA = 0.35
DEBUG_HEATMAP_ALPHA = 0.55            # distance-transform heatmap opacity
DEBUG_BAND_COLOR = (255, 255, 0)      # cyan: palm-center top search band
DEBUG_CUT_COLOR = (255, 0, 255)       # magenta: forearm crop line
DEBUG_PALM_COLOR = (255, 255, 255)    # white: chosen palm center + circle, raw max
DEBUG_INSET_W = 320                   # width (px) of the small raw B/W mask inset
# One BGR color per finger-counting ring (RING_RADIUS_COEFFS order).
DEBUG_RING_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
                     (0, 255, 0), (255, 0, 0), (255, 0, 255)]


def _nearest_track(center, tracks, max_dist):
    """Return the still-unclaimed track whose palm center is closest to `center`
    (within max_dist pixels), or None if none is close enough."""
    best, best_dist = None, max_dist
    for t in tracks:
        if t["matched"]:
            continue
        d = math.hypot(center[0] - t["center"][0], center[1] - t["center"][1])
        if d < best_dist:
            best, best_dist = t, d
    return best


def update_tracks(tracks, detections):
    """Temporal smoothing across frames.

    Match this frame's `detections` (each a dict with center/radius/gesture) to
    the persistent `tracks` by nearest palm center, roll every matched hand's
    gesture history, and drop tracks unseen for too long.

    Returns (tracks, records): `tracks` is the pruned track list to carry into
    the next frame; `records` has one entry per hand seen THIS frame, using the
    majority-vote gesture instead of the jittery current-frame one.
    """
    # Mark every existing track unmatched; matching will flip the ones we pair.
    for t in tracks:
        t["matched"] = False

    for det in detections:
        # Same hand as last frame = nearest previous track within MATCH_MAX_DIST;
        # if none is close, this is a new hand -> start a fresh track.
        t = _nearest_track(det["center"], tracks, MATCH_MAX_DIST)
        if t is None:
            t = {"history": deque(maxlen=SMOOTH_WINDOW)}
            tracks.append(t)
        t["center"] = det["center"]
        t["radius"] = det["radius"]
        t["history"].append(det["gesture"])   # roll in this frame's raw gesture
        t["matched"] = True
        t["missed"] = 0

    # Age tracks with no detection this frame; forget the long-lost ones.
    for t in tracks:
        if not t["matched"]:
            t["missed"] = t.get("missed", 0) + 1
    tracks = [t for t in tracks if t["missed"] <= TRACK_MAX_MISSES]

    # Build display records only for hands actually seen this frame, voting the
    # gesture over the track's recent history (ties: Counter keeps the value
    # seen first, which is fine here).
    records = []
    for t in tracks:
        if t["matched"]:
            voted = Counter(t["history"]).most_common(1)[0][0]
            records.append({"center": t["center"], "radius": t["radius"],
                            "gesture": voted})
    return tracks, records


def draw_debug_overlay(frame, mask, detections):
    """Overlay every pipeline intermediate onto `frame` for diagnosis.

    Each detection carries the hand mask ("hand"), the forearm-cropped mask
    ("cropped"), the palm center/radius, the finger count and the raw gesture.
    All intermediates (distance transform, top search band, crop line, rings)
    are RECOMPUTED here with the same formulas/constants pipeline uses, so the
    picture mirrors exactly what pipeline saw. Nothing here changes pipeline.
    """
    h, w = frame.shape[:2]

    # (1) Skin mask -> translucent blue tint, so holes / bridges are visible.
    blue = np.full_like(frame, DEBUG_MASK_COLOR)
    tinted = cv2.addWeighted(frame, 1.0 - DEBUG_MASK_ALPHA, blue,
                             DEBUG_MASK_ALPHA, 0)
    frame[mask > 0] = tinted[mask > 0]

    angles = np.linspace(0.0, 2.0 * np.pi, pipeline.RING_NUM_SAMPLES,
                         endpoint=False)
    panel = ["DEBUG ON (press d to hide) | blue=mask cyan=search-band "
             "magenta=cut white=palm X=raw-max heat=distXform"]

    for i, det in enumerate(detections):
        hand, cropped = det["hand"], det["cropped"]
        (cx, cy), r = det["center"], det["radius"]

        # (3) Distance-transform heatmap over the hand (SAME transform
        #     palm_center runs). Brightest (red in JET) = deepest interior =
        #     palm-center candidate. Secondary yellow/orange blobs at a finger
        #     base are the false candidates we are hunting.
        dist = cv2.distanceTransform(hand, cv2.DIST_L2, 5)
        norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        hm = hand > 0
        frame[hm] = cv2.addWeighted(frame, 1.0 - DEBUG_HEATMAP_ALPHA, heat,
                                    DEBUG_HEATMAP_ALPHA, 0)[hm]
        # White X = UNCONSTRAINED global max of the distance transform (where the
        # palm center would go with no top-band limit). If the X is on the true
        # palm but the white dot (below) is at a finger base, the band did it.
        _, _, _, raw_max = cv2.minMaxLoc(dist)
        cv2.drawMarker(frame, (int(raw_max[0]), int(raw_max[1])),
                       DEBUG_PALM_COLOR, cv2.MARKER_TILTED_CROSS, 34, 3)

        # (4) palm_center TOP SEARCH BAND (recompute top_y / band_bottom exactly).
        ys, xs = np.where(hand > 0)
        top_y = int(ys.min())
        hand_w = int(xs.max() - xs.min() + 1)
        band_bottom = top_y + int(pipeline.TOP_SEARCH_SCALE * hand_w)
        cv2.line(frame, (0, top_y), (w, top_y), DEBUG_BAND_COLOR, 2)
        cv2.line(frame, (0, band_bottom), (w, band_bottom), DEBUG_BAND_COLOR, 2)

        # (2) crop_forearm CUT LINE (recompute cut_y exactly).
        cut_y = int(cy + r * pipeline.CUT_BELOW_SCALE)
        cv2.line(frame, (0, cut_y), (w, cut_y), DEBUG_CUT_COLOR, 2)

        # (5) The CHOSEN palm center + inscribed circle (the real outputs).
        cv2.circle(frame, (cx, cy), int(r), DEBUG_PALM_COLOR, 2)
        cv2.circle(frame, (cx, cy), 7, DEBUG_PALM_COLOR, -1)

        # (6) Finger-counting rings + inside-hand samples + per-ring counts.
        counts, seqs = pipeline._count_fingers_rings(cropped, (cx, cy), r)
        for j, scale in enumerate(pipeline.RING_RADIUS_COEFFS):
            color = DEBUG_RING_COLORS[j % len(DEBUG_RING_COLORS)]
            ring_r = scale * r
            cv2.circle(frame, (cx, cy), int(ring_r), color, 2)
            rx = np.round(cx + ring_r * np.cos(angles)).astype(int)
            ry = np.round(cy + ring_r * np.sin(angles)).astype(int)
            for k in np.flatnonzero(seqs[j]):
                x, y = int(rx[k]), int(ry[k])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 4, color, -1)

        panel.append("H%d: fingers=%d  %s  r=%.0f  per-ring=%s"
                     % (i + 1, det["fingers"], det["gesture"], r, counts))

    # (1b) Small raw black/white mask inset in the top-right corner.
    inset_h = int(mask.shape[0] * DEBUG_INSET_W / mask.shape[1])
    inset = cv2.cvtColor(cv2.resize(mask, (DEBUG_INSET_W, inset_h)),
                         cv2.COLOR_GRAY2BGR)
    x0 = w - DEBUG_INSET_W - 10
    frame[10:10 + inset_h, x0:x0 + DEBUG_INSET_W] = inset
    cv2.rectangle(frame, (x0, 10), (x0 + DEBUG_INSET_W, 10 + inset_h),
                  (255, 255, 255), 1)

    # Text panel (top-left, under the FPS line), dark backing for contrast.
    y = 60
    for line in panel:
        (tw, th), _ = cv2.getTextSize(line, FPS_FONT, 0.6, 2)
        cv2.rectangle(frame, (8, y - th - 4), (12 + tw, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, line, (10, y), FPS_FONT, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        y += 28
    return frame


def main():
    # Open the selected camera (see CAM_INDEX) and request the capture size.
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("could not open camera index %d" % CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    tracks = []               # persists across frames; one dict per tracked hand
    debug_mode = False        # toggled by the 'd' key
    prev_time = time.time()   # timestamp of the previous frame, for FPS.

    while True:
        ok, frame = cap.read()
        if not ok:
            # Camera hiccup / end of stream — stop the loop.
            break

        # 1. Skin segmentation (kept in `mask` so the debug overlay can reuse it).
        mask = pipeline.skin_mask(frame)

        # 2. Per hand: locate palm, cut forearm, count fingers, name gesture.
        #    Stash the hand + cropped masks so the debug overlay can visualize
        #    the intermediates (these are cheap references; no extra compute).
        detections = []
        for hand in pipeline.split_hands(mask):
            center, radius = pipeline.palm_center(hand)
            cropped = pipeline.crop_forearm(hand, center, radius)
            fingers = pipeline.count_fingers(cropped, center, radius)
            gesture = pipeline.classify(fingers)
            detections.append({"center": center, "radius": radius,
                               "gesture": gesture, "fingers": fingers,
                               "hand": hand, "cropped": cropped})

        # 3. Temporal smoothing: match to previous hands, majority-vote gesture.
        tracks, records = update_tracks(tracks, detections)

        # 4. Judge the SMOOTHED gestures, attach each result to its hand.
        if records:
            results = pipeline.judge([rec["gesture"] for rec in records])
            for rec, result in zip(records, results):
                rec["result"] = result

        # 5. Draw the gesture + result labels (and palm circles) onto the frame.
        frame = pipeline.draw(frame, records)

        # 6. Optional diagnostic layer on top (pipeline intermediates).
        if debug_mode:
            frame = draw_debug_overlay(frame, mask, detections)

        # ---- FPS overlay: 1 / (time since the previous frame) ------------
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FPS_FONT, FPS_SCALE,
                    FPS_COLOR, FPS_THICK, cv2.LINE_AA)

        cv2.imshow("jankenrec", frame)

        # Keys: q = quit, d = toggle the debug overlay.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug_mode = not debug_mode

    # Always release the camera and close windows on exit.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
