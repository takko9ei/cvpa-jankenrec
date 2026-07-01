"""
main.py — 实时主程序（Step 8 + 时序平滑）

打开摄像头，对每一帧跑完整的传统 CV 流水线：
    skin_mask -> split_hands -> (每只手) palm_center -> crop_forearm
    -> count_fingers -> classify   ==> judge ==> draw

单帧分类会抖动（某一帧手指数偶尔数错，Paper/Scissors 来回跳）。所以加了一层
时序平滑：跨帧按掌心就近匹配"同一只手"，为每只手维护最近 N 帧的手势历史，
显示时用多数投票的手势，而不是当前帧的原始结果。平滑逻辑在 update_tracks()，
所有算法仍在 pipeline.py；main.py 只负责取帧、平滑编排、显示。
"""

import math
import time
from collections import deque, Counter

import cv2

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


def main():
    # Open the selected camera (see CAM_INDEX) and request the capture size.
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("could not open camera index %d" % CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    tracks = []               # persists across frames; one dict per tracked hand
    prev_time = time.time()   # timestamp of the previous frame, for FPS.

    while True:
        ok, frame = cap.read()
        if not ok:
            # Camera hiccup / end of stream — stop the loop.
            break

        # 1-2. Detect this frame's hands (raw, unsmoothed): one record per hand.
        detections = []
        for hand in pipeline.split_hands(pipeline.skin_mask(frame)):
            center, radius = pipeline.palm_center(hand)
            cropped = pipeline.crop_forearm(hand, center, radius)
            fingers = pipeline.count_fingers(cropped, center, radius)
            gesture = pipeline.classify(fingers)
            detections.append({"center": center, "radius": radius,
                               "gesture": gesture})

        # 3. Temporal smoothing: match to previous hands, majority-vote gesture.
        tracks, records = update_tracks(tracks, detections)

        # 4. Judge the SMOOTHED gestures, attach each result to its hand. Skip
        #    when no hand is present (judge expects at least one gesture).
        if records:
            results = pipeline.judge([rec["gesture"] for rec in records])
            for rec, result in zip(records, results):
                rec["result"] = result

        # 5. Draw the gesture + result labels (and palm circles) onto the frame.
        frame = pipeline.draw(frame, records)

        # ---- FPS overlay: 1 / (time since the previous frame) ------------
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FPS_FONT, FPS_SCALE,
                    FPS_COLOR, FPS_THICK, cv2.LINE_AA)

        cv2.imshow("jankenrec", frame)

        # Quit when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Always release the camera and close windows on exit.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
