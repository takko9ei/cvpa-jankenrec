"""
test_script_5.py — Stage 5 展示：同心环计数手指 (count_fingers)

依次跑 skin_mask -> split_hands -> palm_center -> crop_forearm -> count_fingers。
窗口上方是上一步（test_script_4）下方的结果：切掉前臂后的掩膜 + 红色掌心圆。
下方在同一张图上叠加画出用于计数的几个同心环（RING_RADIUS_COEFFS 的每个
半径一个圈），并在旁边写出该手的计数结果。按任意键关闭窗口。
"""

import cv2
import numpy as np

import pipeline


# Same source image as the earlier test scripts.
IMAGE_PATH = "data/rock_paper.png"

# Both panels are resized to this width before stacking.
DISPLAY_WIDTH = 800

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.9
LABEL_COLOR = (0, 255, 0)   # BGR green
LABEL_THICK = 2

# Palm center / circle draw style, same convention as test_script_4.py.
PALM_COLOR = (0, 0, 255)   # red, matches test_script_4.py's bottom-panel fix
PALM_CIRCLE_THICK = 3
PALM_DOT_RADIUS = 8
RADIUS_TEXT_SCALE = 0.9
RADIUS_TEXT_THICK = 2

# One distinct BGR color per counting ring (RING_RADIUS_COEFFS order).
RING_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
               (0, 255, 0), (255, 0, 0), (255, 0, 255)]
RING_THICK = 2

# Style for the "fingers=N" result label written beside each hand.
COUNT_TEXT_SCALE = 2.2
COUNT_TEXT_COLOR = (0, 0, 255)   # red
COUNT_TEXT_THICK = 5


def _resize_to_width(img, width):
    """Resize an image to a fixed width, keeping its aspect ratio."""
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))))


def _draw_palm_circle(canvas, center, radius):
    """Draw the palm center dot + inscribed circle + radius label onto canvas
    (same visual as test_script_4.py's bottom panel)."""
    cx, cy = center
    cv2.circle(canvas, (cx, cy), int(radius), PALM_COLOR, PALM_CIRCLE_THICK)
    cv2.circle(canvas, (cx, cy), PALM_DOT_RADIUS, PALM_COLOR, -1)
    cv2.putText(canvas, "r=%.0f" % radius, (cx + 15, cy - 15), LABEL_FONT,
                RADIUS_TEXT_SCALE, PALM_COLOR, RADIUS_TEXT_THICK, cv2.LINE_AA)


def _draw_rings_and_count(canvas, center, radius, fingers):
    """Draw the concentric counting rings (pipeline.RING_RADIUS_COEFFS) and
    write the finger count beside the hand."""
    cx, cy = center
    h, w = canvas.shape[:2]

    for i, scale in enumerate(pipeline.RING_RADIUS_COEFFS):
        color = RING_COLORS[i % len(RING_COLORS)]
        cv2.circle(canvas, (cx, cy), int(scale * radius), color, RING_THICK)

    # Place the result just outside the outermost ring, on the hand's right.
    outer_r = max(pipeline.RING_RADIUS_COEFFS) * radius
    label = "fingers=%d" % fingers
    (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, COUNT_TEXT_SCALE,
                                   COUNT_TEXT_THICK)
    tx = min(int(cx + outer_r) + 20, w - tw - 10)
    ty = cy
    cv2.putText(canvas, label, (tx, ty), LABEL_FONT, COUNT_TEXT_SCALE,
                COUNT_TEXT_COLOR, COUNT_TEXT_THICK, cv2.LINE_AA)


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError("could not read " + IMAGE_PATH)

    mask = pipeline.skin_mask(img)        # Stage 1
    hands = pipeline.split_hands(mask)    # Stage 2

    records = []   # (center, radius, cropped_mask, fingers) per hand
    for hand_mask in hands:
        center, radius = pipeline.palm_center(hand_mask)              # Stage 3
        cropped = pipeline.crop_forearm(hand_mask, center, radius)    # Stage 4
        fingers = pipeline.count_fingers(cropped, center, radius)     # Stage 5
        records.append((center, radius, cropped, fingers))

    cropped_union = np.zeros(mask.shape, dtype=np.uint8)
    for _, _, cropped, _ in records:
        cropped_union = cv2.bitwise_or(cropped_union, cropped)

    # Top panel: the previous step's result (cropped mask + palm circles).
    top_vis = cv2.cvtColor(cropped_union, cv2.COLOR_GRAY2BGR)
    for center, radius, _, _ in records:
        _draw_palm_circle(top_vis, center, radius)

    # Bottom panel: same base image, with counting rings + result overlaid.
    bottom_vis = top_vis.copy()
    for center, radius, _, fingers in records:
        _draw_rings_and_count(bottom_vis, center, radius, fingers)

    top = _resize_to_width(top_vis, DISPLAY_WIDTH)
    bottom = _resize_to_width(bottom_vis, DISPLAY_WIDTH)

    cv2.putText(top, "Stage 4: Forearm Cropped + Palm Circle", (10, 35),
                LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)
    cv2.putText(bottom, "Stage 5: Counting Rings + Finger Count", (10, 35),
                LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)

    combined = cv2.vconcat([top, bottom])   # previous result on top, rings+count below

    cv2.namedWindow("Stage 5: Count Fingers", cv2.WINDOW_NORMAL)
    cv2.imshow("Stage 5: Count Fingers", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
