"""
test_script_6.py — 最终展示：分类 + 判负 + 绘制结果 (classify -> judge -> draw)

跑完整条流水线：skin_mask -> split_hands -> palm_center -> crop_forearm ->
count_fingers -> classify -> judge -> draw。窗口上方是上一步
（test_script_5）下方的结果：计数环 + 手指数叠加在裁剪后的掩膜上。下方是
原图，参照 main.py 最终显示的样式，用 pipeline.draw() 在手部画出结果圆圈
和 "手势: 胜负" 标签。按任意键关闭窗口。
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

# Palm center / circle draw style, same convention as test_script_5.py.
PALM_COLOR = (0, 0, 255)   # red
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

# Style for the final "<gesture>: <result>" label on the bottom panel. Bigger
# and centered INSIDE the palm circle, unlike pipeline.draw()'s default
# (smaller, placed above the circle) -- this override is local to this demo
# script only, pipeline.draw() itself is untouched.
FINAL_TEXT_SCALE = 3.2
FINAL_TEXT_THICK = 6


def _resize_to_width(img, width):
    """Resize an image to a fixed width, keeping its aspect ratio."""
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))))


def _draw_palm_circle(canvas, center, radius):
    """Draw the palm center dot + inscribed circle + radius label onto canvas
    (same visual as test_script_5.py's top-panel base)."""
    cx, cy = center
    cv2.circle(canvas, (cx, cy), int(radius), PALM_COLOR, PALM_CIRCLE_THICK)
    cv2.circle(canvas, (cx, cy), PALM_DOT_RADIUS, PALM_COLOR, -1)
    cv2.putText(canvas, "r=%.0f" % radius, (cx + 15, cy - 15), LABEL_FONT,
                RADIUS_TEXT_SCALE, PALM_COLOR, RADIUS_TEXT_THICK, cv2.LINE_AA)


def _draw_rings_and_count(canvas, center, radius, fingers):
    """Draw the concentric counting rings (pipeline.RING_RADIUS_COEFFS) and
    write the finger count beside the hand (same visual as
    test_script_5.py's bottom panel)."""
    cx, cy = center
    h, w = canvas.shape[:2]

    for i, scale in enumerate(pipeline.RING_RADIUS_COEFFS):
        color = RING_COLORS[i % len(RING_COLORS)]
        cv2.circle(canvas, (cx, cy), int(scale * radius), color, RING_THICK)

    outer_r = max(pipeline.RING_RADIUS_COEFFS) * radius
    label = "fingers=%d" % fingers
    (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, COUNT_TEXT_SCALE,
                                   COUNT_TEXT_THICK)
    tx = min(int(cx + outer_r) + 20, w - tw - 10)
    ty = cy
    cv2.putText(canvas, label, (tx, ty), LABEL_FONT, COUNT_TEXT_SCALE,
                COUNT_TEXT_COLOR, COUNT_TEXT_THICK, cv2.LINE_AA)


def _draw_final_result(canvas, records):
    """Draw the final per-hand result: palm circle in the result color (same
    pipeline.RESULT_COLORS / pipeline.DRAW_THICKNESS as pipeline.draw()) plus
    a "<gesture>: <result>" label centered INSIDE the circle, bigger than
    pipeline.draw()'s default. A local override of pipeline.draw()'s look,
    not a change to pipeline.py itself.
    """
    for rec in records:
        cx, cy = rec["center"]
        radius = int(rec.get("radius", 0))
        color = pipeline.RESULT_COLORS.get(rec["result"], (150, 150, 150))

        if radius > 0:
            cv2.circle(canvas, (cx, cy), radius, color, pipeline.DRAW_THICKNESS)

        label = "%s: %s" % (rec["gesture"], rec["result"])
        (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, FINAL_TEXT_SCALE,
                                       FINAL_TEXT_THICK)
        tx = cx - tw // 2
        ty = cy + th // 2   # vertically centered on the circle center
        cv2.putText(canvas, label, (tx, ty), LABEL_FONT, FINAL_TEXT_SCALE,
                    color, FINAL_TEXT_THICK, cv2.LINE_AA)
    return canvas


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError("could not read " + IMAGE_PATH)

    mask = pipeline.skin_mask(img)        # Stage 1
    hands = pipeline.split_hands(mask)    # Stage 2

    records = []
    for hand_mask in hands:
        center, radius = pipeline.palm_center(hand_mask)             # Stage 3
        cropped = pipeline.crop_forearm(hand_mask, center, radius)   # Stage 4
        fingers = pipeline.count_fingers(cropped, center, radius)    # Stage 5
        gesture = pipeline.classify(fingers)                         # Stage 6
        records.append({"center": center, "radius": radius,
                        "cropped": cropped, "fingers": fingers,
                        "gesture": gesture})

    results = pipeline.judge([rec["gesture"] for rec in records])     # Stage 7
    for rec, result in zip(records, results):
        rec["result"] = result

    # Top panel: the previous step's result (counting rings + finger count on
    # the cropped mask).
    cropped_union = np.zeros(mask.shape, dtype=np.uint8)
    for rec in records:
        cropped_union = cv2.bitwise_or(cropped_union, rec["cropped"])
    top_vis = cv2.cvtColor(cropped_union, cv2.COLOR_GRAY2BGR)
    for rec in records:
        _draw_palm_circle(top_vis, rec["center"], rec["radius"])
        _draw_rings_and_count(top_vis, rec["center"], rec["radius"],
                              rec["fingers"])

    # Bottom panel: raw image + the final result (palm circle in the result
    # color + a bigger "<gesture>: <result>" label centered in the circle;
    # see _draw_final_result for how this differs from pipeline.draw()).
    bottom_vis = _draw_final_result(img.copy(), records)

    top = _resize_to_width(top_vis, DISPLAY_WIDTH)
    bottom = _resize_to_width(bottom_vis, DISPLAY_WIDTH)

    cv2.putText(top, "Stage 5: Counting Rings + Finger Count", (10, 35),
                LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)
    cv2.putText(bottom, "Stage 6-8: Classify + Judge + Draw", (10, 35),
                LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)

    combined = cv2.vconcat([top, bottom])   # previous result on top, final result below

    cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Final Result", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
