"""
test_script_4.py — Stage 4 展示：切掉前臂 (crop_forearm)

依次跑 skin_mask -> split_hands -> palm_center -> crop_forearm。窗口上方
是上一步（test_script_3）的画圆结果：原图上的掌心圆心 + 内切圆。下方展示
切掉前臂后的掩膜，并把同一个掌心圆心 + 内切圆叠加画上去，方便对照裁剪线
跟掌心圆的相对位置。按任意键关闭窗口。
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

# Palm center / circle draw style, same convention as test_script_3.py.
PALM_COLOR = (255, 255, 255)      # top panel: raw color image, white reads fine
CROPPED_PALM_COLOR = (0, 0, 255)  # bottom panel: white-on-white mask is invisible, use red
PALM_CIRCLE_THICK = 3
PALM_DOT_RADIUS = 8
RADIUS_TEXT_SCALE = 0.9
RADIUS_TEXT_THICK = 2


def _resize_to_width(img, width):
    """Resize an image to a fixed width, keeping its aspect ratio."""
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))))


def _draw_palm_circle(canvas, center, radius, color=PALM_COLOR):
    """Draw the palm center dot + inscribed circle + radius label onto canvas
    (same visual as test_script_3.py's bottom panel)."""
    cx, cy = center
    cv2.circle(canvas, (cx, cy), int(radius), color, PALM_CIRCLE_THICK)
    cv2.circle(canvas, (cx, cy), PALM_DOT_RADIUS, color, -1)
    cv2.putText(canvas, "r=%.0f" % radius, (cx + 15, cy - 15), LABEL_FONT,
                RADIUS_TEXT_SCALE, color, RADIUS_TEXT_THICK, cv2.LINE_AA)


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError("could not read " + IMAGE_PATH)

    mask = pipeline.skin_mask(img)        # Stage 1
    hands = pipeline.split_hands(mask)    # Stage 2

    records = []   # (center, radius, cropped_mask) per hand
    for hand_mask in hands:
        center, radius = pipeline.palm_center(hand_mask)             # Stage 3
        cropped = pipeline.crop_forearm(hand_mask, center, radius)   # Stage 4
        records.append((center, radius, cropped))

    # Top panel: the previous step's result (raw image + palm circles).
    circles_vis = img.copy()
    for center, radius, _ in records:
        _draw_palm_circle(circles_vis, center, radius)

    # Bottom panel: union of every hand's forearm-cropped mask, with the same
    # palm circles overlaid on top.
    cropped_union = np.zeros(mask.shape, dtype=np.uint8)
    for _, _, cropped in records:
        cropped_union = cv2.bitwise_or(cropped_union, cropped)
    cropped_vis = cv2.cvtColor(cropped_union, cv2.COLOR_GRAY2BGR)
    for center, radius, _ in records:
        _draw_palm_circle(cropped_vis, center, radius, color=CROPPED_PALM_COLOR)

    top = _resize_to_width(circles_vis, DISPLAY_WIDTH)
    bottom = _resize_to_width(cropped_vis, DISPLAY_WIDTH)

    cv2.putText(top, "Stage 3: Palm Center + Inscribed Circle", (10, 35),
                LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)
    cv2.putText(bottom, "Stage 4: Forearm Cropped + Palm Circle", (10, 35),
                LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)

    combined = cv2.vconcat([top, bottom])   # previous circle result on top, cropped mask below

    cv2.namedWindow("Stage 4: Crop Forearm", cv2.WINDOW_NORMAL)
    cv2.imshow("Stage 4: Crop Forearm", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
