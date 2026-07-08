"""
test_script_3.py — Stage 3 展示：掌心定位 (palm_center)

读取 data/rock_paper.png，依次跑 skin_mask -> split_hands -> palm_center。
窗口上方展示每只手的距离变换热力图（palm_center 正是在这张图上找最大值来
定位掌心的），下方展示在原图上画出的掌心圆心 + 内切圆。两张图都是这一步
才出现的新图，不重复用之前脚本的掩膜/分割图。按任意键关闭窗口。
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

# Palm center / circle draw style (white, matching main.py's debug overlay
# convention for "the chosen palm center + inscribed circle").
PALM_COLOR = (255, 255, 255)
PALM_CIRCLE_THICK = 3
PALM_DOT_RADIUS = 8
RADIUS_TEXT_SCALE = 0.9
RADIUS_TEXT_THICK = 2


def _resize_to_width(img, width):
    """Resize an image to a fixed width, keeping its aspect ratio."""
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))))


def _distance_transform_heatmap(shape, hands):
    """Combine each hand's distance transform into one JET heatmap.

    This is the SAME cv2.distanceTransform() call palm_center() runs
    internally: brightest = deepest interior point = the palm-center
    candidate. Background (outside every hand) is forced to pure black
    instead of JET's dark blue, so the hand silhouettes stay readable.
    """
    combined = np.zeros(shape[:2], dtype=np.float32)
    any_hand = np.zeros(shape[:2], dtype=np.uint8)
    for hand_mask in hands:
        dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
        combined = np.maximum(combined, dist)
        any_hand |= hand_mask

    norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heat[any_hand == 0] = (0, 0, 0)
    return heat


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError("could not read " + IMAGE_PATH)

    mask = pipeline.skin_mask(img)        # Stage 1
    hands = pipeline.split_hands(mask)    # Stage 2

    heat = _distance_transform_heatmap(img.shape, hands)

    circles_vis = img.copy()
    for hand_mask in hands:
        center, radius = pipeline.palm_center(hand_mask)   # Stage 3
        cx, cy = center
        cv2.circle(circles_vis, (cx, cy), int(radius), PALM_COLOR,
                   PALM_CIRCLE_THICK)
        cv2.circle(circles_vis, (cx, cy), PALM_DOT_RADIUS, PALM_COLOR, -1)
        cv2.putText(circles_vis, "r=%.0f" % radius, (cx + 15, cy - 15),
                    LABEL_FONT, RADIUS_TEXT_SCALE, PALM_COLOR,
                    RADIUS_TEXT_THICK, cv2.LINE_AA)

    top = _resize_to_width(heat, DISPLAY_WIDTH)
    bottom = _resize_to_width(circles_vis, DISPLAY_WIDTH)

    cv2.putText(top, "Stage 3: Distance Transform", (10, 35), LABEL_FONT,
                LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)
    cv2.putText(bottom, "Stage 3: Palm Center + Inscribed Circle", (10, 35),
                LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)

    combined = cv2.vconcat([top, bottom])   # distance-transform heat on top, circles below

    cv2.namedWindow("Stage 3: Palm Center", cv2.WINDOW_NORMAL)
    cv2.imshow("Stage 3: Palm Center", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
