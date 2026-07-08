"""
test_script_2.py — Stage 2 展示：连通域拆分 (split_hands)

读取 data/rock_paper.png，跑 pipeline.skin_mask() 得到掩膜，再跑
pipeline.split_hands() 拆出每只手的子掩膜。窗口上方展示 Stage 1 的原始
掩膜，下方在同一张掩膜上给每只手的区域分别上色并标注编号（不是并排画
两张掩膜图）。按任意键关闭窗口。
"""

import cv2
import numpy as np

import pipeline


# Same source image as test_script_1.py; all later test_script_N.py reuse it.
IMAGE_PATH = "data/rock_paper.png"

# Both panels are resized to this width before stacking.
DISPLAY_WIDTH = 800

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.9
LABEL_COLOR = (0, 255, 0)   # BGR green
LABEL_THICK = 2

# Distinct BGR color per hand, cycled if more hands than colors show up.
HAND_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # R, G, B, Y

# Style for the per-hand index number drawn at each hand's centroid.
NUMBER_FONT = cv2.FONT_HERSHEY_SIMPLEX
NUMBER_SCALE = 3.0
NUMBER_COLOR = (255, 255, 255)
NUMBER_THICK = 4


def _resize_to_width(img, width):
    """Resize an image to a fixed width, keeping its aspect ratio."""
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))))


def _colorize_hands(shape, hands):
    """Paint each hand's region a distinct color on a black canvas and label
    it with its 1-based index at the region's centroid.

    Operates on the SAME mask geometry split_hands() produced -- this is a
    recoloring of that one mask, not a second separately-drawn image.
    """
    vis = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for idx, hand_mask in enumerate(hands):
        color = HAND_COLORS[idx % len(HAND_COLORS)]
        vis[hand_mask > 0] = color

        # Centroid of this hand's pixels -> where the index label gets drawn.
        ys, xs = np.where(hand_mask > 0)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(idx + 1)
        (tw, th), _ = cv2.getTextSize(label, NUMBER_FONT, NUMBER_SCALE,
                                       NUMBER_THICK)
        cv2.putText(vis, label, (cx - tw // 2, cy + th // 2), NUMBER_FONT,
                    NUMBER_SCALE, NUMBER_COLOR, NUMBER_THICK, cv2.LINE_AA)
    return vis


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError("could not read " + IMAGE_PATH)

    mask = pipeline.skin_mask(img)      # Stage 1: raw -> binary skin mask
    hands = pipeline.split_hands(mask)  # Stage 2: mask -> one sub-mask per hand

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    split_vis = _colorize_hands(mask.shape, hands)

    top = _resize_to_width(mask_bgr, DISPLAY_WIDTH)
    bottom = _resize_to_width(split_vis, DISPLAY_WIDTH)

    cv2.putText(top, "Stage 1: Skin Mask", (10, 35), LABEL_FONT, LABEL_SCALE,
                LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)
    cv2.putText(bottom, "Stage 2: Split Hands (%d found)" % len(hands),
                (10, 35), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK,
                cv2.LINE_AA)

    combined = cv2.vconcat([top, bottom])   # stage-1 mask on top, colored split below

    cv2.namedWindow("Stage 2: Split Hands", cv2.WINDOW_NORMAL)
    cv2.imshow("Stage 2: Split Hands", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
