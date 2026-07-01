"""
debug.py — 单步调试工具（Step 0 脚手架）

用静态图（data/ 里的样张）逐阶段验证 pipeline。核心是 show()：把若干
(标题, 图像) 对并排显示在一个窗口里，自动处理灰度掩膜与彩色图，并统一
缩放对齐，方便肉眼比较中间结果。
"""

import cv2
import numpy as np

import pipeline


# Height (px) that every panel is resized to before being placed side by side.
PANEL_HEIGHT = 480

# Max on-screen size of the combined window. The whole grid is shrunk to fit
# inside this box (never enlarged), so wide multi-panel views stay visible.
MAX_WINDOW_W = 1600
MAX_WINDOW_H = 850

# How many panels per row. Wide images squished into one long row become
# unreadable, so we wrap them into a grid instead.
PANELS_PER_ROW = 2

# Style for the title text drawn on top of each panel.
TITLE_FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 0.7
TITLE_COLOR = (0, 255, 0)   # BGR green
TITLE_THICK = 2

# Distinct pure BGR colors used to paint each detected hand in the overlay.
HAND_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # R,G,B,Y


def _to_bgr(img):
    """Normalize any panel image to a 3-channel BGR uint8 image.

    Grayscale masks (2-D arrays) are promoted to BGR so they can be stacked
    next to color images. Non-uint8 images are cast to uint8.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        # Grayscale / binary mask -> replicate into 3 channels.
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_to_height(img, height):
    """Resize an image to a fixed height, keeping its aspect ratio."""
    h, w = img.shape[:2]
    scale = height / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, height))


def show(*pairs):
    """Display several (title, image) pairs side by side in one window.

    Args:
        *pairs: any number of (title, image) tuples. `title` is a str,
            `image` is either a grayscale mask (H, W) or a BGR image
            (H, W, 3).

    Each image is converted to BGR, resized to a common height, labeled with
    its title, then horizontally concatenated. Press any key to close.
    """
    panels = []
    for title, img in pairs:
        panel = _to_bgr(img)
        panel = _resize_to_height(panel, PANEL_HEIGHT)
        # Draw the title in the top-left corner of the panel.
        cv2.putText(panel, title, (10, 30), TITLE_FONT, TITLE_SCALE,
                    TITLE_COLOR, TITLE_THICK, cv2.LINE_AA)
        panels.append(panel)

    # Lay panels out in a grid (PANELS_PER_ROW per row) instead of one long
    # row, so wide images don't get squished to a thin unreadable strip.
    rows = []
    for r in range(0, len(panels), PANELS_PER_ROW):
        rows.append(cv2.hconcat(panels[r:r + PANELS_PER_ROW]))
    # Pad shorter rows on the right (black) so every row has the same width.
    grid_w = max(row.shape[1] for row in rows)
    rows = [cv2.copyMakeBorder(row, 0, 0, 0, grid_w - row.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for row in rows]
    grid = cv2.vconcat(rows)

    # Shrink the whole grid so it fits inside the on-screen window box (only
    # ever scale down), otherwise big debug photos run off the screen.
    gh, gw = grid.shape[:2]
    fit = min(1.0, MAX_WINDOW_W / gw, MAX_WINDOW_H / gh)
    if fit < 1.0:
        grid = cv2.resize(grid, (int(gw * fit), int(gh * fit)))

    # WINDOW_NORMAL + resizeWindow makes the window match the fitted image.
    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.imshow("debug", grid)
    cv2.resizeWindow("debug", grid.shape[1], grid.shape[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Step 2: verify split_hands on a two-hand sample.
    # (Sample on disk is two_hands.png, not .jpg.)
    img = cv2.imread("data/two_hands.png")
    if img is None:
        raise FileNotFoundError("could not read data/two_hands.png")

    clean = pipeline.skin_mask(img)          # Step 1 mask
    hands = pipeline.split_hands(clean)      # Step 2: one binary mask per hand
    print(f"split_hands found {len(hands)} hand(s)")

    # Paint each hand a solid, OPAQUE color (direct overwrite, not blended) on
    # a copy of the original, then draw its bounding box + index number.
    overlay = img.copy()
    for idx, hand in enumerate(hands):
        color = HAND_COLORS[idx % len(HAND_COLORS)]
        overlay[hand > 0] = color                       # solid fill
        x, y, w, h = cv2.boundingRect(hand)             # bbox from the mask
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 6)
        # Draw the index in BLACK, centered inside the bounding box.
        label = f"#{idx}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 8)
        org = (x + w // 2 - tw // 2, y + h // 2 + th // 2)
        cv2.putText(overlay, label, org,
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8, cv2.LINE_AA)

    # First hand's raw black/white mask (empty if none found), shown next to
    # the Step 1 clean mask so we can compare fingertips.
    hand0 = hands[0] if hands else np.zeros_like(clean)
    show(("original", img), ("hands overlay", overlay),
         ("hand 0 mask", hand0), ("clean mask", clean))
