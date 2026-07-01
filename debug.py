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

# How many panels per row. 1 = stack all panels vertically (top to bottom);
# larger values wrap the panels into a grid.
PANELS_PER_ROW = 1

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
    # Step 3 (re-verify after the fist-vs-forearm fix): the palm center must
    # land on the palm/fist center for ALL of rock / scissors / paper, and must
    # NOT slide down onto the wrist. The three gesture photos share one capture
    # setup (data/*_s5.png; "stone" == rock).
    TESTS = ["data/stone_s5.png", "data/scissors_s5.png", "data/paper_s5.png"]

    panels = []
    for path in TESTS:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError("could not read " + path)

        clean = pipeline.skin_mask(img)          # Step 1 mask
        hands = pipeline.split_hands(clean)      # Step 2: one mask per hand
        if not hands:
            raise RuntimeError("no hand found in " + path)
        # If several blobs survive, judge the biggest (the actual hand).
        hand = max(hands, key=lambda m: np.count_nonzero(m))
        (cx, cy), r = pipeline.palm_center(hand)  # Step 3 (band-constrained)

        # Recompute the search band the SAME way palm_center does, so we can
        # draw the cyan band-bottom line and see what constrained the search.
        ys, xs = np.where(hand > 0)
        top_y = int(ys.min())
        hand_w = int(xs.max() - xs.min() + 1)
        band_bottom = top_y + int(pipeline.TOP_SEARCH_SCALE * hand_w)

        # Annotate the ORIGINAL photo so we can judge by eye whether the red
        # center dot sits on the palm/fist and not on the forearm.
        vis = img.copy()
        cv2.line(vis, (0, top_y), (vis.shape[1], top_y), (0, 255, 255), 4)      # top_y (yellow)
        cv2.line(vis, (0, band_bottom), (vis.shape[1], band_bottom),
                 (255, 255, 0), 4)                                              # band bottom (cyan)
        cv2.circle(vis, (cx, cy), int(r), (0, 255, 0), 5)   # inscribed circle
        cv2.circle(vis, (cx, cy), 18, (0, 0, 255), -1)      # palm center dot

        name = path.split("/")[-1]
        print("%-16s center=(%d,%d) r=%.0f  band=[%d,%d]"
              % (name, cx, cy, r, top_y, band_bottom))
        panels.append(("%s  c=(%d,%d) r=%.0f" % (name, cx, cy, r), vis))

    # Yellow = top_y, Cyan = band bottom (search limit), Green = inscribed
    # circle, Red = palm center.
    show(*panels)
