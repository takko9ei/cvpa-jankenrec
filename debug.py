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


def _kept_arc_samples(seq):
    """Mark which ring samples belong to a FINGER-WIDTH arc (Step 5 debug).

    Mirrors pipeline._count_ring_arcs' keep rule so the overlay paints exactly
    the arcs that were counted as fingers. Returns a bool array aligned with
    `seq`: True where the sample sits on a kept arc, False on a rejected
    (too-wide fist/palm or too-narrow noise) arc or on background.
    """
    n = len(seq)
    min_len = pipeline.FINGER_ARC_MIN_DEG / 360.0 * n
    max_len = pipeline.FINGER_ARC_MAX_DEG / 360.0 * n
    keep = np.zeros(n, dtype=bool)
    if not seq.any() or seq.all():
        return keep
    # Rotate to start on a background sample (same trick as the pipeline) so no
    # run of 1s wraps the seam; then scan runs left to right.
    shift = int(np.flatnonzero(seq == 0)[0])
    rolled = np.roll(seq, -shift)
    j = 0
    while j < n:
        if rolled[j]:
            k = j
            while k < n and rolled[k]:
                k += 1
            if min_len <= (k - j) <= max_len:
                keep[(np.arange(j, k) + shift) % n] = True   # back to orig index
            j = k
        else:
            j += 1
    return keep


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
    # Step 6: run the whole single-hand subchain end to end on rock / scissors /
    # paper — skin_mask -> split_hands -> palm_center -> crop_forearm ->
    # count_fingers -> classify — and print each hand's finger count and final
    # gesture name. The concentric rings are still drawn (colored dots = kept
    # finger arcs, gray = rejected) so the count stays inspectable.
    TESTS = ["data/stone_s5.png", "data/scissors_s5.png", "data/paper_s5.png"]
    # One distinct BGR color per ring (RING_RADIUS_COEFFS order): red, orange, yellow,
    # green, blue, magenta -- extend if RING_RADIUS_COEFFS grows.
    RING_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
                   (0, 255, 0), (255, 0, 0), (255, 0, 255)]

    panels = []
    for path in TESTS:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError("could not read " + path)

        hands = pipeline.split_hands(pipeline.skin_mask(img))   # Steps 1-2
        if not hands:
            raise RuntimeError("no hand found in " + path)
        hand = max(hands, key=lambda m: np.count_nonzero(m))
        (cx, cy), r = pipeline.palm_center(hand)                # Step 3
        cropped = pipeline.crop_forearm(hand, (cx, cy), r)      # Step 4

        # Step 5: pull the per-ring 0/1 sequences from the shared core so we
        # visualize EXACTLY what count_fingers sampled, plus the final answer.
        counts, seqs = pipeline._count_fingers_rings(cropped, (cx, cy), r)
        fingers = pipeline.count_fingers(cropped, (cx, cy), r)
        gesture = pipeline.classify(fingers)                    # Step 6

        # Draw the rings + inside-hand samples on the original photo.
        vis = img.copy()
        angles = np.linspace(0.0, 2.0 * np.pi, pipeline.RING_NUM_SAMPLES,
                             endpoint=False)
        for i, scale in enumerate(pipeline.RING_RADIUS_COEFFS):
            color = RING_COLORS[i % len(RING_COLORS)]
            ring_r = scale * r
            cv2.circle(vis, (cx, cy), int(ring_r), color, 3)   # the ring itself
            xs = np.round(cx + ring_r * np.cos(angles)).astype(int)
            ys = np.round(cy + ring_r * np.sin(angles)).astype(int)
            # Big colored dot = sample on a KEPT finger-width arc (what got
            # counted); small gray dot = inside-hand sample on a REJECTED arc
            # (fist/palm bulk too wide, or noise too narrow).
            kept = _kept_arc_samples(seqs[i])
            for k in np.flatnonzero(seqs[i]):
                if kept[k]:
                    cv2.circle(vis, (int(xs[k]), int(ys[k])), 9, color, -1)
                else:
                    cv2.circle(vis, (int(xs[k]), int(ys[k])), 4, (128, 128, 128), -1)
        cv2.circle(vis, (cx, cy), 16, (255, 255, 255), -1)     # palm center

        name = path.split("/")[-1]
        print("%-16s per-ring=%s  -> fingers=%d  -> %s"
              % (name, counts, fingers, gesture))
        title = "%s  fingers=%d  %s" % (name, fingers, gesture)
        cv2.putText(vis, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 0), 4, cv2.LINE_AA)
        panels.append((name, vis))

    # Each ring is one color. Big colored dots = samples on kept finger arcs
    # (counted); small gray dots = inside-hand samples on rejected arcs. Title
    # text shows per-ring counts and the final mode.
    show(*panels)
