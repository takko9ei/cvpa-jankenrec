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

# Style for the title text drawn on top of each panel.
TITLE_FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 0.7
TITLE_COLOR = (0, 255, 0)   # BGR green
TITLE_THICK = 2


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

    # Stack all panels horizontally into a single strip and show it.
    strip = cv2.hconcat(panels)
    cv2.imshow("debug", strip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Step 1: verify skin_mask on a static sample.
    # (Sample on disk is paper.png, not paper.jpg.)
    img = cv2.imread("data/paper.jpg")
    if img is None:
        raise FileNotFoundError("could not read data/paper.png")

    # return_stages=True gives us both the raw thresholded mask and the
    # morphology-cleaned mask so we can compare them side by side.
    raw, clean = pipeline.skin_mask(img, return_stages=True)
    show(("original", img), ("raw mask", raw), ("clean mask", clean))
