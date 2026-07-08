"""
test_script_1.py — Stage 1 展示：肤色分割 (skin_mask)

读取 data/rock_paper.png，跑 pipeline.skin_mask() 得到二值掩膜，在同一个
窗口里竖直展示：上方原图，下方黑白掩膜。按任意键关闭窗口。
"""

import cv2

import pipeline


# All later test_script_N.py files use this same source image.
IMAGE_PATH = "data/rock_paper.png"

# Both panels are resized to this width before stacking, so the combined
# window fits on screen regardless of the source image's native resolution.
DISPLAY_WIDTH = 800

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.9
LABEL_COLOR = (0, 255, 0)   # BGR green
LABEL_THICK = 2


def _resize_to_width(img, width):
    """Resize an image to a fixed width, keeping its aspect ratio."""
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))))


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError("could not read " + IMAGE_PATH)

    mask = pipeline.skin_mask(img)                       # Stage 1: raw -> binary mask
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)     # promote to BGR so it can stack with the color image

    top = _resize_to_width(img, DISPLAY_WIDTH)
    bottom = _resize_to_width(mask_bgr, DISPLAY_WIDTH)

    # Label each panel in its top-left corner before stacking.
    cv2.putText(top, "Raw Image", (10, 35), LABEL_FONT, LABEL_SCALE,
                LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)
    cv2.putText(bottom, "Skin Mask", (10, 35), LABEL_FONT, LABEL_SCALE,
                LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)

    combined = cv2.vconcat([top, bottom])   # raw on top, mask on bottom

    cv2.namedWindow("Stage 1: Skin Segmentation", cv2.WINDOW_NORMAL)
    cv2.imshow("Stage 1: Skin Segmentation", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
