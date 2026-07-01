"""
camera_test.py — 摄像头诊断 / 选择脚本

跟 pipeline 无关。用来排查 main.py 黑屏，并解决"调用的是 iPhone（连续互通相机）
而不是 Mac 自带摄像头"的问题。

macOS 的 Continuity Camera 会把附近的 iPhone 当成一个摄像头，而且常常抢占索引 0，
所以 VideoCapture(0) 拿到的是 iPhone。本脚本会：

  1. 打印平台 / OpenCV 版本；
  2. "照抄 main.py 的开相机方式"跑一遍，看它到底连到哪个设备；
  3. 枚举索引 0..N 的每个摄像头，各读若干帧，把非黑的存成
     camera_test_index{N}.png —— 你打开这些图就能认出哪个索引是 Mac 自带、
     哪个是 iPhone；
  4. 给 --show [索引] 时开窗口实时看某个索引，按 q 退出。

用法:
    venv/bin/python camera_test.py            # 枚举所有摄像头 + 各存一张样图
    venv/bin/python camera_test.py --show 1   # 实时预览索引 1（换成你要的号）
"""

import sys
import time
import platform

import cv2
import numpy as np


# How many frames to read per camera. macOS cameras often hand back several
# BLACK frames first while auto-exposure warms up, so we must read a bunch.
WARMUP_FRAMES = 30

# How many camera indices to probe (0 .. MAX_INDEX-1).
MAX_INDEX = 4


def backends_for_platform():
    """Backends worth trying on this OS, best first: list of (name, flag)."""
    system = platform.system()
    if system == "Darwin":                       # macOS
        return [("AVFOUNDATION", cv2.CAP_AVFOUNDATION), ("ANY", cv2.CAP_ANY)]
    if system == "Windows":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF),
                ("ANY", cv2.CAP_ANY)]
    return [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]   # Linux / other


def frame_stats(frame):
    """One-line summary of a frame; makes an all-black frame obvious."""
    if frame is None:
        return "frame = None"
    nonzero = 100.0 * np.count_nonzero(frame) / frame.size
    return ("shape=%s  mean=%.1f  max=%d  nonzero=%.1f%%"
            % (frame.shape, float(frame.mean()), int(frame.max()), nonzero))


def read_best_frame(cap, warmup=WARMUP_FRAMES):
    """Read `warmup` frames; return the brightest (best evidence of a picture).

    Returns None if every read failed. If even the brightest frame is all-black,
    the camera is genuinely handing back black frames (permission / warm-up).
    """
    best = None
    for _ in range(warmup):
        ok, frame = cap.read()
        if ok and frame is not None:
            if best is None or frame.mean() > best.mean():
                best = frame.copy()
        time.sleep(0.02)
    return best


def best_backend():
    """The first backend flag for this OS (AVFoundation on macOS)."""
    return backends_for_platform()[0]


def reproduce_main():
    """Do EXACTLY what main.py does now: VideoCapture(0) + set 1280x720."""
    print("=== 1. reproduce main.py: VideoCapture(0) + set 1280x720 ===")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  could not open camera index 0")
        cap.release()
        print()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame = read_best_frame(cap)
    cap.release()
    print("  index 0 reports %dx%d  ->  %s" % (w, h, frame_stats(frame)))
    if frame is not None and frame.max() > 0:
        cv2.imwrite("camera_test_main.png", frame)
        print("  saved camera_test_main.png (this is what main.py currently sees")
        print("  -- open it: is it the iPhone or the Mac camera?)")
    print()


def enumerate_cameras():
    """Open indices 0..MAX_INDEX-1 on the best backend; save a sample per camera.

    Returns the list of indices that produced a real (non-black) frame.
    """
    name, flag = best_backend()
    print("=== 2. enumerate cameras (backend=%s) ===" % name)
    working = []
    for index in range(MAX_INDEX):
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            print("  index %d: not opened (no such camera)" % index)
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame = read_best_frame(cap)
        cap.release()
        if frame is None or frame.max() == 0:
            print("  index %d: opened %dx%d but only BLACK frames" % (index, w, h))
            continue
        out = "camera_test_index%d.png" % index
        cv2.imwrite(out, frame)
        print("  index %d: OK  %dx%d  %s  -> saved %s"
              % (index, w, h, frame_stats(frame), out))
        working.append(index)
    print()
    return working


def live_view(index):
    """Open a window on one index (best backend) until 'q' is pressed."""
    _, flag = best_backend()
    cap = cv2.VideoCapture(index, flag)
    if not cap.isOpened():
        print(">> could not open index %d for live view" % index)
        return
    print(">> live view of index %d; press q in the window to quit." % index)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("camera_test index %d" % index, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_show_index(argv):
    """Return an int index if '--show [N]' was passed, else None."""
    if "--show" not in argv:
        return None
    i = argv.index("--show")
    if i + 1 < len(argv) and argv[i + 1].isdigit():
        return int(argv[i + 1])
    return 0   # default to index 0 if --show given without a number


def main():
    print("platform:", platform.system(), platform.release())
    print("opencv  :", cv2.__version__)
    print()

    reproduce_main()
    working = enumerate_cameras()

    if not working:
        print(">> No non-black frame from any index.")
        print(">> On macOS this is usually a CAMERA PERMISSION issue: System")
        print("   Settings > Privacy & Security > Camera -> enable it for the app")
        print("   you launch python from (Terminal / iTerm / VS Code), then fully")
        print("   quit & reopen that app and re-run.")
        return

    print(">> Working camera indices: %s" % working)
    print(">> Open the saved camera_test_index*.png and see which index is your")
    print("   MAC BUILT-IN camera vs the iPhone.")
    print(">> The iPhone appearing is macOS *Continuity Camera*; it often grabs")
    print("   index 0, so your built-in camera is usually a HIGHER index.")
    print(">> To use the built-in camera in main.py, open it by that index")
    print("   (I can add a CAM_INDEX constant so you just change one number).")
    print(">> Or turn Continuity Camera OFF on the iPhone: Settings > General >")
    print("   AirPlay & Handoff > Continuity Camera.")

    show_index = parse_show_index(sys.argv)
    if show_index is not None:
        print()
        live_view(show_index)


if __name__ == "__main__":
    main()
