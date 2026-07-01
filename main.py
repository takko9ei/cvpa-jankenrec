"""
main.py — 实时主程序（Step 0 脚手架）

打开默认摄像头，循环读帧并显示，叠加 FPS 文本，按 q 退出。
现阶段还不调用 pipeline 里的任何函数，只验证摄像头采集与显示正常。
"""

import time

import cv2


# FPS text style.
FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_SCALE = 0.8
FPS_COLOR = (0, 255, 0)   # BGR green
FPS_THICK = 2


def main():
    # Open the default camera (index 0).
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("could not open the default camera")

    prev_time = time.time()   # timestamp of the previous frame, for FPS.

    while True:
        ok, frame = cap.read()
        if not ok:
            # Camera hiccup / end of stream — stop the loop.
            break

        # --- FPS: 1 / (time between this frame and the previous one) ---
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        # Overlay the FPS reading in the top-left corner.
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FPS_FONT, FPS_SCALE,
                    FPS_COLOR, FPS_THICK, cv2.LINE_AA)

        cv2.imshow("jankenrec", frame)

        # Quit when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Always release the camera and close windows on exit.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
