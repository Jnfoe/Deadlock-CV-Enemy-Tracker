from pathlib import Path
import time

import cv2
from ultralytics import YOLO


# SETTINGS
CAMERA_INDEX = 1       # Try 1 or 2 if the wrong camera opens.
CONFIDENCE = 0.35      # Lower = more detections, potentially more false positives.
IMAGE_SIZE = 640
DEVICE = "cpu"        # This version uses your CPU, not your RX 6600.


def main():
    folder = Path(__file__).resolve().parent
    model_path = folder / "best.pt"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Put your downloaded best.pt beside this script."
        )

    print("Loading model...")
    model = YOLO(str(model_path))

    # DirectShow capture using the verified camera index.
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}. "
            "Try another CAMERA_INDEX and close other apps using the camera."
        )

    # Camera drivers may choose a different supported resolution.
    # Keep the camera's native resolution for reliable frame capture.


    window_name = "Live enemy detection - Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("Camera running. Click the video window and press Q to quit.")

    smoothed_fps = None

    try:
        while True:
            start = time.perf_counter()

            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the camera.")
                break

            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=DEVICE,
                verbose=False,
            )[0]

            coordinates = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, confidence in zip(coordinates, confidences):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Enemy {confidence:.0%}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            elapsed = time.perf_counter() - start
            current_fps = 1.0 / max(elapsed, 1e-6)

            if smoothed_fps is None:
                smoothed_fps = current_fps
            else:
                smoothed_fps = (
                    0.9 * smoothed_fps + 0.1 * current_fps
                )

            cv2.putText(
                frame,
                f"Processing FPS: {smoothed_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Enemies detected: {len(coordinates)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()