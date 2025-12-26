import cv2
import queue
import sys
import time
import logging

from video_ingestion.video_ingestion import CameraWorker
from ai_perception.ai_perception import PerceptionWorker
from action_detector.action_detector import ActionDetector
from pattern_analyser.pattern_analyser import MotionPatternAnalyzer
from kitchen_environment.snapshot_collector.collector import SnapshotCollector

logging.basicConfig(level=logging.INFO)

# ==============================
# НАСТРОЙКИ
# ==============================

CAMERAS = [
    {
    "camera_id": "Kitchen_1",
    "source": "../res/videos/gloves.mp4",      # 0 = вебкамера, можно заменить на RTSP/файл
    },
]

TARGET_FPS = 5
RESOLUTION = (1280, 720)

# ==============================

# ЗАПУСК

# ==============================

if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=32)
    out_queue = queue.Queue(maxsize=32)
    action_detector_queue = queue.Queue(maxsize=32)
    workers = []

# ---------------------------
# ЗАПУСК CAMERA WORKERS
# ---------------------------
for cam in CAMERAS:
    w = CameraWorker(
        camera_id=cam["camera_id"],
        source=cam["source"],
        out_queue=frame_queue,
        target_fps=TARGET_FPS,
        target_resolution=RESOLUTION,
        brightness_alpha=1.0,
        brightness_beta=0.0
    )
    w.daemon = True
    w.start()
    workers.append(w)
    print(f"[INFO] Started video ingestion for {cam['camera_id']}")
    time.sleep(1.5)

# ---------------------------
# ЗАПУСК AI PERCEPTION
# ---------------------------
perception = PerceptionWorker(frame_queue, out_queue)
perception.daemon = True
perception.start()
print("[INFO] Started AI perception module")

# ---------------------------
# SNAPSHOT COLLECTOR
# ---------------------------
snapshot_collector = SnapshotCollector(interval_sec=30)
snapshot_collector.start()
print("[INFO] SnapshotCollector started (every 5 seconds)")

# ---------------------------
# ACTION DETECTOR
# ---------------------------
pattern_analyser = MotionPatternAnalyzer()
action_detector = ActionDetector(cams=CAMERAS, in_queue=action_detector_queue, analyser=pattern_analyser)
action_detector.daemon = True
action_detector.start()

# ==============================
# ОСНОВНОЙ ЦИКЛ
# ==============================
try:
    last_alive_check = time.time()

    while True:
        try:
            out = out_queue.get(timeout=2)

            camera_id = out.get("camera_id", "Unknown")
            frame = out.get("frame_raw")
            objects = out.get("objects", [])

            try:
                action_detector_queue.put_nowait({"camera_id": camera_id, "objects": objects})
            except queue.Empty:
                print("Failed to put action_detector queue")

            if frame is None:
                continue

            # передаём кадр в snapshot-систему
            try:
                snapshot_collector.register_frame(camera_id, frame)
            except Exception:
                logging.exception("Snapshot registration failed")

            # рисуем детекции на экране
            for obj in objects:
                bbox = [int(v) for v in obj["bbox"]]
                x1, y1, x2, y2 = bbox
                cls_name = str(obj.get("class", "unknown"))
                conf = obj.get("confidence", 0.0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{cls_name} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        except queue.Empty:
            # проверяем живость потоков раз в 5 секунд
            if time.time() - last_alive_check > 5:
                alive_cams = [w.is_alive() for w in workers]
                alive_perc = perception.is_alive()

                logging.info(
                    f"[HEALTH] Cameras: {alive_cams}, Perception: {alive_perc}"
                )

                # перезапуск камер
                if not any(alive_cams):
                    logging.warning("[WARN] All CameraWorkers stopped! Restarting...")
                    for cam in CAMERAS:
                        w = CameraWorker(
                            camera_id=cam["camera_id"],
                            source=cam["source"],
                            out_queue=frame_queue,
                            target_fps=TARGET_FPS,
                            target_resolution=RESOLUTION
                        )
                        w.daemon = True
                        w.start()
                        workers.append(w)

                # перезапуск perception
                if not perception.is_alive():
                    logging.warning("[WARN] PerceptionWorker stopped! Restarting...")
                    perception = PerceptionWorker(frame_queue, out_queue)
                    perception.daemon = True
                    perception.start()

                last_alive_check = time.time()

            continue

except KeyboardInterrupt:
    print("[INFO] Stopping due to Ctrl+C")

finally:
    print("[INFO] Stopping all workers...")

    # останов камер
    for w in workers:
        w.stop()

    # останов perception
    perception.stop()

    # останов snapshot collector
    try:
        snapshot_collector.stop()
    except Exception:
        logging.exception("Failed to stop snapshot_collector cleanly")

    # дождаться потоков
    for w in workers:
        w.join(timeout=2.0)
    perception.join(timeout=2.0)

    cv2.destroyAllWindows()
    print("[INFO] All stopped cleanly.")
    sys.exit(0)

