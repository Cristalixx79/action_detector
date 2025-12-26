import cv2
import os
import threading
import time
from datetime import datetime


class SnapshotCollector:
    def __init__(self, interval_sec: int = 30, base_dir: str = "../../../res/snapshots"):
        self.interval = interval_sec
        self.base_dir = base_dir

        os.makedirs(base_dir, exist_ok=True)

        # сюда будет складываться самый свежий кадр для каждой камеры
        self.frames = {}

        # поток для сохранения
        self._running = False
        self._thread = None

    def register_frame(self, camera_id: str, frame):
        """
        Сюда run_multi_camera передаёт кадры из RTSP
        """
        self.frames[camera_id] = frame

    def start(self):
        """
        Запускает фоновый поток, который раз в N секунд сохраняет снимки
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        print(f"[SnapshotCollector] Started (interval={self.interval}s)")

    def stop(self):
        self._running = False

    def _loop(self):
        """
        Фоновый метод, сохраняет снимки
        """
        while self._running:
            time.sleep(self.interval)

            for cam_id, frame in self.frames.items():
                try:
                    folder = os.path.join("C:/Temp", cam_id)
                    os.makedirs(folder, exist_ok=True)

                    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
                    filename = os.path.join(folder, f"{timestamp}.jpg")
                    cv2.imwrite(filename, frame)

                    print(f"[Snapshot] Saved for {cam_id} -> {filename}")

                except Exception as e:
                    print(f"[SnapshotCollector] ERROR saving frame for {cam_id}: {e}")
