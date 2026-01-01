#video_ingestion
import cv2
import time
import threading
import base64
import json
import queue
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video_ingestion")

# Helper: encode frame to JPEG and base64
def frame_to_base64_jpeg(frame, jpeg_quality=80):
    ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ret:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return b64

class ExponentialBackoff:
    def __init__(self, base=1.0, factor=2.0, max_delay=60.0):
        self.base = base
        self.factor = factor
        self.max_delay = max_delay
        self.attempt = 0

    def next_delay(self):
        delay = min(self.base * (self.factor ** self.attempt), self.max_delay)
        self.attempt += 1
        # add jitter
        jitter = delay * 0.1
        return delay + (jitter * (2 * (random.random() - 0.5)))

    def reset(self):
        self.attempt = 0

import random

class CameraWorker(threading.Thread):
    def __init__(self,
                 camera_id,
                 source,  # e.g. rtsp://..., /dev/video0, http://ip/stream or integer for webcam
                 out_queue=None,
                 target_fps=5,
                 target_resolution=(1280, 720),
                 roi=None,  # tuple (x, y, w, h) in pixels or None
                 brightness_alpha=1.0,  # contrast multiplier
                 brightness_beta=0.0,   # brightness addition
                 gst_pipeline=None,  # optional custom GStreamer pipeline string
                 jpeg_quality=80,
                 reconnect_base=1.0,
                 reconnect_max=60.0
                 ):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.out_queue = out_queue
        self.target_fps = min(target_fps, 30)
        self.frame_interval = 1.0 / max(0.0001, self.target_fps)
        self.target_resolution = target_resolution
        self.roi = roi
        self.brightness_alpha = brightness_alpha
        self.brightness_beta = brightness_beta
        self.jpeg_quality = jpeg_quality
        self.stop_event = threading.Event()
        self.capture = None
        self.frame_id = 0
        self.backoff = ExponentialBackoff(base=reconnect_base, max_delay=reconnect_max)
        self.gst_pipeline = gst_pipeline

    def build_gst_pipeline(self):
        # if user provided a pipeline, use it; otherwise attempt RTSP-friendly pipeline
        if self.gst_pipeline:
            return self.gst_pipeline
        # Basic RTSP pipeline for H264 decode -> appsink
        pipeline = (
            f'rtspsrc location={self.source} latency=200 ! rtph264depay ! h264parse ! avdec_h264 ! '
            f'videoconvert ! videoscale ! video/x-raw, width={self.target_resolution[0]}, height={self.target_resolution[1]} ! appsink'
        )
        return pipeline

    def open_capture(self):
        # Prefer GStreamer if source looks like RTSP and OpenCV supports GStreamer
        try:
            # close previous capture if exists
            try:
                if self.capture:
                    try:
                        self.capture.release()
                    except Exception:
                        pass
                    self.capture = None
            except Exception:
                pass

            if isinstance(self.source, str) and (self.source.startswith('rtsp://') or self.source.startswith('rtsps://')):
                gst = self.build_gst_pipeline()
                logger.info(f"[{self.camera_id}] Trying GStreamer pipeline")
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cap is not None and cap.isOpened():
                    logger.info(f"[{self.camera_id}] Opened via GStreamer pipeline")
                    return cap
                else:
                    logger.warning(f"[{self.camera_id}] GStreamer VideoCapture failed, trying plain rtsp")

            # handle numeric/int webcam source
            if isinstance(self.source, int) or (isinstance(self.source, str) and str(self.source).isdigit()):
                idx = int(self.source)
                logger.info(f"[{self.camera_id}] Opening local webcam {idx}")
                cap = cv2.VideoCapture(idx)
            else:
                logger.info(f"[{self.camera_id}] Opening capture to {self.source}")
                cap = cv2.VideoCapture(self.source)

            # wait a bit for the capture to become ready (up to ~10s)
            for attempt in range(10):
                if cap is not None and cap.isOpened():
                    break
                logger.debug(f"[{self.camera_id}] Waiting for VideoCapture to open (attempt {attempt+1}/10)...")
                time.sleep(1.0)
            if cap is None or not cap.isOpened():
                # try one more time with a fresh VideoCapture (some devices need reopen)
                try:
                    if isinstance(self.source, int) or (isinstance(self.source, str) and str(self.source).isdigit()):
                        cap = cv2.VideoCapture(int(self.source))
                    else:
                        cap = cv2.VideoCapture(self.source)
                    time.sleep(1.0)
                except Exception:
                    pass

            if cap is None or not cap.isOpened():
                raise RuntimeError(f"[{self.camera_id}] Failed to open capture source: {self.source}")

            # Try set resolution (may be ignored by some backends)
            try:
                w, h = self.target_resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            except Exception:
                pass

            logger.info(f"[{self.camera_id}] Capture opened successfully")
            return cap
        except Exception as e:
            logger.exception(f"[{self.camera_id}] open_capture exception: {e}")
            raise

    def apply_roi_and_brightness(self, frame):
        if self.roi is not None:
            x, y, w, h = self.roi
            # clamp ROI
            h_frame, w_frame = frame.shape[:2]
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = max(1, min(w, w_frame - x))
            h = max(1, min(h, h_frame - y))
            frame = frame[y:y+h, x:x+w]
        if self.brightness_alpha != 1.0 or self.brightness_beta != 0.0:
            # cv2.convertScaleAbs does: dst = saturate(src*alpha + beta)
            frame = cv2.convertScaleAbs(frame, alpha=self.brightness_alpha, beta=self.brightness_beta)
        return frame

    def make_packet(self, frame):
        timestamp = datetime.now(timezone.utc).isoformat()
        packet = {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp": timestamp,
            "frame": frame_to_base64_jpeg(frame, jpeg_quality=self.jpeg_quality)
        }
        return packet

    def run(self):
        # maximum consecutive read failures before forcing a reconnect
        max_consecutive_failures = 5

        while not self.stop_event.is_set():
            try:
                logger.info(f"[{self.camera_id}] Opening capture to {self.source}")
                try:
                    self.capture = self.open_capture()
                except Exception:
                    # open_capture already logs exception
                    # wait using backoff and retry outer loop
                    delay = min(self.backoff.base * (self.backoff.factor ** self.backoff.attempt), self.backoff.max_delay)
                    jitter = delay * 0.1 * (2 * (random.random() - 0.5))
                    delay = max(0.5, delay + jitter)
                    self.backoff.attempt += 1
                    logger.info(f"[{self.camera_id}] Reconnect scheduled in {delay:.1f}s")
                    time.sleep(delay)
                    continue

                # successfully opened
                self.backoff.reset()
                self.frame_id = 0
                consecutive_failures = 0

                while not self.stop_event.is_set():
                    start = time.time()

                    # read frame
                    try:
                        ret, frame = self.capture.read()
                    except Exception as e:
                        logger.warning(f"[{self.camera_id}] capture.read() raised: {e}")
                        ret, frame = False, None

                    if not ret or frame is None:
                        consecutive_failures += 1
                        logger.warning(f"[{self.camera_id}] Frame read failed ({consecutive_failures}/{max_consecutive_failures})")
                        # small backoff between consecutive read attempts to avoid busy loop
                        time.sleep(0.1)
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning(f"[{self.camera_id}] Too many consecutive read failures -> reconnecting capture")
                            try:
                                if self.capture:
                                    self.capture.release()
                            except Exception:
                                pass
                            break  # break inner loop -> will reconnect by outer loop/backoff
                        else:
                            # try reading again without tearing down the capture
                            continue

                    # successful read -> reset failure counter
                    consecutive_failures = 0

                    # resize to target resolution if necessary (guard against small frames)
                    try:
                        tw, th = self.target_resolution
                        if frame.shape[1] != tw or frame.shape[0] != th:
                            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        # if resize fails, keep original frame
                        pass

                    # apply ROI and brightness adjustments
                    try:
                        frame = self.apply_roi_and_brightness(frame)
                    except Exception:
                        logger.debug(f"[{self.camera_id}] ROI/brightness apply failed", exc_info=True)

                    self.frame_id += 1

                    # form packet and push non-blocking
                    packet = None
                    try:
                        packet = self.make_packet(frame)
                        if self.out_queue is not None:
                            try:
                                self.out_queue.put_nowait(packet)
                            except Exception:
                                try:
                                    # fallback: block shortly
                                    self.out_queue.put(packet, timeout=0.1)
                                except Exception:
                                    logger.debug(f"[{self.camera_id}] out_queue full; dropping frame {self.frame_id}")
                    except Exception:
                        logger.exception(f"[{self.camera_id}] Failed to encode/put packet")
                        # continue processing next frames

                    # enforce target fps
                    elapsed = time.time() - start
                    to_sleep = self.frame_interval - elapsed
                    if to_sleep > 0:
                        time.sleep(to_sleep)

                # end inner capture loop -> either stop_event set or we need to reconnect
            except Exception as e:
                logger.exception(f"[{self.camera_id}] Capture error: {e}")

            # If we reach here, we attempt reconnect with exponential backoff (unless stop_event)
            if self.stop_event.is_set():
                break

            delay = min(self.backoff.base * (self.backoff.factor ** self.backoff.attempt), self.backoff.max_delay)
            jitter = delay * 0.1 * (2 * (random.random() - 0.5))
            delay = max(0.5, delay + jitter)
            self.backoff.attempt += 1
            logger.info(f"[{self.camera_id}] Reconnecting in {delay:.1f}s ...")
            try:
                if self.capture:
                    try:
                        self.capture.release()
                    except Exception:
                        pass
                    self.capture = None
            except Exception:
                pass
            time.sleep(delay)
            # try again (outer while) to open capture

    def stop(self):
        self.stop_event.set()
        try:
            if self.capture:
                try:
                    self.capture.release()
                except Exception:
                    pass
        except Exception:
            pass


# Example usage: run multiple cameras and consume JSON packets
if __name__ == "__main__":
    import signal
    import sys

    # Simple queue for demo (bounded to avoid mem blow)
    out_q = queue.Queue(maxsize=64)

    cams = [
        {"camera_id": "Kitchen_1", "source": "rtsp://user:pass@192.168.1.10:554/stream1"},
        {"camera_id": "Dock_1", "source": "/dev/video0"},
    ]

    workers = []
    for c in cams:
        w = CameraWorker(camera_id=c["camera_id"],
                         source=c["source"],
                         out_queue=out_q,
                         target_fps=5,
                         target_resolution=(1280, 720),
                         roi=None,
                         brightness_alpha=1.0,
                         brightness_beta=0.0)
        w.start()
        workers.append(w)

    def sigint_handler(sig, frame):
        logger.info("Stopping workers...")
        for w in workers:
            w.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    # Consumer loop: print JSON packets (in real life — push to message bus / socket / queue)
    try:
        while True:
            pkt = out_q.get()
            # Here we have a JSON-serializable dict; you can stream it via websocket/zeromq/http
            print(json.dumps(pkt))
            out_q.task_done()
    except KeyboardInterrupt:
        sigint_handler(None, None)