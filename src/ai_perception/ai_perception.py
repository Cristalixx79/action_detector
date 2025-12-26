# ai_perception.py
import base64
import json
import os
import threading
from datetime import datetime, timezone
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_perception")

# Try to import ultralytics YOLO (v8)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not available. Install via `pip install ultralytics` to enable YOLO inference.")

# ByteTrack optional stub
BYTETRACK_AVAILABLE = False
try:
    from bytetrack import BYTETracker  # type: ignore
    BYTETRACK_AVAILABLE = True
except Exception:
    BYTETRACK_AVAILABLE = False
    logger.info("ByteTrack not available — using SimpleTracker fallback.")

# Device selection
def select_device():
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Using device: cuda")
            return "cuda"
        else:
            logger.info("Using device: cpu")
            return "cpu"
    except Exception:
        logger.info("torch not available; defaulting to cpu")
        return "cpu"

DEVICE = select_device()

# Canonical desired classes (as required)
CANONICAL_CLASSES = {
    "person",
    "gloved_hand",
    "bare_hand",
    "knife",
    "cutting_board",
    "hat",
    "food"
}

# Synonym mapping: model-specific class names -> canonical class
# Add more synonyms if needed for your custom models
SYNONYM_MAP = {
    # food / edible categories from COCO -> 'food'
    "banana": "food", "apple": "food", "sandwich": "food", "orange": "food",
    "broccoli": "food", "carrot": "food", "hot dog": "food", "hot_dog": "food",
    "pizza": "food", "donut": "food", "cake": "food", "bowl": "food",
    # COCO names already available:
    "knife": "knife", "spoon": "spoon", "cup": "food", "bottle": "food",
    # garments / gloves mapping
    "glove": "glove", "kitchen_glove": "glove", "oven_glove": "glove",
    "hat": "hat", "cap": "hat",
    "cutting_board": "cutting_board", "board": "cutting_board",
    # hands
    "hand": "hand", "hands": "hand"
}

# приводим исходное имя класса к каноническому (в нижнем регистре)
def normalize_class_name(raw_name: str):
    if raw_name is None:
        return None
    name = str(raw_name).lower().strip()
    # канонические классы
    if name in CANONICAL_CLASSES:
        return name
    # синонимы
    if name in SYNONYM_MAP:
        return SYNONYM_MAP[name]
    name_simple = name.replace("_", " ")
    if name_simple in SYNONYM_MAP:
        return SYNONYM_MAP[name_simple]
    if name in {"apple", "banana", "pizza", "donut", "cake", "sandwich", "orange", "broccoli", "carrot", "hot dog", "hot_dog"}:
        return "food"
    return name

# Simple tracker fallback
class SimpleTracker:
    def __init__(self):
        self._next_id = 1

    def update(self, detections):
        # assign incremental ids
        for d in detections:
            if "id" not in d or d.get("id") is None:
                d["id"] = self._next_id
                self._next_id += 1
        return detections

# MobileNetV3 classifier loader (optional)
def load_mobilenetv3_classifier(model_path=None):
    try:
        import torch
        from torchvision import models, transforms
        # prefer weights API
        try:
            weights_enum = getattr(models, "MobileNet_V3_Small_Weights", None)
            if weights_enum:
                weights = weights_enum.DEFAULT
                model = models.mobilenet_v3_small(weights=weights)
            else:
                model = models.mobilenet_v3_small(pretrained=True)
        except Exception:
            model = models.mobilenet_v3_small(pretrained=True)
        model.eval()
        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return {"model": model, "preprocess": preprocess, "torch": torch}
    except Exception:
        logger.exception("Failed to load MobileNetV3 classifier (torch/torchvision required)")
        return None

# IoU util for deduplication
def iou_xyxy(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union

class PerceptionWorker(threading.Thread):
    def __init__(self, in_queue, out_queue=None, model_dir="../../res/models", use_deepsort=False, tracker_backend="simple"):
        """
        in_queue: queue.Queue() where ingestion puts packets:
                  {"camera_id": str, "frame_id": int, "timestamp": str, "frame": base64_jpeg}
        out_queue: queue.Queue() where this worker will put result JSON dicts (or None to print)
        model_dir: directory to load models from and watch for hot-reload
        """
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model_dir = model_dir
        self.use_deepsort = use_deepsort
        self.stop_event = threading.Event()

        # models and names
        self.primary_yolo = None
        self.extra_models = []  # list of tuples (model, names_dict)
        self.class_names_primary = {}
        self.tracker = None
        self.classifier = None

        # load models
        self.load_models()

        # tracker
        if tracker_backend == "bytetrack" and BYTETRACK_AVAILABLE:
            try:
                self.tracker = BYTETracker()
                logger.info("ByteTrack initialized.")
            except Exception:
                logger.exception("Failed to init ByteTrack; falling back to SimpleTracker.")
                self.tracker = SimpleTracker()
        else:
            self.tracker = SimpleTracker()

        # hot-reload (watchdog)
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            self._observer = Observer()
            class ModelChangeHandler(FileSystemEventHandler):
                def __init__(self, outer):
                    self.outer = outer
                def on_modified(self, event):
                    logger.info("Model dir modified, reloading models")
                    self.outer.load_models()
                def on_created(self, event):
                    logger.info("Model file created, reloading models")
                    self.outer.load_models()
            handler = ModelChangeHandler(self)
            os.makedirs(self.model_dir, exist_ok=True)
            self._observer.schedule(handler, self.model_dir, recursive=True)
            self._observer.start()
            logger.info("Watchdog observer started for model hot-reload.")
        except Exception:
            self._observer = None
            logger.info("watchdog not available; hot-reload disabled.")

    def load_models(self):
        """
        Загружает модели YOLO строго по указанным вручную путям.
        """

        # Очистка
        self.primary_yolo = None
        self.extra_models = []
        self.class_names_primary = {}

        if not ULTRALYTICS_AVAILABLE:
            logger.warning("ultralytics not available — YOLO will not run.")
            return

        # ============================
        # ПУТИ К МОДЕЛЯМ
        # ============================
        MODEL_PATHS = [
            r"../../res/models/yolov8s.pt",
            r"../../res/models/best_glove_model.pt",
            r"../../res/models/cutting_boards.pt",
        ]

        # ----------------------------
        # Загружаем первую как primary
        # ----------------------------
        try:
            primary_path = MODEL_PATHS[0]
            logger.info(f"Loading PRIMARY YOLO model: {primary_path}")
            self.primary_yolo = YOLO(primary_path)

            names = getattr(self.primary_yolo, "names", {})
            self.class_names_primary = {int(k): str(v) for k, v in names.items()}
        except Exception:
            logger.exception("Failed to load PRIMARY YOLO model")
            self.primary_yolo = None
            self.class_names_primary = {}

        # ----------------------------
        # Загружаем остальные как extras
        # ----------------------------
        for path in MODEL_PATHS[1:]:
            try:
                logger.info(f"Loading EXTRA YOLO model: {path}")
                m = YOLO(path)
                names = getattr(m, "names", {})
                names_dict = {int(k): str(v) for k, v in names.items()}
                self.extra_models.append((m, names_dict))
            except Exception:
                logger.exception(f"Failed to load extra YOLO model: {path}")

        logger.info(f"Loaded {1 + len(self.extra_models)} YOLO models total.")

        # === MobileNet (если нужен) ===
        try:
            clf_path = os.path.join("../../res/models", "mobilenetv3.pth")
            clf = load_mobilenetv3_classifier(clf_path)
            if clf:
                self.classifier = clf
                logger.info("MobileNetV3 classifier loaded.")
            else:
                self.classifier = None
        except Exception:
            self.classifier = None

        logger.info(
            f"Primary classes: {list(self.class_names_primary.values())[:20]} (total {len(self.class_names_primary)})")
        logger.info(
            f"Extra models loaded: {[path for (_, path) in self.extra_models]} (count={len(self.extra_models)})")

    def decode_frame(self, b64jpeg):
        try:
            data = base64.b64decode(b64jpeg)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def run_yolo_on_model(self, model, names_dict, frame, conf_thresh=0.55, imgsz=640):
        detections = []
        try:
            results = model.predict(frame, imgsz=imgsz, conf=conf_thresh, device=DEVICE, verbose=False)
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    # coordinates
                    try:
                        xyxy = box.xyxy.cpu().numpy().tolist()[0]
                    except Exception:
                        try:
                            xyxy = box.xyxy[0].tolist()
                        except Exception:
                            continue

                    # conf
                    try:
                        conf = float(box.conf.cpu().numpy()[0])
                    except Exception:
                        conf = float(box.conf[0])

                    # class idx
                    try:
                        cls_idx = int(box.cls.cpu().numpy()[0])
                    except Exception:
                        cls_idx = int(box.cls[0])

                    raw_name = names_dict.get(cls_idx, str(cls_idx))
                    canonical = normalize_class_name(raw_name)

                    detections.append({
                        "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        "confidence": float(conf),
                        "raw_class": raw_name,
                        "class": canonical
                    })
        except Exception:
            logger.exception("YOLO model predict failed for one model.")
        return detections

    def merge_and_dedup(self, dets_list, iou_thresh=0.5):
        """
        dets_list: список параметров обнаружения из разных моделей.
        Объединяем их, удалив дубликаты (IoU > iou_thresh и один и тот же канонический класс).
        Поддерживает обнаружение с большей достоверностью.
        """
        merged = []
        for d in sorted(dets_list, key=lambda x: -x["confidence"]):
            keep = True
            for m in merged:
                if d["class"] == m["class"]:
                    if iou_xyxy(d["bbox"], m["bbox"]) > iou_thresh:
                        keep = False
                        break
            if keep:
                merged.append(d.copy())
        return merged

    def detect(self, frame, conf_thresh=0.55, imgsz=640):
        all_dets = []

        # основная модель
        if self.primary_yolo is not None:
            names = self.class_names_primary
            dets = self.run_yolo_on_model(self.primary_yolo, names, frame, conf_thresh=conf_thresh, imgsz=imgsz)
            all_dets.extend(dets)

        # доп модели
        for (m, names) in self.extra_models:
            dets = self.run_yolo_on_model(m, names, frame, conf_thresh=conf_thresh, imgsz=imgsz)
            all_dets.extend(dets)

        merged = self.merge_and_dedup(all_dets, iou_thresh=0.5)

        filtered = []
        for d in merged:
            if d["class"] in CANONICAL_CLASSES:
                filtered.append({
                    "bbox": d["bbox"],
                    "confidence": d["confidence"],
                    "class": d["class"]
                })

        return filtered

    def classify_crop(self, crop):
        if self.classifier is None:
            return None
        try:
            torch = self.classifier["torch"]
            model = self.classifier["model"]
            preprocess = self.classifier["preprocess"]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp = preprocess(crop_rgb).unsqueeze(0)
            with torch.no_grad():
                out = model(inp)
                prob = torch.nn.functional.softmax(out[0], dim=0)
                topk = torch.topk(prob, k=1)
                idx = int(topk.indices[0].cpu().numpy())
                score = float(topk.values[0].cpu().numpy())
                return {"label_id": idx, "score": score}
        except Exception:
            logger.exception("Classification failed")
            return None

    def make_output_packet(self, camera_id, timestamp, objects):
        return {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "objects": objects
        }

    def run(self):
        logger.info("PerceptionWorker started.")
        frame_count = 0
        while not self.stop_event.is_set():
            try:
                pkt = self.in_queue.get(timeout=1.0)
            except Exception:
                continue

            try:
                camera_id = pkt.get("camera_id", "unknown")
                timestamp_in = pkt.get("timestamp", datetime.now(timezone.utc).isoformat())
                b64 = pkt.get("frame")
                if b64 is None:
                    continue
                img = self.decode_frame(b64)
                if img is None:
                    logger.warning(f"[{camera_id}] Failed to decode frame")
                    continue

                # Обнаружение запуска (основное + дополнительное)
                detections = self.detect(img, conf_thresh=0.25, imgsz=640)
                logger.info(f"[{camera_id}] Detections after merge/filter: {len(detections)}")

                # Опционное отслеживание классификации по каждому объекту (присвоение идентификаторов)
                tracked = self.tracker.update(detections)

                # ===== ВИЗУАЛИЗАЦИЯ =====
                for det in tracked:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    cls_name = det.get("class", "?")
                    conf = det.get("confidence", 0)
                    label = f"{cls_name} {conf:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                cv2.imshow(f"YOLO Detection - {camera_id}", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Stopping perception worker by user request (q pressed).")
                    self.stop()

                # Классификация по каждому объекту
                objects_out = []
                for o in tracked:
                    obj = {
                        "id": o.get("id"),
                        "class": o.get("class"),
                        "bbox": [float(x) for x in o.get("bbox", [])],
                        "confidence": float(o.get("confidence", 0.0))
                    }

                    # дополнительный классификатор (например, для человека или для проверки одежды)
                    try:
                        if self.classifier:
                            x1, y1, x2, y2 = map(int, obj["bbox"])
                            h, w = img.shape[:2]
                            x1 = max(0, min(w - 1, x1))
                            x2 = max(0, min(w - 1, x2))
                            y1 = max(0, min(h - 1, y1))
                            y2 = max(0, min(h - 1, y2))
                            if x2 > x1 and y2 > y1:
                                crop = img[y1:y2, x1:x2]
                                clf_res = self.classify_crop(crop)
                                if clf_res:
                                    obj["classifier"] = clf_res
                    except Exception:
                        logger.debug("Per-object classification failed", exc_info=True)

                    objects_out.append(obj)

                # Формирование и вывод JSON-пакета

                # Создаём основной пакет
                out_pkt = self.make_output_packet(camera_id, timestamp_in, objects_out)

                # Добавляем кадр, чтобы run_multi_camera мог его отобразить
                out_pkt["frame_raw"] = img

                # Отправляем пакет в очередь
                if self.out_queue:
                    try:
                        self.out_queue.put_nowait(out_pkt)
                    except Exception:
                        try:
                            self.out_queue.put(out_pkt, timeout=0.1)
                        except Exception:
                            logger.debug("Failed to put out_pkt to out_queue (dropped).")

                objects = []
                for o in out_pkt["objects"]:
                    o_pretty = o.copy()
                    if "bbox" in o_pretty:
                        # форматируем bbox как строку с округлением
                        o_pretty["bbox"] = "[" + ", ".join(f"{x:.2f}" for x in o_pretty["bbox"]) + "]"
                    objects.append(o_pretty)

                frame_count += 1

            except Exception:
                logger.exception("Perception processing error")

        logger.info("PerceptionWorker stopping.")

    def stop(self):
        self.stop_event.set()
        if getattr(self, "_observer", None):
            try:
                self._observer.stop()
                self._observer.join(timeout=1.0)
            except Exception:
                pass


