import json
import queue
import time
import logging
import uuid
import threading
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO)


class Camera:
    def __init__(self, name):
        self.name = name


class ActionDetector(threading.Thread):
    def __init__(self, cams, in_queue, analyser):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.in_queue = in_queue

        self.__cameras = [Camera(cam) for cam in cams]
        # Список того, с чем может взаимодействовать человек, наверное что-то добавится в будущем
        self.__items = ["knife", "spoon", "desk", "plate", "food", "hat"]
        self.__hand_classes = ["bare_hand", "gloved_hand"]
        # Список камер, где в данный момент времени возможно действие
        self.__action_possible_cameras = set()
        # минимальный уровень confidence, при котором идёт детекция действия
        self.__detection_threshold = 0.5

        self.__pattern_analyser = analyser

        # Предыдущая позиция объектов камеры
        # Пример: {"Kitchen_1": {"person_1": [243.64 -> центр по x, 534.65 -> центр по y, 1764438539.9258504 -> время, \
        # [322.6, 542.54, 12.7, 98.6] -> bbox], ...}, ...}
        self.__previous_position: Dict[str, Dict[str, Tuple[float, float, float, List[float]]]] = defaultdict(dict)

        # Данные о движении объектов на данной камере
        # Пример: {"Kitchen_1": {"person_1": [[134.36 -> центр по x, 352.756 -> центр по y, \
        # 63.865 -> скорость по x, 524.754 -> скорость по y], ...], ...}, ...}
        self.__movement_vectors: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=20)))

        # Данные о действиях, замеченных на данной камере
        self.__detected_actions: Dict[str, Dict[str, Any]] = {}

        self.__ACTION_DETECTION_DELAY = 0.5  # минимальная длительность действия
        self.__ACTION_END_DELAY = 0.4  # задержка перед завершением действия
        self.__ACTION_CHECK_TIMEOUT = 30  # таймаут проверки возможности действия

    def is_action_possible(self, json_data: Dict) -> bool:
        """Проверяет возможно ли действие"""
        camera_id = json_data["camera_id"]
        detected_classes = set()

        for item in json_data["objects"]:
            obj_class = item["class"]
            obj_confidence = item["confidence"]

            if obj_confidence < self.__detection_threshold:
                continue

            detected_classes.add(obj_class)

        # Проверка возможности действия
        has_person = "person" in detected_classes
        has_hand = any(hand in detected_classes for hand in self.__hand_classes)
        has_item = any(item in detected_classes for item in self.__items)

        if has_person and has_hand and has_item:
            self.__action_possible_cameras.add(camera_id)
            self.__detected_actions[camera_id]["timestamp"] = time.time()
            logging.info(f"Action is possible on {camera_id}")
            return True

        # Очистка данных камеры
        self.__clear_cam_data(camera_id)
        return False

    def analise_motion(self, data_json: Dict) -> Optional[Tuple[Dict, List]]:
        """Анализирует движение объектов и возвращает паттерны"""
        camera_id = data_json["camera_id"]

        if not self.action_possible_on_cam(camera_id):
            self.__clear_cam_data(camera_id)
            print(f" -- Error with {camera_id}: \"Action is impossible\"")
            return None

        current_timestamp = time.time()
        object_counter = defaultdict(int)
        center_position_list = []

        for item in data_json["objects"]:
            obj_class = item["class"]
            confidence = item["confidence"]

            if confidence < self.__detection_threshold:
                continue

            # Уникализируем идентификаторы объектов
            object_counter[obj_class] += 1
            unique_id = f"{obj_class}_{object_counter[obj_class]}"

            bbox = self.__parse_bbox(item["bbox"])
            if not bbox:
                continue

            current_center = self.__calculate_bbox_center(bbox)

            # Сохраняем информацию о позиции для определения расстояний
            center_position_list.append({
                "position": current_center,
                "class": obj_class,
                "unique_id": unique_id
            })

            # Анализ движения если есть предыдущая позиция
            if (camera_id in self.__previous_position and
                    unique_id in self.__previous_position[camera_id]):

                prev_data = self.__previous_position[camera_id][unique_id]
                prev_center_x, prev_center_y, prev_time, _ = prev_data
                curr_center_x, curr_center_y = current_center

                # Вычисляем вектор движения
                dx = curr_center_x - prev_center_x
                dy = curr_center_y - prev_center_y

                # Вычисляем скорость
                time_diff = current_timestamp - prev_time
                if time_diff > 0:
                    speed_x = dx / time_diff
                    speed_y = dy / time_diff
                else:
                    speed_x, speed_y = 0, 0

                # Сохраняем вектор движения
                movement_data = (curr_center_x, curr_center_y, speed_x, speed_y)
                self.__movement_vectors[camera_id][unique_id].append(movement_data)

            # Обновляем предыдущую позицию
            self.__previous_position[camera_id][unique_id] = (
                current_center[0], current_center[1], current_timestamp, bbox
            )

        # Анализируем паттерны движения
        if self.__movement_vectors[camera_id]:
            patterns = self.__pattern_analyser.analyze_motion_patterns(
                dict(self.__movement_vectors[camera_id])
            )
        else:
            patterns = {}

        return patterns, center_position_list

    def detect_action(self, patterns: Dict, center_positions: List, json_data: Dict) -> Optional[Dict]:
        """Определяет тип действия на основе паттернов движения"""
        camera_id = json_data["camera_id"]

        if not patterns or not center_positions:
            logging.warning(f"Action detection is impossible on {camera_id}")
            return None

        current_state = self.__detected_actions[camera_id]["state"]
        current_time = time.time()

        # Проверяем наличие взаимодействия руки с предметом
        action_detected = self.__check_hand_item_interaction(patterns, center_positions)

        print(f"Current state on {camera_id}: {current_state}")
        if current_state == "IDLE":
            if action_detected:
                # Начинаем отслеживание потенциального действия
                self.__detected_actions[camera_id].update({
                    "timestamp": current_time,
                    "state": "ACTION_CANDIDATE",
                    "action_type": action_detected["type"],
                    "action_detected": False
                })
            elif current_time - self.__detected_actions[camera_id]["timestamp"] >= self.__ACTION_CHECK_TIMEOUT:
                # Проверяем возможность действия по таймауту
                logging.info(f"No action detected on {camera_id} for {self.__ACTION_CHECK_TIMEOUT}s, rechecking")
                self.is_action_possible(json_data)

        elif current_state == "ACTION_CANDIDATE":
            if action_detected and action_detected["type"] == self.__detected_actions[camera_id]["action_type"]:
                # Проверяем длительность действия
                time_since_start = current_time - self.__detected_actions[camera_id]["timestamp"]
                if time_since_start >= self.__ACTION_DETECTION_DELAY:
                    # Действие подтверждено
                    self.__detected_actions[camera_id].update({
                        "state": "ACTION_ACTIVE",
                        "action_detected": True,
                        "timestamp_start": self.__detected_actions[camera_id]["timestamp"],
                        "timestamp_end": current_time,
                        "action_id": str(uuid.uuid4())
                    })
            else:
                # Сбрасываем если действие прервалось
                self.__reset_camera_state(camera_id)

        elif current_state == "ACTION_ACTIVE":
            if action_detected and action_detected["type"] == self.__detected_actions[camera_id]["action_type"]:
                # Обновляем время окончания
                self.__detected_actions[camera_id]["timestamp_end"] = current_time
            else:
                # Проверяем задержку перед завершением
                time_since_last = current_time - self.__detected_actions[camera_id]["timestamp_end"]
                if time_since_last >= self.__ACTION_END_DELAY:
                    # Действие завершено
                    output_packet = self.make_output_packet(camera_id)
                    self.__reset_camera_state(camera_id)
                    return output_packet

        return None

    def __check_hand_item_interaction(self, patterns: Dict, center_positions: List) -> Optional[Dict]:
        """Проверяет взаимодействие руки с предметом"""
        hands = [pos for pos in center_positions if self.__is_hand(pos["class"])]
        items = [pos for pos in center_positions if self.__is_instrument(pos["class"])]

        for hand in hands:
            for item in items:
                # Проверяем расстояние
                distance = self.__calculate_distance(hand["position"], item["position"])
                if distance > 50:
                    continue

                # Определяем тип действия по паттернам
                hand_pattern = patterns.get(hand["unique_id"], "")
                item_pattern = patterns.get(item["unique_id"], "")

                # Режем (нож + вертикальное/линейное движение)
                if ("knife" in item["class"] and
                        hand_pattern in ["vertical", "linear"] and
                        item_pattern in ["vertical", "linear"]):
                    return {"type": "CUT", "hand": hand, "item": item}

                # Перемешиваем (тарелка + круговое движение руки)
                elif ("plate" in item["class"] and
                      item_pattern == "stationary" and
                      hand_pattern == "circular"):
                    return {"type": "MIX", "hand": hand, "item": item}

                # Подаем (тарелка + линейное движение)
                elif ("plate" in item["class"] and
                      item_pattern == "linear" and
                      hand_pattern == "linear"):
                    return {"type": "SERVE", "hand": hand, "item": item}

        return None

    def make_output_packet(self, camera_id: str) -> Dict:
        packet_uuid = uuid.uuid4()
        # Формирование выходного пакета
        output_packet = {
            "action_id": packet_uuid,
            # Тут должно быть employee_id из ArcFace
            "employee_id": "undefined",
            "camera_id": camera_id,
            # Хз откуда брать это, возможно, при конфигурации приложения будет захардкожено
            "zone_id": "undefined",
            "action_type": self.__detected_actions[camera_id]["action_type"],
            "timestamp_start": self.__detected_actions[camera_id]["timestamp_start"],
            "timestamp_end": self.__detected_actions[camera_id]["timestamp_end"]
        }
        return output_packet

    def run(self):
        while not self.stop_event.is_set():
            try:
                pkt = self.in_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                camera_id = pkt["camera_id"]
                # Проверка на то, находится ли камера в списке, камер, где возможно действие
                if self.action_possible_on_cam(camera_id):
                    cam_data = self.analise_motion(pkt)
                    if cam_data is not None:
                        pattern, distance_list = cam_data
                        self.detect_action(pattern, distance_list, pkt)
                else:
                    self.is_action_possible(pkt)
            except Exception:
                continue

    def action_possible_on_cam(self, cam: str):
        """Проверяет возможно ли действие на данной камере"""
        return cam in self.__action_possible_cameras

    def stop(self):
        self.stop_event.set()

    # =================== тут приватные функции =================== #

    def __clear_cam_data(self, camera_id: str):
        """Очищает данные камеры"""
        if camera_id in self.__action_possible_cameras:
            self.__action_possible_cameras.remove(camera_id)

        self.__previous_position.pop(camera_id, None)
        self.__movement_vectors.pop(camera_id, None)
        self.__reset_camera_state(camera_id)

    def __reset_camera_state(self, camera_id: str):
        """Сбрасывает состояние камеры к дефолтному"""
        self.__detected_actions[camera_id] = {
            "timestamp": time.time(),
            "state": "IDLE",
            "action_detected": False,
            "action_type": "NONE",
            "timestamp_start": 0,
            "timestamp_end": 0,
            "action_id": None
        }

    def __get_index(self, cam: str):
        """Находит индекс камеры по её названию"""
        for c in range(len(self.__cameras)):
            if self.__cameras[c].name == cam:
                return c
        return -1

    def __is_instrument(self, item: str):
        """Проверяет на инструмент"""
        for i in self.__items:
            if str(item).startswith(i):
                return True
        return False

    def __is_hand(self, item: str):
        """Проверяет на руку"""
        return (str(item).startswith("bare_hand")) or (str(item).startswith("gloved_hand"))

    def __calculate_bbox_center(self, bbox: Any):
        """Вычисляет центр bounding box"""
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def __calculate_bbox_area(self, bbox: Any):
        """Вычисляет площадь bounding box"""
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def __check_items(self, item: str):
        """Проверяет есть ли в кадре нужные для потенциального действия объекты"""
        if item not in self.__items:
            return False
        return True

    def __calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Вычисляет евклидово расстояние между двумя точками"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def __parse_bbox(self, bbox_str: str) -> Optional[List[float]]:
        """Парсит строку с bounding box"""
        try:
            if isinstance(bbox_str, str):
                # Убираем квадратные скобки и разбиваем
                bbox_str = bbox_str.strip("[]")
                return [float(x) for x in bbox_str.split(", ")]
            elif isinstance(bbox_str, list):
                return [float(x) for x in bbox_str]
            return None
        except (ValueError, AttributeError):
            logging.error(f"Failed to parse bbox: {bbox_str}")
            return None
