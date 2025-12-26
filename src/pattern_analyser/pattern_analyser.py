import numpy as np
import math


class MotionPatternAnalyzer:
    """Анализирует паттерны движения объектов"""
    def __init__(self, min_samples=20, circle_threshold=0.7, stability_threshold=0.8):
        self.min_samples = min_samples
        self.circle_threshold = circle_threshold
        self.stability_threshold = stability_threshold

    def analyze_motion_patterns(self, motion_data):
        """Анализирует паттерны движения для всех объектов во всех камерах"""
        patterns = {}

        for obj_class, obj_trajectory in motion_data.items():
            if len(obj_trajectory) < self.min_samples:
                patterns[obj_class] = "insufficient_data"
                continue

            pattern = self._analyze_single_object_pattern(obj_trajectory)
            patterns[obj_class] = pattern

        return patterns

    def _analyze_single_object_pattern(self, trajectory):
        """Анализирует паттерн движения для одного объекта"""
        # Извлекаем данные из траектории
        positions = np.array([[point[0], point[1]] for point in trajectory])  # x, y
        velocities = np.array([[point[2], point[3]] for point in trajectory])  # vx, vy

        patterns = []

        if self._is_horizontal_motion(velocities):
            patterns.append("horizontal")

        if self._is_vertical_motion(velocities):
            patterns.append("vertical")

        if self._is_circular_motion(positions):
            patterns.append("circular")

        if self._is_stationary(velocities):
            patterns.append("stationary")

        if self._is_linear_motion(positions):
            patterns.append("linear")

        if self._is_chaotic_motion(velocities):
            patterns.append("chaotic")

        if not patterns:
            return "undefined"

        # Если есть несколько паттернов, выбираем наиболее вероятный
        if len(patterns) == 1:
            return patterns[0]
        else:
            return self._resolve_multiple_patterns(patterns)

    def _is_horizontal_motion(self, velocities):
        """Проверяет преобладание горизонтального движения"""
        vx = velocities[:, 0]  # скорости по X
        vy = velocities[:, 1]  # скорости по Y

        # Вычисляем отношение горизонтальной скорости к общей
        horizontal_ratio = np.mean(np.abs(vx) / (np.abs(vx) + np.abs(vy) + 1e-8))
        return horizontal_ratio > 0.7

    def _is_vertical_motion(self, velocities):
        """Проверяет преобладание вертикального движения"""
        vx = velocities[:, 0]  # скорости по X
        vy = velocities[:, 1]  # скорости по Y

        # Вычисляем отношение вертикальной скорости к общей
        vertical_ratio = np.mean(np.abs(vy) / (np.abs(vx) + np.abs(vy) + 1e-8))
        return vertical_ratio > 0.7

    def _is_circular_motion(self, positions):
        """Проверяет круговое движение через анализ кривизны и угловой скорости"""
        if len(positions) < 3:
            return False

        # Вычисляем угловую скорость
        angular_velocities = []
        for i in range(1, len(positions) - 1):
            # Векторы между последовательными точками
            v1 = positions[i] - positions[i - 1]
            v2 = positions[i + 1] - positions[i]

            # Угол между векторами
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angular_velocities.append(angle)

        if not angular_velocities:
            return False

        # Средняя угловая скорость
        mean_angular_velocity = np.mean(angular_velocities)

        # Проверяем постоянство направления движения
        directions = []
        for i in range(len(positions) - 1):
            dx = positions[i + 1, 0] - positions[i, 0]
            dy = positions[i + 1, 1] - positions[i, 1]
            direction = math.atan2(dy, dx)
            directions.append(direction)

        # Вычисляем изменение направления
        direction_changes = []
        for i in range(len(directions) - 1):
            change = abs(directions[i + 1] - directions[i])
            if change > math.pi:
                change = 2 * math.pi - change
            direction_changes.append(change)

        mean_direction_change = np.mean(direction_changes) if direction_changes else 0

        # Круговое движение характеризуется постоянным изменением направления
        return (mean_angular_velocity > 0.2 and
                0.1 < mean_direction_change < 1.0)

    def _is_stationary(self, velocities):
        """Проверяет, находится ли объект в стационарном состоянии"""
        speeds = np.linalg.norm(velocities, axis=1)
        mean_speed = np.mean(speeds)
        return mean_speed < 2.0  # порог скорости в пикселях/сек

    def _is_linear_motion(self, positions):
        """Проверяет линейное движение через анализ корреляции"""
        if len(positions) < 3:
            return False

        x = positions[:, 0]
        y = positions[:, 1]

        # Линейная регрессия для проверки линейности
        if len(np.unique(x)) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            # Высокая корреляция или антикорреляция указывает на линейное движение
            return abs(correlation) > 0.8
        return False

    def _is_chaotic_motion(self, velocities):
        """Проверяет хаотичное/нерегулярное движение"""
        speeds = np.linalg.norm(velocities, axis=1)
        speed_variance = np.var(speeds)

        # Вычисляем изменение направления
        directions = []
        for vx, vy in velocities:
            if abs(vx) + abs(vy) > 0.1:
                direction = math.atan2(vy, vx)
                directions.append(direction)

        if len(directions) < 2:
            return False

        direction_changes = []
        for i in range(len(directions) - 1):
            change = abs(directions[i + 1] - directions[i])
            if change > math.pi:
                change = 2 * math.pi - change
            direction_changes.append(change)

        mean_direction_change = np.mean(direction_changes)

        # Хаотичное движение: высокая дисперсия скорости и частые изменения направления
        return speed_variance > 10.0 and mean_direction_change > 0.5

    def _resolve_multiple_patterns(self, patterns):
        """Разрешает конфликт при обнаружении нескольких паттернов"""
        # Приоритет паттернов
        priority_order = ["circular", "chaotic", "linear", "horizontal", "vertical", "stationary"]

        for pattern in priority_order:
            if pattern in patterns:
                return pattern

        return patterns[0]  # возвращаем первый обнаруженный