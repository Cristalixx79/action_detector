# kitchen_environment/surface_detector/detector.py

import random
import asyncio


class SurfaceDetector:
    def __init__(self):
        print("[SurfaceDetector] Loaded U-Net/SAM stub")

    async def process(self, frame):
        """
        Здесь будет ML: SAM2 / U-Net / сегментация грязи.
        Пока имитация — случайные значения.
        """

        await asyncio.sleep(0.01)  # микрозадержка для имитации ML

        return {
            "surface_cleanliness": round(random.uniform(0.78, 0.99), 2),
            "waste_detected": random.choice([False, False, False, True]),
            "open_containers": random.randint(0, 2)
        }
