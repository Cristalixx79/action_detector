# kitchen_environment/run_environment.py

import asyncio
import yaml

from snapshot_collector.collector import SnapshotCollector
from surface_detector.detector import SurfaceDetector
from environment_aggregator.aggregator import EnvironmentAggregator


async def start_camera(camera_config, detector, aggregator):
    """
    Создаёт Collector и связывает его с pipeline обработки.
    """

    camera_id = camera_config["id"]
    zone_id = camera_config["zone"]
    rtsp = camera_config["url"]
    interval = camera_config.get("interval", 30)

    async def pipeline(frame):
        detection = await detector.process(frame)
        await aggregator.push(camera_id, zone_id, detection)

    collector = SnapshotCollector(rtsp_url=rtsp, interval_sec=interval)
    await collector.start(pipeline)


async def main():
    print("[Environment] Started.")

    # загружаем список RTSP-камер
    with open(r"D:\PycharmProjects\kitchen_cv\kitchen_environment\config.yaml", "r") as f:
        config = yaml.safe_load(f)

    detector = SurfaceDetector()
    aggregator = EnvironmentAggregator()

    tasks = []

    for cam in config["rtsp_sources"]:
        tasks.append(
            asyncio.create_task(
                start_camera(cam, detector, aggregator)
            )
        )

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
