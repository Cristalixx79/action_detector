# kitchen_environment/environment_aggregator/aggregator.py

import datetime
import json


class EnvironmentAggregator:

    async def push(self, camera_id, zone_id, detect_result):
        """
        Собирает финальный JSON-пакет, который потом:
        - будет отправлен в Gateway
        - может быть логирован в Clickhouse/DB
        """
        event = {
            "camera_id": camera_id,
            "zone_id": zone_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            **detect_result
        }

        print("[EnvironmentEvent]", json.dumps(event, ensure_ascii=False))
        return event
