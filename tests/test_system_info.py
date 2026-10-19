import json
import unittest

from system_info import collect_system_info


class SystemInfoTest(unittest.TestCase):
    def test_snapshot_is_json_serializable(self) -> None:
        snapshot = collect_system_info()
        self.assertIn("gpus", snapshot)
        json.dumps(snapshot)
