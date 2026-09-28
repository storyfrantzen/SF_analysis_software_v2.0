from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_momentum_corrections import merge_parameter_files


class MergeMomentumCorrectionTests(unittest.TestCase):
    @staticmethod
    def _payload(pid: int, detector: int) -> dict[str, object]:
        return {
            "schema": "elastic_momentum_correction/v1",
            "correctionType": "fractionalMomentum",
            "beamEnergyGeV": 6.535,
            "torus": 1,
            "regions": [{"pid": pid, "detector": detector, "sector": 0}],
        }

    def test_merges_nonoverlapping_particles_and_promotes_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, payload in enumerate((self._payload(11, 1), self._payload(22, 0))):
                path = Path(directory) / f"p{index}.json"
                path.write_text(json.dumps(payload))
                paths.append(path)
            merged = merge_parameter_files(paths)
        self.assertEqual(merged["schema"], "particle_momentum_correction/v2")
        self.assertEqual([region["pid"] for region in merged["regions"]], [11, 22])

    def test_rejects_duplicate_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(2):
                path = Path(directory) / f"p{index}.json"
                path.write_text(json.dumps(self._payload(11, 1)))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                merge_parameter_files(paths)


if __name__ == "__main__":
    unittest.main()
