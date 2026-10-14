#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED_SCHEMAS = {
    "elastic_momentum_correction/v1",
    "particle_momentum_correction/v2",
}


def merge_parameter_files(paths: list[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one parameter file is required")
    payloads = [json.loads(path.read_text()) for path in paths]
    first = payloads[0]
    beam_energy = float(first["beamEnergyGeV"])
    torus = int(first["torus"])
    regions: list[dict[str, object]] = []
    keys: set[tuple[int, int, int]] = set()
    for path, payload in zip(paths, payloads):
        if payload.get("schema") not in SUPPORTED_SCHEMAS:
            raise ValueError(f"unsupported schema in {path}: {payload.get('schema')}")
        if payload.get("correctionType") != "fractionalMomentum":
            raise ValueError(f"unsupported correction type in {path}")
        if abs(float(payload["beamEnergyGeV"]) - beam_energy) > 1.0e-9:
            raise ValueError("parameter files have different beam energies")
        if int(payload["torus"]) != torus:
            raise ValueError("parameter files have different torus polarities")
        for region in payload.get("regions", []):
            key = (
                int(region["pid"]), int(region["detector"]),
                int(region.get("sector", 0)),
            )
            overlapping = any(
                prior_pid == key[0] and prior_detector == key[1] and
                (prior_sector == key[2] or prior_sector == 0 or key[2] == 0)
                for prior_pid, prior_detector, prior_sector in keys
            )
            if overlapping:
                raise ValueError(
                    "duplicate or overlapping pid/detector/sector region while merging: "
                    f"{key}"
                )
            keys.add(key)
            regions.append(region)
    return {
        "schema": "particle_momentum_correction/v2",
        "correctionType": "fractionalMomentum",
        "beamEnergyGeV": beam_energy,
        "torus": torus,
        "calibrationRole": "mergedValidatedStages",
        "stageOrder": [
            "protonEnergyLoss", "electronMomentum", "protonMomentum",
            "photonEnergy",
        ],
        "sources": [str(path) for path in paths],
        "regions": regions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge non-overlapping electron, proton, and photon correction "
            "regions into one production parameter file."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_parameter_files(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    particles = sorted({int(region["pid"]) for region in result["regions"]})
    print(
        f"Wrote {len(result['regions'])} regions for PIDs {particles} to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
