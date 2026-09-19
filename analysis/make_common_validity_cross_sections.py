#!/usr/bin/env python3

"""Create paired cross-section artifacts with an identical shared-bin mask."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.campaign_comparison import common_validity_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Intersect two campaign final-validity masks in exactly matching 4D "
            "bins and write paired cross-section artifacts for common-mask fits."
        )
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-output", type=Path, required=True)
    parser.add_argument("--right-output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_mask(artifact, path: Path) -> np.ndarray:
    values = np.asarray(artifact["reduced_cross_section"], dtype=float)
    uncertainty = np.asarray(artifact["uncertainty"], dtype=float)
    if values.shape != uncertainty.shape:
        raise ValueError(f"{path} cross section and uncertainty shapes differ")
    if "final_validity_mask" in artifact.files:
        mask = np.asarray(artifact["final_validity_mask"], dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(f"{path} final_validity_mask has an incompatible shape")
        return mask
    return np.isfinite(values) & np.isfinite(uncertainty) & (uncertainty > 0.0)


def payload(
    artifact,
    *,
    common_mask: np.ndarray,
    source: Path,
    partner: Path,
    self_q2_indices: np.ndarray,
    partner_q2_indices: np.ndarray,
) -> dict:
    result = {name: artifact[name] for name in artifact.files}
    result["comparison_original_final_validity_mask"] = final_mask(artifact, source)
    result["final_validity_mask"] = common_mask
    result["reduced_cross_section"] = np.where(
        common_mask,
        np.asarray(artifact["reduced_cross_section"], dtype=float),
        np.nan,
    )
    result["uncertainty"] = np.where(
        common_mask,
        np.asarray(artifact["uncertainty"], dtype=float),
        np.nan,
    )
    result["comparison_validity_definition"] = np.asarray(
        "intersection of both campaigns' final_validity_mask in exactly matching "
        "Q2, xB, -t, and phi bins"
    )
    result["comparison_source_cross_section"] = np.asarray(str(source.resolve()))
    result["comparison_source_sha256"] = np.asarray(sha256(source))
    result["comparison_partner_cross_section"] = np.asarray(str(partner.resolve()))
    result["comparison_partner_sha256"] = np.asarray(sha256(partner))
    result["comparison_self_q2_indices"] = self_q2_indices
    result["comparison_partner_q2_indices"] = partner_q2_indices
    return result


def main() -> int:
    args = parse_args()
    left_path = args.left.resolve()
    right_path = args.right.resolve()
    left = np.load(left_path, allow_pickle=False)
    right = np.load(right_path, allow_pickle=False)
    masks = common_validity_masks(
        final_mask(left, left_path),
        final_mask(right, right_path),
        left_q2_edges=left["q2_edges"],
        right_q2_edges=right["q2_edges"],
        left_xb_edges=left["xb_edges"],
        right_xb_edges=right["xb_edges"],
        left_t_edges=left["t_edges"],
        right_t_edges=right["t_edges"],
        left_phi_edges=left["phi_edges"],
        right_phi_edges=right["phi_edges"],
    )
    args.left_output.parent.mkdir(parents=True, exist_ok=True)
    args.right_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.left_output,
        **payload(
            left,
            common_mask=masks.left,
            source=left_path,
            partner=right_path,
            self_q2_indices=masks.left_q2_indices,
            partner_q2_indices=masks.right_q2_indices,
        ),
    )
    np.savez_compressed(
        args.right_output,
        **payload(
            right,
            common_mask=masks.right,
            source=right_path,
            partner=left_path,
            self_q2_indices=masks.right_q2_indices,
            partner_q2_indices=masks.left_q2_indices,
        ),
    )
    print(
        "Matched Q2 bins:",
        ", ".join(
            f"left[{left_index}]=right[{right_index}]"
            for left_index, right_index in zip(
                masks.left_q2_indices, masks.right_q2_indices
            )
        ),
    )
    print(f"Common valid 4D bins: {int(np.count_nonzero(masks.left))}")
    print(f"Wrote {args.left_output}")
    print(f"Wrote {args.right_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
