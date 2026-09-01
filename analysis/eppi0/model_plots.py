from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .bin_centering import AAO_CROSS_SECTION_CONVERSION
from .binning import AnalysisBinning
from .structure_functions import (
    STRUCTURE_FUNCTION_NAMES,
    epsilon_from_xb_q2,
    harmonic_reference_coordinates,
    harmonic_to_structure_functions,
)


Array = np.ndarray


def _text(data, name: str, fallback: str) -> str:
    if name not in data.files:
        return fallback
    value = np.asarray(data[name])
    return str(value.item()) if value.shape == () else str(value)


def _same_edges(left, right, name: str) -> Array:
    first = np.asarray(left[name], dtype=float)
    second = np.asarray(right[name], dtype=float)
    if first.shape != second.shape or not np.allclose(first, second, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"comparison artifacts have different {name}")
    return first


def _harmonic_chi2(data: Array, model: Array, covariance: Array) -> float:
    if not (
        np.all(np.isfinite(data))
        and np.all(np.isfinite(model))
        and np.all(np.isfinite(covariance))
    ):
        return np.nan
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        return np.nan
    residual = data - model
    value = float(residual @ inverse @ residual)
    return value if np.isfinite(value) and value >= 0.0 else np.nan


def render_model_comparison(
    cross_section_path: str | Path,
    harmonics_path: str | Path,
    model_paths: list[str | Path],
    *,
    beam_energy: float,
    output_dir: str | Path,
    include_quality_rejected: bool = False,
) -> tuple[int, int, Path]:
    """Render phi and structure-function comparisons plus a numerical CSV."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    cross_section = np.load(cross_section_path, allow_pickle=False)
    harmonics = np.load(harmonics_path, allow_pickle=False)
    models = [np.load(path, allow_pickle=False) for path in model_paths]
    q2_edges = _same_edges(cross_section, harmonics, "q2_edges")
    xb_edges = _same_edges(cross_section, harmonics, "xb_edges")
    t_edges = _same_edges(cross_section, harmonics, "t_edges")
    phi_edges = _same_edges(cross_section, harmonics, "phi_edges")
    binning = AnalysisBinning(q2_edges, xb_edges, t_edges, phi_edges)

    data_values = np.asarray(cross_section["reduced_cross_section"], dtype=float)
    data_uncertainties = np.asarray(cross_section["uncertainty"], dtype=float)
    data_parameters = np.asarray(harmonics["parameters"], dtype=float)
    data_covariance = np.asarray(harmonics["covariance"], dtype=float)
    expected_3d = binning.shape[:3]
    for label, values, expected in (
        ("cross-section values", data_values, binning.shape),
        ("cross-section uncertainties", data_uncertainties, binning.shape),
        ("harmonic parameters", data_parameters, expected_3d + (3,)),
        ("harmonic covariance", data_covariance, expected_3d + (3, 3)),
    ):
        if values.shape != expected:
            raise ValueError(f"{label} have shape {values.shape}; expected {expected}")
    data_fit_success = (
        np.asarray(harmonics["fit_success"], dtype=bool)
        if "fit_success" in harmonics.files
        else np.all(np.isfinite(data_parameters), axis=-1)
    )
    data_fit_mask = data_fit_success.copy()
    if "quality_mask" in harmonics.files and not include_quality_rejected:
        data_fit_mask &= np.asarray(harmonics["quality_mask"], dtype=bool)
    if "final_validity_mask" in cross_section.files:
        data_point_mask = np.asarray(cross_section["final_validity_mask"], dtype=bool)
    else:
        data_point_mask = (
            np.isfinite(data_values)
            & np.isfinite(data_uncertainties)
            & (data_uncertainties > 0.0)
        )

    q2_reference, xb_reference = harmonic_reference_coordinates(
        binning, cross_section
    )
    epsilon = epsilon_from_xb_q2(q2_reference, xb_reference, beam_energy)
    data_structure = harmonic_to_structure_functions(
        data_parameters, data_covariance, epsilon
    )
    data_sf_mask = data_fit_mask & data_structure.valid

    model_records = []
    for index, model in enumerate(models):
        for name, edges in (
            ("q2_edges", q2_edges),
            ("xb_edges", xb_edges),
            ("t_edges", t_edges),
            ("phi_edges", phi_edges),
        ):
            candidate = np.asarray(model[name], dtype=float)
            if candidate.shape != edges.shape or not np.allclose(
                candidate, edges, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"model artifact has incompatible {name}")
        name = _text(model, "model_name", f"model_{index + 1}")
        if _text(model, "model_source_kind", "") == "aao_executable" and _text(
            model, "aao_cross_section_conversion", ""
        ) != AAO_CROSS_SECTION_CONVERSION:
            raise ValueError(
                f"AAO model artifact {name!r} lacks the required angular-to-t "
                "cross-section conversion; regenerate it with model-prediction"
            )
        if "beam_energy" in model.files and not np.isclose(
            float(np.asarray(model["beam_energy"]).item()),
            beam_energy,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(f"model artifact {name!r} has a different beam energy")
        model_values = np.asarray(model["reduced_cross_section"], dtype=float)
        model_reliable = np.asarray(model["reliable"], dtype=bool)
        model_parameters = np.asarray(model["parameters"], dtype=float)
        model_fit_success = np.asarray(model["fit_success"], dtype=bool)
        model_structure = np.asarray(model["structure_functions"], dtype=float)
        model_structure_valid = np.asarray(
            model["structure_function_valid"], dtype=bool
        )
        for label, values, expected in (
            ("reduced_cross_section", model_values, binning.shape),
            ("reliable", model_reliable, binning.shape),
            ("parameters", model_parameters, expected_3d + (3,)),
            ("fit_success", model_fit_success, expected_3d),
            ("structure_functions", model_structure, expected_3d + (3,)),
            ("structure_function_valid", model_structure_valid, expected_3d),
        ):
            if values.shape != expected:
                raise ValueError(
                    f"model artifact {name!r} {label} has shape {values.shape}; "
                    f"expected {expected}"
                )
        model_records.append(
            {
                "name": name,
                "values": model_values,
                "reliable": model_reliable,
                "parameters": model_parameters,
                "fit_success": model_fit_success,
                "structure": model_structure,
                "structure_valid": model_structure_valid,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phi_pdf = output_dir / "model_comparison_vs_phi.pdf"
    structure_pdf = output_dir / "model_comparison_structure_functions.pdf"
    csv_path = output_dir / "model_comparison_summary.csv"
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    phi_curve = np.linspace(float(phi_edges[0]), float(phi_edges[-1]), 721)
    phi_radians = np.deg2rad(phi_curve)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    csv_rows: list[list[object]] = [
        [
            "model",
            "iq2",
            "q2_low",
            "q2_high",
            "ixb",
            "xb_low",
            "xb_high",
            "it",
            "t_low",
            "t_high",
            "epsilon",
            "harmonic_chi2",
            "A_data",
            "A_model",
            "B_data",
            "B_model",
            "C_data",
            "C_model",
            "sigma_U_data",
            "sigma_U_uncertainty",
            "sigma_U_model",
            "sigma_U_pull",
            "sigma_LT_data",
            "sigma_LT_uncertainty",
            "sigma_LT_model",
            "sigma_LT_pull",
            "sigma_TT_data",
            "sigma_TT_uncertainty",
            "sigma_TT_model",
            "sigma_TT_pull",
        ]
    ]

    phi_pages = 0
    with PdfPages(phi_pdf) as pdf:
        for grid_index in np.ndindex(binning.shape[:3]):
            point_mask = data_point_mask[grid_index]
            visible_model = any(
                record["fit_success"][grid_index] for record in model_records
            )
            if np.count_nonzero(point_mask) < 1 or not visible_model:
                continue
            iq2, ixb, it = grid_index
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.errorbar(
                phi_centers[point_mask],
                data_values[grid_index][point_mask],
                yerr=data_uncertainties[grid_index][point_mask],
                fmt="o",
                color="black",
                ecolor="black",
                capsize=2,
                markersize=4,
                label="data",
                zorder=5,
            )
            if data_fit_mask[grid_index]:
                a, b, c = data_parameters[grid_index]
                ax.plot(
                    phi_curve,
                    a + b * np.cos(phi_radians) + c * np.cos(2.0 * phi_radians),
                    color="black",
                    linewidth=1.1,
                    alpha=0.7,
                    label="data fit",
                )
            for model_index, record in enumerate(model_records):
                if not record["fit_success"][grid_index]:
                    continue
                color = colors[model_index % len(colors)]
                a, b, c = record["parameters"][grid_index]
                curve = a + b * np.cos(phi_radians) + c * np.cos(2.0 * phi_radians)
                ax.plot(
                    phi_curve,
                    curve,
                    color=color,
                    linewidth=1.5,
                    label=record["name"],
                )
                model_points = record["reliable"][grid_index]
                if np.any(model_points):
                    ax.plot(
                        phi_centers[model_points],
                        record["values"][grid_index][model_points],
                        "s",
                        color=color,
                        markersize=2.5,
                        alpha=0.65,
                    )
            ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
            ax.set_xlim(float(phi_edges[0]), float(phi_edges[-1]))
            ax.set_xlabel("Trento phi [deg]")
            ax.set_ylabel("Reduced cross section [nb/(GeV^2 rad)]")
            ax.set_title(
                "Data and forward-averaged model harmonics\n"
                f"Q2 {q2_edges[iq2]:g}-{q2_edges[iq2 + 1]:g}, "
                f"xB {xb_edges[ixb]:g}-{xb_edges[ixb + 1]:g}, "
                f"-t {t_edges[it]:g}-{t_edges[it + 1]:g}"
            )
            ax.grid(True, alpha=0.22)
            ax.legend(loc="best", fontsize="small")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            phi_pages += 1

    structure_pages = 0
    with PdfPages(structure_pdf) as pdf:
        for iq2 in range(binning.shape[0]):
            for ixb in range(binning.shape[1]):
                visible = data_sf_mask[iq2, ixb]
                if not np.any(visible):
                    continue
                fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)
                for sf_index, ax in enumerate(axes):
                    ax.errorbar(
                        t_centers[visible],
                        data_structure.values[iq2, ixb, visible, sf_index],
                        yerr=data_structure.uncertainties[
                            iq2, ixb, visible, sf_index
                        ],
                        fmt="o",
                        color="black",
                        capsize=2,
                        markersize=4,
                        label="data",
                    )
                    for model_index, record in enumerate(model_records):
                        model_mask = record["structure_valid"][iq2, ixb]
                        if not np.any(model_mask):
                            continue
                        ax.plot(
                            t_centers[model_mask],
                            record["structure"][iq2, ixb, model_mask, sf_index],
                            "s-",
                            color=colors[model_index % len(colors)],
                            linewidth=1.2,
                            markersize=3,
                            label=record["name"],
                        )
                    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
                    ax.set_ylabel(
                        f"{STRUCTURE_FUNCTION_NAMES[sf_index]} [nb/GeV^2]"
                    )
                    ax.grid(True, alpha=0.22)
                axes[0].legend(loc="best", fontsize="small")
                axes[-1].set_xlabel("-t bin center [GeV^2]")
                fig.suptitle(
                    "Structure functions from common harmonic convention\n"
                    f"Q2 {q2_edges[iq2]:g}-{q2_edges[iq2 + 1]:g}, "
                    f"xB {xb_edges[ixb]:g}-{xb_edges[ixb + 1]:g}"
                )
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
                pdf.savefig(fig)
                plt.close(fig)
                structure_pages += 1

    for grid_index in np.ndindex(binning.shape[:3]):
        iq2, ixb, it = grid_index
        for record in model_records:
            if not (data_fit_mask[grid_index] and record["fit_success"][grid_index]):
                continue
            chi2 = _harmonic_chi2(
                data_parameters[grid_index],
                record["parameters"][grid_index],
                data_covariance[grid_index],
            )
            row: list[object] = [
                record["name"],
                iq2,
                q2_edges[iq2],
                q2_edges[iq2 + 1],
                ixb,
                xb_edges[ixb],
                xb_edges[ixb + 1],
                it,
                t_edges[it],
                t_edges[it + 1],
                epsilon[grid_index],
                chi2,
            ]
            for coefficient in range(3):
                row.extend(
                    (
                        data_parameters[grid_index][coefficient],
                        record["parameters"][grid_index][coefficient],
                    )
                )
            for sf_index in range(3):
                data_value = data_structure.values[grid_index][sf_index]
                uncertainty = data_structure.uncertainties[grid_index][sf_index]
                model_value = record["structure"][grid_index][sf_index]
                pull = (
                    (data_value - model_value) / uncertainty
                    if np.isfinite(uncertainty) and uncertainty > 0.0
                    else np.nan
                )
                row.extend((data_value, uncertainty, model_value, pull))
            csv_rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows(csv_rows)
    return phi_pages, structure_pages, csv_path
