"""Writes the reduced-cost analysis for a solved scenario, registered on
``Event.after_postprocessing``.
"""

import logging

from . import _classify, _dual_dump, _heatmaps


def write_reduced_cost_analysis(optimization_setup, postprocess, config):
    """Builds and writes the reduced-cost analysis for one scenario.

    Writes ``capacity_addition_analysis.csv`` (the full reduced-cost table) and,
    if ``config["generate_classification"]`` is set, the classified
    ``capacity_addition_analysis_{conversion,transport,storage}.csv`` files next to
    it. Optionally renders heatmaps from the classified tables and writes a
    per-constraint dual decomposition for diagnostic targets.

    Args:
        optimization_setup: The solved OptimizationSetup to read the model, duals,
            and parameters from.
        postprocess: The just-created Postprocess instance for this scenario; its
            ``name_dir`` attribute is the output directory results were written to.
        config (dict): The plugin's configuration, see ``plugin.py``.
    """
    output_dir = postprocess.name_dir
    df = _classify.build_capacity_addition_analysis(optimization_setup)
    if df is None:
        return

    csv_path = output_dir.joinpath("capacity_addition_analysis.csv")
    df.to_csv(csv_path)
    logging.info(
        f"reduced_cost plugin: capacity addition analysis written to {csv_path}"
    )

    if not config.get("generate_classification", True):
        return
    try:
        classified = _classify.split_classified(
            optimization_setup, df, config.get("storage_power_perturbation", 0.0)
        )
    except Exception as error:
        logging.warning(
            f"reduced_cost plugin: could not classify the analysis: {error}"
        )
        return
    for tech_type, sub_df in classified.items():
        sub_path = output_dir.joinpath(f"capacity_addition_analysis_{tech_type}.csv")
        sub_df.to_csv(sub_path, index=False)
        logging.info(
            f"reduced_cost plugin: classified {tech_type} analysis written to "
            f"{sub_path} ({len(sub_df)} rows)"
        )

    if config.get("generate_heatmaps", False):
        try:
            _heatmaps.run(
                str(output_dir),
                vmax=config.get("heatmap_vmax", 150.0),
                storage_metric=config.get("storage_metric", "proportional"),
                annotate=config.get("heatmap_annotate", True),
            )
            heatmaps_dir = output_dir / "heatmaps"
            logging.info(
                f"reduced_cost plugin: heatmaps generated under {heatmaps_dir}"
            )
        except Exception as error:
            logging.warning(
                f"reduced_cost plugin: could not generate heatmaps: {error}"
            )

    if config.get("dual_dump", False):
        try:
            _dual_dump.dump_rc_decomposition(
                optimization_setup,
                config.get("dual_dump_targets", []),
                output_dir.joinpath("rc_dual_decomposition.csv"),
            )
        except Exception as error:
            logging.warning(
                f"reduced_cost plugin: could not dump the dual decomposition: {error}"
            )
