"""Reduced-cost analysis: how far an unbuilt technology is from being built.

For every unbuilt (or capacity-limited) technology, transport line, or storage
system, reconstructs the CAPEX-equivalent reduced cost of ``capacity_addition``:
the amount by which CAPEX would have to fall for the technology to become optimal
to build. The reconstruction relies on the model's LP duals, which are only
uniquely determined where the corresponding constraint is not degenerate; this
plugin also provides two optional constraint perturbations that resolve the most
common source of that degeneracy (an unbuilt technology whose lifetime-balance
constraint reads ``0 = 0``) without changing the optimal primal solution.

Three ZEN-garden events are used:

* ``after_construct_params``: optionally fixes the duration (energy-to-power
  ratio) of storage technologies, via :mod:`._param_overrides`.
* ``after_model_construction``: optionally perturbs the technology-lifetime and
  storage-power constraints to pin degenerate duals to their economically correct
  value, via :mod:`._perturbation`.
* ``after_postprocessing``: reconstructs and classifies the reduced cost of every
  capacity-addition decision, and optionally renders heatmaps and a diagnostic
  dual decomposition, via :mod:`._postprocess`.

See :mod:`.capex_override` for a standalone utility (not tied to any event) that
overrides a technology's CAPEX for a single (technology, node, year) in the
dataset's input files, and :mod:`.cli` for regenerating heatmaps from an existing
output folder.
"""

from typing import Any

from zen_garden.plugin_system.events import (  # type: ignore[import-untyped]
    Event,
    EventPublisher,
)

from . import _param_overrides, _perturbation, _postprocess

config: dict[str, Any] = {
    # after_construct_params: {storage_technology: fixed_duration_in_hours, ...}
    "tight_e2p": {},
    # after_model_construction: perturbation sizes in capacity units (e.g. GW);
    # 0 disables the corresponding perturbation
    "lifetime_rhs_perturbation": 0.0,
    "storage_power_perturbation": 0.0,
    # after_postprocessing
    "generate_classification": True,
    "generate_heatmaps": False,
    "heatmap_vmax": 150.0,
    "storage_metric": "proportional",
    "heatmap_annotate": True,
    "dual_dump": False,
    "dual_dump_targets": [],
}


@EventPublisher.register(Event.after_construct_params)
def apply_tight_energy_to_power_ratio(optimization_setup):
    """Fixes the duration of the storage technologies in ``config["tight_e2p"]``.

    Args:
        optimization_setup: The OptimizationSetup passed by the triggering event.
    """
    _param_overrides.apply_tight_energy_to_power_ratio(
        optimization_setup, config["tight_e2p"]
    )


@EventPublisher.register(Event.after_model_construction)
def perturb_degenerate_constraints(optimization_setup):
    """Applies the lifetime-RHS and storage-power reduced-cost perturbations."""
    _perturbation.perturb_lifetime_rhs(
        optimization_setup, config["lifetime_rhs_perturbation"]
    )
    _perturbation.perturb_storage_power_addition(
        optimization_setup, config["storage_power_perturbation"]
    )


@EventPublisher.register(Event.after_postprocessing)
def write_reduced_cost_analysis(optimization_setup, postprocess):
    """Reconstructs, classifies, and (optionally) visualizes the reduced cost."""
    _postprocess.write_reduced_cost_analysis(optimization_setup, postprocess, config)
