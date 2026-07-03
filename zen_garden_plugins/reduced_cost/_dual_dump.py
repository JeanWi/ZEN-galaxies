"""Diagnostic per-constraint decomposition of a capacity-addition reduced cost.

For each configured target, computes the true reduced cost directly from the
solved model as

    rc_true = objective_coefficient(variable) - sum_over_constraints(coefficient * dual)

and reports the contribution of every constraint that the variable appears in,
flagging which of them the reduced-cost reconstruction in
:mod:`zen_garden_plugins.reduced_cost._classify` accounts for (lifetime, diffusion,
and the storage energy-to-power-ratio constraints). This reveals whether an
unaccounted-for constraint carries a contribution the reconstruction misses.
"""

import logging

import numpy as np
import pandas as pd

# constraints the reduced-cost reconstruction accounts for
_ACCOUNTED_FOR = {
    "constraint_technology_lifetime",
    "constraint_technology_diffusion_limit",
    "constraint_technology_diffusion_limit_total",
    "constraint_capacity_energy_to_power_ratio_min",
    "constraint_capacity_energy_to_power_ratio_max",
}


def dump_rc_decomposition(optimization_setup, targets, output_path):
    """Writes a per-constraint reduced-cost decomposition for ``targets`` to CSV.

    Args:
        optimization_setup: The solved OptimizationSetup to read the model and
            duals from.
        targets (list[dict]): Each entry is
            ``{"tech", "node"[, "capacity_type"]}``.
        output_path (pathlib.Path): File to write the decomposition to. Not
            written if ``targets`` is empty or none of the targets are found.
    """
    if not targets:
        return
    model = optimization_setup.model
    labels = model.variables["capacity_addition"].labels
    try:
        objective_expression = model.objective.expression
        objective_vars = np.asarray(objective_expression.vars.values)
        objective_coeffs = np.asarray(objective_expression.coeffs.values)
    except Exception:
        objective_vars = objective_coeffs = None

    rows = []
    for target in targets:
        tech, node = target["tech"], target["node"]
        capacity_type = target.get("capacity_type", "power")
        try:
            selected_labels = labels.sel(
                set_technologies=tech,
                set_capacity_types=capacity_type,
                set_location=node,
            )
        except Exception as error:
            logging.warning(
                f"reduced_cost plugin: dual-dump target {target} not found ({error})"
            )
            continue
        for year in [
            int(v)
            for v in np.atleast_1d(selected_labels["set_time_steps_yearly"].values)
        ]:
            label = int(selected_labels.sel(set_time_steps_yearly=year).item())
            if label < 0:
                continue
            contributions = {}
            for constraint_name, constraint in model.constraints.items():
                dual = getattr(constraint, "dual", None)
                variables = getattr(constraint, "vars", None)
                if dual is None or variables is None or "_term" not in variables.dims:
                    continue
                if not (np.asarray(variables.values) == label).any():
                    continue
                coefficient_row = constraint.coeffs.where(variables == label, 0.0).sum(
                    "_term"
                )
                contribution = float((coefficient_row * dual.fillna(0.0)).sum().item())
                if abs(contribution) > 1e-12:
                    contributions[constraint_name] = contribution
            objective_coefficient = (
                float(objective_coeffs[objective_vars == label].sum())
                if (objective_vars is not None and (objective_vars == label).any())
                else 0.0
            )
            rc_true = objective_coefficient - sum(contributions.values())
            for constraint_name, contribution in sorted(
                contributions.items(), key=lambda kv: -abs(kv[1])
            ):
                rows.append(
                    dict(
                        tech=tech,
                        node=node,
                        capacity_type=capacity_type,
                        year_index=year,
                        constraint=constraint_name,
                        contribution_objective_units=contribution,
                        accounted_for=(constraint_name in _ACCOUNTED_FOR),
                    )
                )
            rows.append(
                dict(
                    tech=tech,
                    node=node,
                    capacity_type=capacity_type,
                    year_index=year,
                    constraint="== rc_true (objective units) ==",
                    contribution_objective_units=rc_true,
                    accounted_for="",
                )
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logging.info(
            f"reduced_cost plugin: dual decomposition written to {output_path}"
        )
