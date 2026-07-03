"""Constraint perturbations that pin degenerate reduced costs at unbuilt technologies.

Registered on ``Event.after_model_construction``, which fires once the full
optimization problem (sets, parameters, variables, constraints, objective) has been
built, but before scaling and solving. Both functions modify the right-hand side or
the bounds of already-built constraints; neither changes the optimal primal solution
in the limit of a vanishing perturbation.

Background: at an unbuilt (greenfield) technology, ``capacity`` and
``capacity_addition`` are both 0, so the technology-lifetime equality
``capacity - sum(capacity_addition) = capacity_existing`` reads ``0 = 0``. Its dual
is then not pinned to a unique value, and a reduced-cost reconstruction built on
that dual can read an arbitrary point of a whole feasible interval instead of the
economically correct value. Both perturbations below break this degeneracy by
adding a small positive value to one side of the constraint, which selects the
economically correct endpoint of the interval without changing the optimal primal
solution as the perturbation size goes to 0.
"""

import logging

import numpy as np
import xarray as xr


def perturb_lifetime_rhs(optimization_setup, delta):
    """Adds a phantom existing capacity to break lifetime-dual degeneracy.

    Adds ``delta`` to the right-hand side of ``constraint_technology_lifetime`` for
    every degenerate greenfield entry: an entry is perturbed only if its existing
    capacity is (numerically) 0 AND its capacity ceiling (the smaller of the
    ``capacity`` variable's upper bound and ``capacity_limit``) is at least
    ``delta``. Both conditions are required to preserve feasibility: perturbing a
    technology whose ceiling is 0 would push the added phantom capacity above its
    upper bound and make the model infeasible. Brownfield entries (existing capacity
    != 0) are skipped because their lifetime dual is not degenerate to begin with.

    Storage technologies are excluded here; their lifetime-RHS phantom would land in
    ``capacity`` while the energy-to-power-ratio constraint reads
    ``capacity_addition``, decoupling the two. Storage is instead handled by
    :func:`perturb_storage_power_addition`.

    Args:
        optimization_setup: The OptimizationSetup whose model constraints are
            modified.
        delta (float): The perturbation size, in capacity units (e.g. GW). A
            falsy value (0 or None) disables the perturbation.
    """
    if not delta:
        return
    name = "constraint_technology_lifetime"
    if name not in optimization_setup.model.constraints:
        logging.warning(
            f"reduced_cost plugin: '{name}' not found in the model — lifetime-RHS "
            f"perturbation skipped"
        )
        return
    constraint = optimization_setup.model.constraints[name]
    rhs = constraint.rhs
    try:
        capacity_upper = xr.align(
            rhs, optimization_setup.model.variables["capacity"].upper, join="left"
        )[1].fillna(np.inf)
        capacity_limit = xr.align(
            rhs, optimization_setup.parameters.capacity_limit, join="left"
        )[1].fillna(np.inf)
        ceiling = np.minimum(capacity_upper, capacity_limit)
        is_greenfield = np.abs(rhs) < 1e-9
        storage_techs = list(optimization_setup.sets["set_storage_technologies"])
        is_storage = rhs["set_technologies"].isin(storage_techs)
        addition = xr.where(
            is_greenfield & (ceiling >= delta) & ~is_storage, float(delta), 0.0
        )
        addition = addition.broadcast_like(rhs).transpose(*rhs.dims)
        addition_data = addition.data
    except Exception as error:
        logging.error(
            f"reduced_cost plugin: lifetime-RHS perturbation guard failed "
            f"({error}); skipping the perturbation entirely to avoid infeasibility"
        )
        return
    n_perturbed = int(np.count_nonzero(addition_data))
    n_total = int(addition_data.size)
    constraint.rhs.data = rhs.data + addition_data
    logging.info(
        f"reduced_cost plugin: added +{delta} to the RHS of {name} "
        f"({n_perturbed}/{n_total} entries perturbed; "
        f"{n_total - n_perturbed} skipped for insufficient head-room)"
    )


def perturb_storage_power_addition(optimization_setup, magnitude):
    """Forces a small storage-power addition to break storage bundle degeneracy.

    Storage is built as two separate capacities (power and energy) that are only
    coupled through operation. At an unbuilt greenfield node both are 0, so the
    marginal value of either component alone is 0 and the reduced cost of each
    component collapses to its own capex instead of reflecting the joint
    (power + energy) cost of a viable storage bundle.

    Raises the lower bound of ``capacity_addition`` for the power component to
    ``magnitude`` at every genuinely unbuilt greenfield storage node (existing
    capacity 0, positive finite ``energy_to_power_ratio_min``, and enough head-room
    for both the forced power and the resulting forced energy). The
    energy-to-power-ratio constraint then forces a matching energy addition, so the
    forced power capacity stays non-basic and its reduced cost reflects the joint
    bundle distance to becoming a viable storage system.

    Args:
        optimization_setup: The OptimizationSetup whose model variables are
            modified.
        magnitude (float): The forced lower bound on the power capacity addition,
            in capacity units (e.g. GW). A falsy value (0 or None) disables the
            perturbation.
    """
    if not magnitude:
        return
    storage_techs = list(optimization_setup.sets["set_storage_technologies"])
    if not storage_techs:
        return
    variable = optimization_setup.model.variables["capacity_addition"]
    try:
        lower = variable.lower
        capacity_upper = xr.align(
            lower, optimization_setup.model.variables["capacity"].upper, join="left"
        )[1].fillna(np.inf)
        capacity_limit = xr.align(
            lower, optimization_setup.parameters.capacity_limit, join="left"
        )[1].fillna(np.inf)
        ceiling = np.minimum(capacity_upper, capacity_limit)
        existing = xr.align(
            lower,
            optimization_setup.model.constraints["constraint_technology_lifetime"].rhs,
            join="left",
        )[1].fillna(0.0)
        e2p_min = optimization_setup.parameters.energy_to_power_ratio_min.rename(
            {"set_storage_technologies": "set_technologies"}
        )
        e2p_min = e2p_min.reindex(
            set_technologies=lower.indexes["set_technologies"], fill_value=0.0
        )
        ceiling_energy = ceiling.sel(set_capacity_types="energy")

        is_power = lower["set_capacity_types"] == "power"
        is_storage = lower["set_technologies"].isin(storage_techs)
        is_greenfield = existing < 1e-9
        has_finite_duration = (e2p_min > 0) & np.isfinite(e2p_min)
        power_has_room = ceiling >= magnitude
        energy_has_room = ceiling_energy >= (e2p_min * magnitude)

        apply = (
            is_power
            & is_storage
            & is_greenfield
            & has_finite_duration
            & power_has_room
            & energy_has_room
        )
        apply = apply.broadcast_like(lower).transpose(*lower.dims)
        new_lower = xr.where(apply, float(magnitude), lower)
    except Exception as error:
        logging.error(
            f"reduced_cost plugin: storage-power perturbation guard failed "
            f"({error}); skipping the perturbation entirely to avoid infeasibility"
        )
        return
    n_perturbed = int(apply.sum())
    variable.lower = new_lower
    logging.info(
        f"reduced_cost plugin: raised the storage-power capacity-addition lower "
        f"bound to +{magnitude} at {n_perturbed} greenfield storage nodes"
    )
