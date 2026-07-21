"""Reduced-cost reconstruction and classification for capacity-addition decisions.

Reconstructs, for every (technology, capacity_type, node, year) combination, the
CAPEX-equivalent reduced cost of ``capacity_addition``: the amount by which CAPEX
would have to fall for the technology to become optimal to build. Two
reconstructions are computed:

* ``rc_capex_equivalent``: derived from the dual of ``constraint_technology_lifetime``.
  Exact when the lifetime dual is uniquely determined, but can pick up the reduced
  cost of the ``capacity`` variable itself when that variable is non-basic (i.e. the
  technology is unbuilt but still operated), which biases the reconstruction.
* ``rc_capex_equivalent_operational``: derived from the dual of
  ``constraint_capacity_factor_conversion`` / ``..._transport`` (the per-timestep
  dispatch value of one unit of capacity) instead of the lifetime dual. This
  excludes the reduced-cost leak of the ``capacity`` variable and is therefore
  reliable at unbuilt-but-operated technologies where the lifetime-dual
  reconstruction is not. Only defined for conversion and transport technologies
  (storage needs the arbitrage value of a whole storage bundle, which this
  reconstruction does not provide); NaN otherwise.

Both reconstructions require the model to have been solved with a method and
presolve setting that returns proper LP duals for a degenerate basis (e.g. Gurobi
Method=2/Barrier with Crossover=1, Presolve=0).
"""

import logging

import numpy as np
import pandas as pd

# tolerances used by classify()
CEILING_ZERO = 1e-6  # ceiling ~ 0 -> nothing can be built
CEILING_RELATIVE = 1e-4  # |capacity - ceiling| (relative) -> sitting at the limit
RC_ZERO = 1e-6  # |reduced cost| ~ 0
VALUE_TOLERANCE = 1e-6  # value > tolerance -> built (non-storage)


def _pick_dimension(dims, *needles):
    """Returns the first dimension name in ``dims`` containing any of ``needles``."""
    return next((d for d in dims for needle in needles if needle in d), None)


def capex_specific_lookup(optimization_setup):
    """Returns lookups from (technology[, capacity_type], node, year) to CAPEX.

    Source: the ``capex_specific_{conversion,storage,transport}`` parameters, i.e.
    the specific CAPEX the model actually uses for the planning year (the input CSV
    value scaled by the model's fraction-of-year factor). This is exactly the term
    the reduced cost is measured against.

    Args:
        optimization_setup: The OptimizationSetup to read parameters from.

    Returns:
        dict: Mapping of both ``(tech, node, year)`` and
            ``(tech, capacity_type, node, year)`` to the CAPEX value in model units.
    """
    parameters = optimization_setup.parameters
    lookup = {}
    for parameter_name in (
        "capex_specific_conversion",
        "capex_specific_storage",
        "capex_specific_transport",
    ):
        parameter = getattr(parameters, parameter_name, None)
        if parameter is None:
            continue
        try:
            for index, value in parameter.to_series().dropna().items():
                if not isinstance(index, tuple):
                    index = (index,)
                tech, year, node = index[0], index[-1], index[-2]
                capacity_type = index[1] if len(index) == 4 else None
                lookup.setdefault((tech, node, year), float(value))
                lookup.setdefault((tech, capacity_type, node, year), float(value))
        except Exception:
            continue
    return lookup


def _dual_series(optimization_setup, constraint_name):
    """Returns the (scaling-corrected) dual of a constraint as a pandas Series."""
    if constraint_name not in optimization_setup.model.constraints:
        return None
    constraint = optimization_setup.model.constraints[constraint_name]
    dual = constraint.dual
    if optimization_setup.solver.use_scaling:
        labels = constraint.labels.data
        dual = dual * optimization_setup.scaling.D_r_inv[labels]
    return dual.to_series().dropna()


def compute_rc_capex_equivalent(optimization_setup, df):
    """Computes the lifetime-dual and operational reduced-cost reconstructions.

    Uses the LP optimality condition for ``capacity_addition`` at its lower bound:

        rc = capex_specific - (capacity_value - diffusion_contribution) / scaling

    where ``capacity_value`` is either the (negated) dual of
    ``constraint_technology_lifetime`` (lifetime reconstruction) or the dispatch
    value derived from the capacity-factor dual net of fixed OPEX (operational
    reconstruction), ``diffusion_contribution`` captures that
    ``capacity_addition[year]`` also relaxes future technology-diffusion limits, and
    ``scaling`` is the annuity factor times the sum of discount factors over the
    technology's payback years.

    Args:
        optimization_setup: The solved OptimizationSetup to read duals and
            parameters from.
        df (pandas.DataFrame): Indexed by
            ``(technology, capacity_type, node, year)``, one row per
            ``capacity_addition`` entry.

    Returns:
        tuple[list[float], list[float]]: ``(lifetime_rc, operational_rc)``, one
            value per row of ``df``, in model (internal) units.
    """
    parameters = optimization_setup.parameters
    system = optimization_setup.system
    energy_system = optimization_setup.energy_system

    discount_rate = float(parameters.discount_rate)
    interval = system.interval_between_years
    years = list(energy_system.set_time_steps_yearly)
    first_year = years[0]
    last_year = energy_system.set_time_steps_yearly_entire_horizon[-1]

    discount_factors = {}
    for year in years:
        n_years_represented = 1 if year == last_year else interval
        discount_factors[year] = sum(
            (1.0 / (1.0 + discount_rate)) ** (interval * (year - first_year) + i)
            for i in range(n_years_represented)
        )

    lifetime_dual = _dual_series(optimization_setup, "constraint_technology_lifetime")
    diffusion_total_dual = _dual_series(
        optimization_setup, "constraint_technology_diffusion_limit_total"
    )
    diffusion_dual = _dual_series(
        optimization_setup, "constraint_technology_diffusion_limit"
    )
    e2p_min_dual = _dual_series(
        optimization_setup, "constraint_capacity_energy_to_power_ratio_min"
    )
    e2p_max_dual = _dual_series(
        optimization_setup, "constraint_capacity_energy_to_power_ratio_max"
    )
    e2p_min_param = getattr(parameters, "energy_to_power_ratio_min", None)
    e2p_max_param = getattr(parameters, "energy_to_power_ratio_max", None)

    capex_lookup = capex_specific_lookup(optimization_setup)

    # operational valuation inputs (conversion and transport only): the dual of the
    # capacity-factor constraint gives the per-timestep marginal operating value of
    # one unit of capacity; summed over a year and netted of fixed OPEX it gives the
    # capacity's net operating value without the capacity-variable reduced-cost leak.
    operational_value = {}
    operational_techs = set()
    fixed_opex_lookup = {}

    def _pick(columns, kind):
        for name in columns:
            if not isinstance(name, str):
                continue
            lower = name.lower()
            if kind == "time" and "time" in lower and "oper" in lower:
                return name
            if kind == "tech" and "technolog" in lower:
                return name
            if kind == "node" and (
                "node" in lower or "location" in lower or "edge" in lower
            ):
                return name
        return None

    try:
        max_load = parameters.max_load.to_series().dropna().rename("m").reset_index()
        max_load_time = _pick(max_load.columns, "time")
        max_load_tech = _pick(max_load.columns, "tech")
        max_load_node = _pick(max_load.columns, "node")
        for constraint_name in (
            "constraint_capacity_factor_conversion",
            "constraint_capacity_factor_transport",
        ):
            dual = _dual_series(optimization_setup, constraint_name)
            if dual is None:
                continue
            capacity_factor = dual.rename("dual").reset_index()
            time_col = _pick(capacity_factor.columns, "time")
            tech_col = _pick(capacity_factor.columns, "tech")
            node_col = _pick(capacity_factor.columns, "node")
            if time_col is None or tech_col is None or node_col is None:
                continue
            capacity_factor["__year"] = capacity_factor[time_col].map(
                lambda t: int(
                    energy_system.time_steps.convert_time_step_operation2year(t)
                )
            )
            if max_load_time and max_load_tech and max_load_node:
                merged = capacity_factor.merge(
                    max_load,
                    left_on=[tech_col, node_col, time_col],
                    right_on=[max_load_tech, max_load_node, max_load_time],
                    how="left",
                )
                merged["m"] = merged["m"].fillna(1.0)
            else:
                merged = capacity_factor.assign(m=1.0)
            merged["__weighted_dual"] = merged["m"] * merged["dual"]
            grouped = merged.groupby([tech_col, node_col, "__year"])[
                "__weighted_dual"
            ].sum()
            for (tech, node, year), value in grouped.items():
                operational_value[(str(tech), str(node), int(year))] = float(value)
                operational_techs.add(str(tech))
        fixed_opex = getattr(parameters, "opex_specific_fixed", None)
        if fixed_opex is not None:
            for index, value in fixed_opex.to_series().dropna().items():
                if not isinstance(index, tuple):
                    index = (index,)
                if len(index) == 4:  # (tech, capacity_type, node, year)
                    fixed_opex_lookup[(index[0], index[1], index[2], index[3])] = float(
                        value
                    )
                elif len(index) == 3:  # (tech, node, year)
                    fixed_opex_lookup[(index[0], None, index[1], index[2])] = float(
                        value
                    )
    except Exception as error:
        logging.debug(f"reduced_cost plugin: operational RC precompute failed: {error}")
        operational_value, operational_techs, fixed_opex_lookup = {}, set(), {}

    tech_cache = {}
    diffusion_param_cache = {}
    lifetime_rc = []
    operational_rc = []

    for index in df.index:
        tech, capacity_type, node, invest_year = index[0], index[1], index[2], index[3]
        try:
            if tech not in tech_cache:
                lifetime_years = float(
                    np.squeeze(
                        parameters.depreciation_time.sel(set_technologies=tech).values
                    )
                )
                annuity_factor = (
                    ((1.0 + discount_rate) ** lifetime_years * discount_rate)
                    / ((1.0 + discount_rate) ** lifetime_years - 1.0)
                    if discount_rate != 0
                    else 1.0 / lifetime_years
                )
                tech_cache[tech] = (
                    annuity_factor,
                    max(int(np.floor(lifetime_years / interval)), 1),
                )
            annuity_factor, n_payback_periods = tech_cache[tech]

            payback_years = [
                y
                for y in years
                if invest_year <= y <= invest_year + n_payback_periods - 1
            ]
            discount_sum = sum(discount_factors[y] for y in payback_years)
            scaling = annuity_factor * discount_sum
            if scaling <= 0:
                lifetime_rc.append(np.nan)
                operational_rc.append(np.nan)
                continue

            capex = capex_lookup.get(
                (tech, capacity_type, node, invest_year),
                capex_lookup.get((tech, node, invest_year), np.nan),
            )
            if np.isnan(capex):
                lifetime_rc.append(np.nan)
                operational_rc.append(np.nan)
                continue

            capacity_value = 0.0
            for year in payback_years:
                if lifetime_dual is None:
                    continue
                mask = (
                    (lifetime_dual.index.get_level_values(0) == tech)
                    & (lifetime_dual.index.get_level_values(1) == capacity_type)
                    & (lifetime_dual.index.get_level_values(2) == node)
                    & (lifetime_dual.index.get_level_values(3) == year)
                )
                matches = lifetime_dual[mask]
                if not matches.empty:
                    capacity_value += -float(matches.iloc[0])
                # else: the lifetime dual is degenerate here (a diffusion constraint
                # is the active bound instead); its value is captured below.

            # diffusion contribution: capacity_addition[invest_year] also relaxes
            # future diffusion limits (year > invest_year); its dual contribution is
            # dual * technology_diffusion_rate * knowledge_depreciation.
            diffusion_contribution = 0.0
            future_years = [y for y in years if y > invest_year]
            if future_years and (
                diffusion_total_dual is not None or diffusion_dual is not None
            ):
                if tech not in diffusion_param_cache:
                    try:
                        knowledge_depreciation_rate = float(
                            np.squeeze(
                                parameters.knowledge_depreciation_rate.sel(
                                    set_technologies=tech
                                ).values
                            )
                        )
                    except Exception:
                        knowledge_depreciation_rate = 0.0
                    try:
                        spillover_rate = float(
                            np.squeeze(
                                parameters.knowledge_spillover_rate.sel(
                                    set_technologies=tech
                                ).values
                            )
                        )
                    except Exception:
                        spillover_rate = np.inf
                    diffusion_param_cache[tech] = (
                        knowledge_depreciation_rate,
                        spillover_rate,
                    )
                knowledge_depreciation_rate, spillover_rate = diffusion_param_cache[
                    tech
                ]

                for future_year in future_years:
                    knowledge_decay = (1.0 - knowledge_depreciation_rate) ** (
                        interval * (future_year - 1 - invest_year)
                    )
                    try:
                        max_diffusion_rate = float(
                            np.squeeze(
                                parameters.max_diffusion_rate.sel(
                                    set_technologies=tech,
                                    set_time_steps_yearly=future_year,
                                ).values
                            )
                        )
                    except Exception:
                        continue
                    technology_diffusion_rate = (
                        1.0 + max_diffusion_rate
                    ) ** interval - 1.0
                    if technology_diffusion_rate <= 0:
                        continue
                    coefficient = technology_diffusion_rate * knowledge_decay

                    if diffusion_total_dual is not None:
                        mask = (
                            (diffusion_total_dual.index.get_level_values(0) == tech)
                            & (
                                diffusion_total_dual.index.get_level_values(1)
                                == capacity_type
                            )
                            & (
                                diffusion_total_dual.index.get_level_values(2)
                                == future_year
                            )
                        )
                        if mask.any():
                            diffusion_contribution += (
                                float(diffusion_total_dual[mask].iloc[0]) * coefficient
                            )

                    if diffusion_dual is not None and not np.isinf(spillover_rate):
                        mask_same_node = (
                            (diffusion_dual.index.get_level_values(0) == tech)
                            & (
                                diffusion_dual.index.get_level_values(1)
                                == capacity_type
                            )
                            & (diffusion_dual.index.get_level_values(2) == node)
                            & (diffusion_dual.index.get_level_values(3) == future_year)
                        )
                        if mask_same_node.any():
                            diffusion_contribution += (
                                float(diffusion_dual[mask_same_node].iloc[0])
                                * coefficient
                            )
                        for other_node in diffusion_dual.index.get_level_values(
                            2
                        ).unique():
                            if other_node == node:
                                continue
                            mask_other_node = (
                                (diffusion_dual.index.get_level_values(0) == tech)
                                & (
                                    diffusion_dual.index.get_level_values(1)
                                    == capacity_type
                                )
                                & (
                                    diffusion_dual.index.get_level_values(2)
                                    == other_node
                                )
                                & (
                                    diffusion_dual.index.get_level_values(3)
                                    == future_year
                                )
                            )
                            if mask_other_node.any():
                                diffusion_contribution += (
                                    float(diffusion_dual[mask_other_node].iloc[0])
                                    * technology_diffusion_rate
                                    * spillover_rate
                                    * knowledge_decay
                                )

            # energy-to-power-ratio dual contribution: only the power component of
            # storage technologies has coefficient -e2p in this constraint, so its
            # contribution is added like the diffusion contribution above.
            e2p_contribution = 0.0
            for dual, param in (
                (e2p_min_dual, e2p_min_param),
                (e2p_max_dual, e2p_max_param),
            ):
                if dual is None or capacity_type != "power" or param is None:
                    continue
                try:
                    ratio = float(
                        np.squeeze(param.sel(set_storage_technologies=tech).values)
                    )
                except Exception:
                    ratio = 0.0
                if np.isfinite(ratio) and ratio > 0:
                    mask = (
                        (dual.index.get_level_values(0) == tech)
                        & (dual.index.get_level_values(-2) == node)
                        & (dual.index.get_level_values(-1) == invest_year)
                    )
                    if mask.any():
                        e2p_contribution += float(dual[mask].iloc[0]) * ratio

            lifetime_rc.append(
                np.nan
                if lifetime_dual is None
                else capex
                - (capacity_value - diffusion_contribution - e2p_contribution) / scaling
            )

            if tech in operational_techs:
                operational_capacity_value = 0.0
                for year in payback_years:
                    operational_capacity_value += operational_value.get(
                        (tech, node, year), 0.0
                    )
                    fixed_opex = fixed_opex_lookup.get(
                        (tech, capacity_type, node, year),
                        fixed_opex_lookup.get((tech, None, node, year), 0.0),
                    )
                    operational_capacity_value -= fixed_opex * discount_factors[year]
                operational_rc.append(
                    capex
                    - (
                        operational_capacity_value
                        - diffusion_contribution
                        - e2p_contribution
                    )
                    / scaling
                )
            else:
                operational_rc.append(np.nan)

        except Exception:
            lifetime_rc.append(np.nan)
            operational_rc.append(np.nan)

    return lifetime_rc, operational_rc


def build_capacity_addition_analysis(optimization_setup):
    """Builds the base reduced-cost analysis table for ``capacity_addition``.

    Args:
        optimization_setup: The solved OptimizationSetup to read the solution,
            duals, and parameters from.

    Returns:
        pandas.DataFrame | None: Indexed by
            ``(technology, capacity_type, node, year)``, with columns ``value``
            (optimal capacity addition), ``capacity`` (total installed capacity),
            ``capacity_ceiling`` (binding upper limit),
            ``rc_capex_equivalent[_input_units]`` and
            ``rc_capex_equivalent_operational[_input_units]``. ``None`` if the
            model has no ``capacity_addition`` variable or solution.
    """
    if "capacity_addition" not in optimization_setup.model.variables:
        logging.info(
            "reduced_cost plugin: capacity_addition variable not found — skipping"
        )
        return None

    values = optimization_setup.model.solution["capacity_addition"]
    df = values.to_series().dropna().to_frame("value")
    if df.empty:
        logging.warning("reduced_cost plugin: no capacity_addition data to analyze")
        return None

    try:
        df["capacity"] = (
            optimization_setup.model.solution["capacity"].to_series().reindex(df.index)
        )
    except Exception as error:
        logging.debug(f"reduced_cost plugin: could not retrieve capacity: {error}")
        df["capacity"] = np.nan
    try:
        capacity_upper = (
            optimization_setup.model.variables["capacity"]
            .upper.to_series()
            .reindex(df.index)
        )
        capacity_limit = (
            optimization_setup.parameters.capacity_limit.to_series().reindex(df.index)
        )
        df["capacity_ceiling"] = np.minimum(
            capacity_upper.fillna(np.inf), capacity_limit.fillna(np.inf)
        )
    except Exception as error:
        logging.debug(
            f"reduced_cost plugin: could not retrieve capacity ceiling: {error}"
        )
        df["capacity_ceiling"] = np.nan

    try:
        lifetime_rc, operational_rc = compute_rc_capex_equivalent(
            optimization_setup, df
        )
        df["rc_capex_equivalent"] = lifetime_rc
        df["rc_capex_equivalent_operational"] = operational_rc
    except Exception as error:
        logging.warning(
            f"reduced_cost plugin: could not compute rc_capex_equivalent: {error}"
        )
        df["rc_capex_equivalent"] = np.nan
        df["rc_capex_equivalent_operational"] = np.nan

    # Convert to input units (e.g. Euro/kW): ZEN-garden stores capex_specific
    # internally as input_value * fraction_year, so dividing by fraction_year
    # recovers the original input unit.
    try:
        fraction_year = (
            optimization_setup.system.unaggregated_time_steps_per_year
            / optimization_setup.system.total_hours_per_year
        )
        df["rc_capex_equivalent_input_units"] = (
            df["rc_capex_equivalent"] / fraction_year
        )
        df["rc_capex_equivalent_operational_input_units"] = (
            df["rc_capex_equivalent_operational"] / fraction_year
        )
    except Exception as error:
        logging.warning(
            f"reduced_cost plugin: could not compute input-unit reduced costs: {error}"
        )
        df["rc_capex_equivalent_input_units"] = np.nan
        df["rc_capex_equivalent_operational_input_units"] = np.nan

    return df[
        [
            "value",
            "capacity",
            "capacity_ceiling",
            "rc_capex_equivalent",
            "rc_capex_equivalent_input_units",
            "rc_capex_equivalent_operational",
            "rc_capex_equivalent_operational_input_units",
        ]
    ]


def classify(value, capacity, ceiling, reduced_cost, built_tolerance):
    """Classifies a single capacity-addition decision into a reduced-cost case.

    Args:
        value (float): Optimal capacity addition.
        capacity (float): Total installed capacity.
        ceiling (float): Binding upper limit on capacity (inf if unbounded).
        reduced_cost (float): The reduced cost driving the case decision.
        built_tolerance (float): Threshold above which ``value`` counts as built.

    Returns:
        str: One of ``"built"``, ``"not_buildable"``, ``"at_limit"``,
            ``"buildable_rc"``, ``"blocked_profitable"``, or
            ``"breakeven_unreliable"``.
    """
    value = value if np.isfinite(value) else 0.0
    capacity = capacity if np.isfinite(capacity) else 0.0
    reduced_cost = reduced_cost if np.isfinite(reduced_cost) else 0.0
    is_unbounded = not np.isfinite(ceiling)
    if value > built_tolerance:
        return "built"
    if (not is_unbounded) and ceiling <= CEILING_ZERO:
        return "not_buildable"
    if (
        (not is_unbounded)
        and capacity > CEILING_ZERO
        and abs(capacity - ceiling) <= CEILING_RELATIVE * max(1.0, abs(ceiling))
    ):
        return "at_limit"
    if reduced_cost > RC_ZERO:
        return "buildable_rc"
    if reduced_cost < -RC_ZERO:
        return "blocked_profitable"
    return "breakeven_unreliable"


def split_classified(optimization_setup, df, storage_power_perturbation=0.0):
    """Splits and classifies the base analysis by technology class.

    Args:
        optimization_setup: The solved OptimizationSetup to read sets and
            parameters from.
        df (pandas.DataFrame): The table returned by
            :func:`build_capacity_addition_analysis`.
        storage_power_perturbation (float): The magnitude used by
            :func:`_perturbation.perturb_storage_power_addition`, if any. Used to
            set the built-detection tolerance for storage, so the forced probe
            capacity addition is not mistaken for a real build.

    Returns:
        dict[str, pandas.DataFrame]: Mapping of ``"conversion"``, ``"transport"``,
            and ``"storage"`` to their classified analysis tables. A technology
            class is omitted if it has no rows.
    """
    sets = optimization_setup.sets
    storage_built_tolerance = max(
        5.0 * (storage_power_perturbation or 0.0), VALUE_TOLERANCE
    )

    type_of_tech = {}
    for set_name, tech_type in (
        ("set_conversion_technologies", "conversion"),
        ("set_transport_technologies", "transport"),
        ("set_storage_technologies", "storage"),
        ("set_retrofitting_technologies", "conversion"),
    ):
        try:
            for tech in sets[set_name]:
                type_of_tech[str(tech)] = tech_type
        except Exception:
            continue

    e2p_of_tech = {}
    e2p_param = getattr(
        optimization_setup.parameters, "energy_to_power_ratio_min", None
    )
    try:
        storage_techs = list(sets["set_storage_technologies"])
    except Exception:
        storage_techs = []
    for tech in storage_techs:
        ratio = np.nan
        if e2p_param is not None:
            try:
                ratio = float(
                    np.squeeze(e2p_param.sel(set_storage_technologies=tech).values)
                )
            except Exception:
                ratio = np.nan
        e2p_of_tech[str(tech)] = ratio if (np.isfinite(ratio) and ratio > 0) else np.nan

    fraction_year = 1.0
    try:
        fraction_year = (
            optimization_setup.system.unaggregated_time_steps_per_year
            / optimization_setup.system.total_hours_per_year
        )
    except Exception:
        pass

    capex_lookup = capex_specific_lookup(optimization_setup)

    def capex_input(tech, capacity_type, node, year):
        capex = capex_lookup.get(
            (tech, capacity_type, node, year),
            capex_lookup.get((tech, node, year), np.nan),
        )
        return (capex / fraction_year) if np.isfinite(capex) else np.nan

    work = df.reset_index()
    tech_col, capacity_type_col, node_col, year_col = list(work.columns)[:4]

    work["tech_type"] = work[tech_col].map(
        lambda t: type_of_tech.get(str(t), "unknown")
    )
    work["capex_specific_input_units"] = [
        capex_input(row[tech_col], row[capacity_type_col], row[node_col], row[year_col])
        for _, row in work.iterrows()
    ]

    result = {}

    # non-storage: conversion and transport (single "power" component)
    non_storage = work[work["tech_type"].isin(["conversion", "transport"])].copy()
    if not non_storage.empty:

        def rc_for_case(row):
            if row["tech_type"] == "conversion":
                operational = row["rc_capex_equivalent_operational_input_units"]
                if np.isfinite(operational):
                    return operational
            return row["rc_capex_equivalent_input_units"]

        non_storage["case"] = [
            classify(
                row["value"],
                row["capacity"],
                row["capacity_ceiling"],
                rc_for_case(row),
                VALUE_TOLERANCE,
            )
            for _, row in non_storage.iterrows()
        ]
        non_storage["rc_reliable"] = non_storage["case"].isin(["built", "buildable_rc"])

        def ratio(row, rc_column):
            capex = row["capex_specific_input_units"]
            rc = row[rc_column]
            if (
                row["case"] in ("built", "buildable_rc")
                and np.isfinite(capex)
                and capex > CEILING_ZERO
                and np.isfinite(rc)
            ):
                return rc / capex
            return np.nan

        non_storage["ratio_reduction"] = [
            ratio(row, "rc_capex_equivalent_input_units")
            for _, row in non_storage.iterrows()
        ]
        non_storage["ratio_reduction_operational"] = [
            ratio(row, "rc_capex_equivalent_operational_input_units")
            for _, row in non_storage.iterrows()
        ]

        columns = [
            tech_col,
            capacity_type_col,
            node_col,
            year_col,
            "tech_type",
            "value",
            "capacity",
            "capacity_ceiling",
            "case",
            "capex_specific_input_units",
            "rc_capex_equivalent",
            "rc_capex_equivalent_input_units",
            "ratio_reduction",
            "rc_capex_equivalent_operational",
            "rc_capex_equivalent_operational_input_units",
            "ratio_reduction_operational",
            "rc_reliable",
        ]
        for tech_type in ("conversion", "transport"):
            subset = non_storage[non_storage["tech_type"] == tech_type]
            if not subset.empty:
                result[tech_type] = subset[columns].reset_index(drop=True)

    # storage: bundle structure, one row per (technology, node, year), carried by
    # the power component
    storage = work[work["tech_type"] == "storage"].copy()
    if not storage.empty:
        power = storage[storage[capacity_type_col] == "power"].set_index(
            [tech_col, node_col, year_col]
        )
        energy = storage[storage[capacity_type_col] == "energy"].set_index(
            [tech_col, node_col, year_col]
        )
        rows = []
        for key in power.index.union(energy.index):
            tech = key[0]
            power_row = power.loc[key] if key in power.index else None
            energy_row = energy.loc[key] if key in energy.index else None
            duration = e2p_of_tech.get(str(tech), np.nan)

            value = float(power_row["value"]) if power_row is not None else np.nan
            capacity = float(power_row["capacity"]) if power_row is not None else np.nan
            ceiling = (
                float(power_row["capacity_ceiling"])
                if power_row is not None
                else np.nan
            )
            rc_power = (
                float(power_row["rc_capex_equivalent_input_units"])
                if power_row is not None
                else np.nan
            )
            capex_power = (
                float(power_row["capex_specific_input_units"])
                if power_row is not None
                else np.nan
            )
            capex_energy = (
                float(energy_row["capex_specific_input_units"])
                if energy_row is not None
                else np.nan
            )

            case = (
                classify(value, capacity, ceiling, rc_power, storage_built_tolerance)
                if power_row is not None
                else "no_power_row"
            )

            # multi-year carry-over: a storage built in a prior year persists via
            # its lifetime; the forced probe on an already-built storage then
            # produces a meaningless reduced cost.
            carried_over = (
                power_row is not None
                and np.isfinite(capacity)
                and np.isfinite(value)
                and (capacity - value) > 1e-2
            )
            if carried_over and case == "buildable_rc":
                case = "built"
                rc_power = np.nan

            bundle_capex = (
                (capex_power + duration * capex_energy)
                if (
                    np.isfinite(duration)
                    and np.isfinite(capex_power)
                    and np.isfinite(capex_energy)
                )
                else np.nan
            )
            # closing the bundle gap with the same absolute reduction on both
            # components: delta_power + duration * delta_energy = rc_power,
            # delta_power = delta_energy = delta -> delta = rc_power / (1 + duration)
            delta = (
                (rc_power / (1.0 + duration))
                if (np.isfinite(duration) and np.isfinite(rc_power))
                else np.nan
            )
            ratio_power = (
                (delta / capex_power)
                if (
                    np.isfinite(delta)
                    and np.isfinite(capex_power)
                    and capex_power > CEILING_ZERO
                )
                else np.nan
            )
            ratio_energy = (
                (delta / capex_energy)
                if (
                    np.isfinite(delta)
                    and np.isfinite(capex_energy)
                    and capex_energy > CEILING_ZERO
                )
                else np.nan
            )
            ratio_proportional = (
                (rc_power / bundle_capex)
                if (
                    np.isfinite(rc_power)
                    and np.isfinite(bundle_capex)
                    and bundle_capex > CEILING_ZERO
                )
                else np.nan
            )

            is_placeholder = not np.isfinite(duration)
            rc_reliable = (case in ("built", "buildable_rc")) and (not is_placeholder)
            if case not in ("built", "buildable_rc"):
                delta = ratio_power = ratio_energy = ratio_proportional = np.nan

            rows.append(
                {
                    tech_col: tech,
                    node_col: key[1],
                    year_col: key[2],
                    "tech_type": "storage",
                    "value_power": value,
                    "value_energy": (
                        float(energy_row["value"]) if energy_row is not None else np.nan
                    ),
                    "capacity_power": capacity,
                    "capacity_energy": (
                        float(energy_row["capacity"])
                        if energy_row is not None
                        else np.nan
                    ),
                    "capacity_ceiling_power": ceiling,
                    "capacity_ceiling_energy": (
                        float(energy_row["capacity_ceiling"])
                        if energy_row is not None
                        else np.nan
                    ),
                    "case": case,
                    "rc_mathematical": rc_power,
                    "e2p": duration,
                    "capex_power": capex_power,
                    "capex_energy": capex_energy,
                    "C_bundle": bundle_capex,
                    "RC_power": delta,
                    "RC_energy": delta,
                    "ratio_power_reduction": ratio_power,
                    "ratio_energy_reduction": ratio_energy,
                    "ratio_reduction_proportional": ratio_proportional,
                    "rc_reliable": rc_reliable,
                }
            )
        if rows:
            result["storage"] = (
                pd.DataFrame(rows)
                .sort_values([tech_col, node_col, year_col])
                .reset_index(drop=True)
            )

    return result
