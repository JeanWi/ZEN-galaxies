
import logging

import numpy as np
import pandas as pd


_RUN_TIMESTAMP: str | None = None

def _get_run_timestamp() -> str:
    """Return a single timestamp for the whole run, cached at module level.

    :return: timestamp string shared by all steps of one program run
    """
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        from datetime import datetime

        _RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _RUN_TIMESTAMP

def _get_output_dir(optimization_setup) -> str:
    """Visualization directory of the dataset currently being optimized.

    Derived from ``analysis.dataset`` so figures land in
    ``<dataset_root>/visualization`` of the respective dataset.

    :param optimization_setup: the optimization setup of the current step
    :return: visualization directory path as ``str``
    """
    from pathlib import Path

    dataset = Path(optimization_setup.analysis.dataset)
    return str(dataset.parent.parent / "visualization")

def _get_run_output_dir(optimization_setup):
    """Timestamped per-run subfolder under the dataset's visualization dir.

    :param optimization_setup: the optimization setup of the current step
    :return: created per-run output ``Path``, shared by every figure/CSV
    """
    from pathlib import Path

    out = Path(_get_output_dir(optimization_setup)) / _get_run_timestamp()
    out.mkdir(parents=True, exist_ok=True)
    return out


# general helper functions
def _normalize_interval(optimization_setup):
    """Interval between optimized years, used to normalize the dual variables.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` of the per-year interval factor
    """
    system = optimization_setup.system
    interval = system.interval_between_years
    years = list(range(0, system.optimized_years))

    factor = pd.Series(index=years, dtype=float)
    for year in years:
        factor[year] = 1 if year == years[-1] else interval
    return factor

def get_lifetime(optimization_setup) -> pd.Series:
    """Lifetime per conversion technology, in calendar years.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` of lifetimes indexed by technology
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])    
    lifetime = optimization_setup.parameters.lifetime.to_series()
    lifetime.name = "lifetime"
    lt = lifetime.reindex(techs)
    return lt

def get_discount_rate(optimization_setup) -> pd.Series:
    """Discount rate per (conversion technology, node).

    The model has only a system-wide scalar ``discount_rate``, so every
    (tech, node) pair receives the same value.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` indexed by (set_technologies, set_nodes)
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    rate = float(np.asarray(optimization_setup.parameters.discount_rate).item())
    idx = pd.MultiIndex.from_product(
        [techs, nodes], names=["set_technologies", "set_nodes"]
    )
    discount_rate = pd.Series(rate, index=idx, name="discount_rate")
    return discount_rate

def _capacity_addition(optimization_setup, capacity_gw: float = 1.0) -> pd.Series:
    """Capacity addition per conversion technology (fixed ``capacity_gw`` for all).

    Thin shell so per-technology/node overrides can be added without touching
    callers.

    :param optimization_setup: the optimization setup of the current step
    :param capacity_gw: capacity addition assigned to every technology
    :return: ``pd.Series`` indexed by set_conversion_technologies
    """
    techs = list(optimization_setup.sets["set_conversion_technologies"])
    return pd.Series(
        capacity_gw,
        index=pd.Index(techs, name="set_conversion_technologies"),
        name="capacity_addition_gw",
    )

def _get_investment_delay(optimization_setup, delay_years: int = 0) -> pd.Series:
    """Investment delay (years) per conversion technology (fixed ``delay_years`` for all).

    Thin shell so per-technology/node overrides can be added without touching
    callers.

    :param optimization_setup: the optimization setup of the current step
    :param delay_years: delay assigned to every technology
    :return: ``pd.Series`` indexed by set_conversion_technologies
    """
    techs = list(optimization_setup.sets["set_conversion_technologies"])
    return pd.Series(
        delay_years,
        index=pd.Index(techs, name="set_conversion_technologies"),
        name="investment_delay_years",
    )


# dual extraction
def extract_aggregated_dual(optimization_setup):
    """Nodal energy-balance duals per aggregated time step (per-year annuity removed).

    :param optimization_setup: the solved optimization setup
    :return: ``pd.DataFrame`` of duals (carrier, node) x time step, or ``None``
    """
    model = optimization_setup.model
    constraint ="constraint_nodal_energy_balance"
    if constraint not in model.constraints:
        logging.warning(
            f"Constraint '{constraint}' not found in the "
            "model. Cannot extract nodal energy balance duals."
        )
        return None

    duals = model.constraints[constraint].dual
    if duals is None:
        logging.warning(
            f"Duals for '{constraint}' are None. Make "
            "sure `solver.save_duals` is enabled and the solver returned duals."
        )
        return None

    # raw duals indexed by (carrier, node, aggregated operation time step)
    df_duals = duals.to_dataframe(name="dual").unstack("set_time_steps_operation")
    df_duals.columns = df_duals.columns.get_level_values("set_time_steps_operation")

    # divide by per-year annuity, matching `Results.get_full_ts` for duals
    time_steps = optimization_setup.energy_system.time_steps
    annuity = _normalize_interval(optimization_setup)
    op2year = pd.Series(time_steps.time_steps_operation2year)
    annuity_per_op = op2year.reindex(df_duals.columns).map(annuity)
    df_duals = df_duals.div(annuity_per_op, axis=1)

    return df_duals

def extract_normalized_dual(optimization_setup):
    """Shadow prices per aggregated time step, normalized to per-hour values.

    :param optimization_setup: the solved optimization setup
    :return: ``pd.DataFrame`` of per-hour duals (carrier, node) 
    """
    df_duals = extract_aggregated_dual(optimization_setup)
    time_steps = optimization_setup.energy_system.time_steps

    # divide by duration: the raw dual is duration * per-hour price
    durations = pd.Series(time_steps.time_steps_operation_duration).reindex(
        df_duals.columns
    )
    df_duals = df_duals.div(durations, axis=1)
   
    return df_duals

def extract_shadow_prices_full_ts(optimization_setup):
    """Shadow prices expanded to the full base time-step resolution (e.g. 8760).

    :param optimization_setup: the solved optimization setup
    :return: ``pd.DataFrame`` of per-hour duals over the base time steps
    """
    time_steps = optimization_setup.energy_system.time_steps
    df_duals = extract_normalized_dual(optimization_setup)
    # map aggregated operation time steps onto the underlying base time steps
    sequence = np.asarray(time_steps.sequence_time_steps_operation)
    full_ts = df_duals.reindex(columns=sequence)
    full_ts.columns = pd.RangeIndex(len(sequence), name="base_time_step")

    return full_ts

def extract_average_shadow_prices(optimization_setup):
    """Average nodal energy-balance shadow price across all time steps.

    :param optimization_setup: the solved optimization setup
    :return: ``pd.DataFrame`` of average shadow prices, or ``None``
    """
    full_ts = extract_shadow_prices_full_ts(optimization_setup)
    if full_ts is None:
        return None

    average_duals = full_ts.mean(axis=1).to_frame(name="average_shadow_price")
    print("\n--- Average shadow prices across time steps ---")
    print(average_duals)
    return average_duals

# flow, production and shadow price calculations
def get_specific_production(optimization_setup) -> pd.Series:
    """Production per GW installed capacity per aggregated operational time step.

    Returns ``flow_conversion_output[t] * duration[t] / capacity[year(t)]``;
    summed over a year this yields the full-load-equivalent production per GW.

    :param optimization_setup: the solved optimization setup
    :return: ``pd.Series`` indexed by (set_technologies, set_output_carriers,
        set_nodes, set_time_steps_operation); ``NaN`` where capacity is zero
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    durations = pd.Series(time_steps.time_steps_operation_duration)
    op_level = "set_time_steps_operation"

    flow = (
        optimization_setup.model.solution["flow_conversion_output"]
        .to_series()
        .dropna()
    )
    flow = flow[
        flow.index.get_level_values("set_conversion_technologies").isin(techs)
    ]
    flow.index = flow.index.rename(
        {"set_conversion_technologies": "set_technologies"}
    )
    # weight each flow value by the duration of its aggregated time step
    flow = flow.mul(flow.index.get_level_values(op_level).map(durations))

    capacity = (
        optimization_setup.model.solution["capacity"]
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    capacity = capacity[
        capacity.index.get_level_values("set_technologies").isin(techs)
    ]
    capacity.index = capacity.index.rename({"set_location": "set_nodes"})

    flow_df = flow.to_frame("flow_energy").reset_index()
    flow_df["set_time_steps_yearly"] = flow_df[op_level].map(op2year)
    cap_df = capacity.to_frame("capacity").reset_index()

    merged = flow_df.merge(
        cap_df,
        on=["set_technologies", "set_nodes", "set_time_steps_yearly"],
        how="left",
    )
    merged["spec"] = merged["flow_energy"] / merged["capacity"]
    result = merged.set_index(
        ["set_technologies", "set_output_carriers", "set_nodes", op_level]
    )["spec"]
    result.name = "specific_production_per_gw"
    return result

def get_flow_reference_carrier(optimization_setup) -> pd.Series:
    """Reference-carrier flow per GW installed capacity per operational time step.

    The reference carrier may be an input or output carrier; both
    ``flow_conversion_output`` and ``flow_conversion_input`` are checked.

    :param optimization_setup: the solved optimization setup
    :return: ``pd.Series`` indexed by (set_technologies, reference_carrier,
        set_nodes, set_time_steps_operation); ``NaN`` where capacity is zero
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    durations = pd.Series(time_steps.time_steps_operation_duration)
    op_level = "set_time_steps_operation"

    ref_carriers = sets["set_reference_carriers"]  # dict: tech -> [carrier]

    def _load_flow(var_name, carrier_level):
        s = optimization_setup.model.solution[var_name].to_series().dropna()
        s = s[s.index.get_level_values("set_conversion_technologies").isin(techs)]
        s.index = s.index.rename({"set_conversion_technologies": "set_technologies"})
        if "set_location" in s.index.names:
            s.index = s.index.rename({"set_location": "set_nodes"})
        return s, carrier_level

    flow_out, out_level = _load_flow("flow_conversion_output", "set_output_carriers")
    flow_in, in_level = _load_flow("flow_conversion_input", "set_input_carriers")

    segments = []
    tech_ref_map: dict[str, str] = {}
    for tech in techs:
        ref = ref_carriers[tech][0]

        # try output carriers first, then input carriers
        for flow_series, carrier_level in [(flow_out, out_level), (flow_in, in_level)]:
            tech_mask = flow_series.index.get_level_values("set_technologies") == tech
            carrier_mask = flow_series.index.get_level_values(carrier_level) == ref
            segment = flow_series[tech_mask & carrier_mask]
            if not segment.empty:
                segments.append(segment.droplevel(carrier_level))
                tech_ref_map[tech] = ref
                break

    if not segments:
        return pd.Series(dtype=float, name="specific_ref_flow_per_gw")

    ref_flow = pd.concat(segments)

    # weight by duration: [energy/time] → [energy] per time-step interval
    ref_flow = ref_flow.mul(ref_flow.index.get_level_values(op_level).map(durations))

    capacity = (
        optimization_setup.model.solution["capacity"]
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    capacity = capacity[capacity.index.get_level_values("set_technologies").isin(techs)]
    if "set_location" in capacity.index.names:
        capacity.index = capacity.index.rename({"set_location": "set_nodes"})

    flow_df = ref_flow.to_frame("flow_energy").reset_index()
    flow_df["set_time_steps_yearly"] = flow_df[op_level].map(op2year)
    cap_df = capacity.to_frame("capacity").reset_index()

    merged = flow_df.merge(
        cap_df,
        on=["set_technologies", "set_nodes", "set_time_steps_yearly"],
        how="left",
    )
    merged["spec"] = merged["flow_energy"] / merged["capacity"]
    merged["reference_carrier"] = merged["set_technologies"].map(tech_ref_map)
    result = merged.set_index(
        ["set_technologies", "reference_carrier", "set_nodes", op_level]
    )["spec"]
    result.name = "specific_ref_flow_per_gw"

    return result

def get_shadow_price(optimization_setup) -> pd.Series | None:
    """Per-aggregated-time-step shadow prices mapped to technologies via their output carriers.

    :param optimization_setup: the solved optimization setup
    :return: ``pd.Series`` indexed by (set_technologies, set_output_carriers,
        set_nodes, set_time_steps_operation), or ``None``
    """
    df = extract_normalized_dual(optimization_setup)
    if df is None:
        return None
    series = df.stack().dropna()
    series.name = "shadow_price"

    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    output_carriers_by_tech = sets["set_output_carriers"]

    long = series.reset_index()
    tech_carrier = pd.DataFrame(
        [(t, c) for t in techs for c in output_carriers_by_tech[t]],
        columns=["set_technologies", "set_output_carriers"],
    )
    merged = tech_carrier.merge(
        long,
        left_on="set_output_carriers",
        right_on="set_carriers",
        how="inner",
    )
    mapped_shadow_price = merged.set_index(
        [
            "set_technologies",
            "set_output_carriers",
            "set_nodes",
            "set_time_steps_operation",
        ]
    )["shadow_price"]
    return mapped_shadow_price
 
def get_flow_input_carrier(optimization_setup) -> pd.Series:
    """Input-carrier flow per GW installed capacity per operational time step.

    Technologies without an input carrier do not appear in the index; treat
    missing entries as 0.

    :param optimization_setup: the solved optimization setup
    :return: ``pd.Series`` indexed by (set_technologies, set_input_carriers,
        set_nodes, set_time_steps_operation); 0 where capacity is zero
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    durations = pd.Series(time_steps.time_steps_operation_duration)
    op_level = "set_time_steps_operation"

    flow = optimization_setup.model.solution["flow_conversion_input"].to_series().dropna()
    flow = flow[flow.index.get_level_values("set_conversion_technologies").isin(techs)]
    flow.index = flow.index.rename({"set_conversion_technologies": "set_technologies"})
    if "set_location" in flow.index.names:
        flow.index = flow.index.rename({"set_location": "set_nodes"})

    if flow.empty:
        return pd.Series(dtype=float, name="specific_input_flow_per_gw")

    # weight by duration: [energy/time] → [energy] per time-step interval
    flow = flow.mul(flow.index.get_level_values(op_level).map(durations))

    capacity = (
        optimization_setup.model.solution["capacity"]
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    capacity = capacity[capacity.index.get_level_values("set_technologies").isin(techs)]
    if "set_location" in capacity.index.names:
        capacity.index = capacity.index.rename({"set_location": "set_nodes"})

    flow_df = flow.to_frame("flow_energy").reset_index()
    flow_df["set_time_steps_yearly"] = flow_df[op_level].map(op2year)
    cap_df = capacity.to_frame("capacity").reset_index()

    merged = flow_df.merge(
        cap_df,
        on=["set_technologies", "set_nodes", "set_time_steps_yearly"],
        how="left",
    )
    merged["spec"] = merged["flow_energy"] / merged["capacity"]
    merged["spec"] = merged["spec"].fillna(0.0)
    result = merged.set_index(
        ["set_technologies", "set_input_carriers", "set_nodes", op_level]
    )["spec"]
    result.name = "specific_input_flow_per_gw"
    return result



# revenue and cost calculations
def calculate_revenue(optimization_setup) -> pd.Series:
    """Discounted lifetime revenue for a hypothetical capacity addition.

    Per (tech, output carrier, node) the annual revenue
    ``Σ_t shadow_price * specific_production`` of the single optimization year
    is taken as representative and discounted over the technology lifetime,
    shifted by the investment delay and scaled by the capacity addition.

    :param optimization_setup: the solved optimization setup
    :return: ``pd.Series`` of discounted revenue indexed by
        (set_technologies, set_output_carriers, set_nodes)
    """
    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    spec_prod = get_specific_production(optimization_setup)
    prices = get_shadow_price(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition_gw = _capacity_addition(optimization_setup)
    if spec_prod is None or prices is None or spec_prod.empty:
        print("\n--- Revenue calculation skipped: missing specific production or shadow price data ---\n")
        return pd.Series(dtype=float, name="discounted_revenue")

    # aggregate spec_prod * shadow_price over all operational time steps
    product = spec_prod.mul(prices).dropna()
    product.name = "prod_revenue"
    product_df = product.reset_index()
    annual_rev = (
        product_df.groupby(
            ["set_technologies", "set_output_carriers", "set_nodes"]
        )["prod_revenue"].sum()
    )


    group_index = annual_rev.index

    revenue = pd.Series(0.0, index=group_index, name="discounted_revenue")
    for tech, carrier, node in group_index:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = investment_delay[tech]
        capacity = capacity_addition_gw[tech]
        r = float(discount_rate.loc[(tech, node)])
        total = 0.0
        for offset in range(0 + delay,tech_lifetime + delay):
            disc = (1 + r) ** offset
            key = (tech, carrier, node)
            if key not in annual_rev.index:
                continue
            val = annual_rev.loc[key]
            if pd.isna(val):
                continue
            total += float(val) / disc
        revenue.loc[(tech, carrier, node)] = capacity * total
    return revenue

def get_capex(optimization_setup) -> pd.Series:
    """Upfront (overnight) investment cost [MEUR] for the capacity addition.

    Supports linear (``capex_specific_conversion * capacity_addition``) and PWA
    technologies (interpolated from the breakpoint curve, node-independent).
    Conversion technologies only.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` indexed by (set_conversion_technologies, set_nodes)
    """
    from zen_garden.model.technology.conversion_technology import ConversionTechnology

    sets = optimization_setup.sets
    parameters = optimization_setup.parameters
    latest_year = max(sets["set_time_steps_yearly"])
    nodes = list(sets["set_nodes"])
    capacity = _capacity_addition(optimization_setup)

    def _select_year(param) -> pd.Series:
        series = param.to_series().dropna()
        for level_name in ("set_time_steps_yearly", "year"):
            if level_name in series.index.names:
                return series.xs(latest_year, level=level_name)
        return series

    records: list[dict] = []

    # linear conversion technologies
    if hasattr(parameters, "capex_specific_conversion"):
        capex_specific = _select_year(parameters.capex_specific_conversion)
        for (tech, node), specific in capex_specific.items():
            if tech not in capacity.index:
                continue
            records.append(
                {
                    "set_conversion_technologies": tech,
                    "set_nodes": node,
                    "capex": float(specific) * float(capacity[tech]),
                }
            )

    # PWA conversion technologies (node-independent curve → same value for all nodes)
    for tech in sets["set_conversion_technologies"]:
        if not optimization_setup.get_attribute_of_specific_element(
            ConversionTechnology, tech, "capex_is_pwa"
        ):
            continue
        pwa_data = optimization_setup.get_attribute_of_specific_element(
            ConversionTechnology, tech, "pwa_capex"
        )
        print(f"\n used PWA for " + tech)
        cap = float(capacity[tech])
        total_capex = float(
            np.interp(cap, pwa_data["capacity_addition"], pwa_data["capex"])
        )
        for node in nodes:
            records.append(
                {
                    "set_conversion_technologies": tech,
                    "set_nodes": node,
                    "capex": total_capex,
                }
            )

    if not records:
        result = pd.Series(dtype=float, name="capex")
    else:
        result = pd.DataFrame(records).set_index(
            ["set_conversion_technologies", "set_nodes"]
        )["capex"]
        result.name = "capex"

   
    return result

def get_fixed_opex_discounted(optimization_setup) -> pd.Series:
    """Discounted lifetime fixed OPEX [MEUR] for the capacity addition.

    Takes the decision-year ``opex_specific_fixed`` as constant over the
    lifetime and discounts it with the same capacity addition and investment
    delay as the revenue calculation. Conversion technologies only.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` indexed by (set_conversion_technologies, set_nodes)
    """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters

    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    base_year = max(sets["set_time_steps_yearly"])

    # sum over capacity types; rename set_location → set_nodes
    opex_series = (
        parameters.opex_specific_fixed
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    if "set_location" in opex_series.index.names:
        opex_series.index = opex_series.index.rename({"set_location": "set_nodes"})
    opex_series = opex_series[
        opex_series.index.get_level_values("set_technologies").isin(techs)
    ]

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    idx = pd.MultiIndex.from_product(
        [techs, nodes],
        names=["set_conversion_technologies", "set_nodes"],
    )
    result = pd.Series(0.0, index=idx, name="fixed_opex_discounted")

    for tech in techs:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        cap = float(capacity_addition[tech])

        for node in nodes:
            r = float(discount_rate.loc[(tech, node)])
            total = 0.0

            try:
                opex_val = float(opex_series.loc[(tech, node, base_year)])
            except KeyError:
                continue

            for offset in range(delay, tech_lifetime + delay):
                total += opex_val / (1 + r) ** offset

            result.loc[(tech, node)] = cap * total

    return result

def get_variable_opex_discounted(optimization_setup) -> pd.Series:
    """Discounted lifetime variable OPEX [MEUR] for the capacity addition.

    Multiplies the decision-year ``opex_specific_variable`` by the specific
    reference-carrier flow per GW, then discounts over the lifetime with the
    same investment delay as the revenue calculation. Conversion technologies
    only.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` indexed by (set_conversion_technologies, set_nodes)
    """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters

    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    op_level = "set_time_steps_operation"
    base_year = max(sets["set_time_steps_yearly"])

    # extract opex_specific_variable [EUR/MWh] per operational time step
    opex_var = parameters.opex_specific_variable.to_series().dropna()
    opex_var = opex_var[opex_var.index.get_level_values("set_technologies").isin(techs)]
    if "set_location" in opex_var.index.names:
        opex_var.index = opex_var.index.rename({"set_location": "set_nodes"})

    # specific reference flow per GW [MWh/GW per time step]
    ref_flow = get_flow_reference_carrier(optimization_setup)
    ref_flow = ref_flow.droplevel("reference_carrier")

    # multiply: [EUR/MWh] * [MWh/GW] = [EUR/GW] per time step, then sum per year
    opex_var_df = opex_var.rename("opex_var").reset_index()
    ref_flow_df = ref_flow.rename("ref_flow").reset_index()
    product_df = opex_var_df.merge(
        ref_flow_df,
        on=["set_technologies", "set_nodes", op_level],
        how="inner",
    )
    product_df["cost"] = product_df["opex_var"] * product_df["ref_flow"]
    product_df["set_time_steps_yearly"] = product_df[op_level].map(op2year)
    
    annual_var_opex = (
        product_df
        .groupby(["set_technologies", "set_nodes", "set_time_steps_yearly"])["cost"]
        .sum()
    )

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    idx = pd.MultiIndex.from_product(
        [techs, nodes],
        names=["set_conversion_technologies", "set_nodes"],
    )
    result = pd.Series(0.0, index=idx, name="variable_opex_discounted")

    for tech in techs:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        cap = float(capacity_addition[tech])

        for node in nodes:
            r = float(discount_rate.loc[(tech, node)])
            total = 0.0

            try:
                opex_val = float(annual_var_opex.loc[(tech, node, base_year)])
            except KeyError:
                continue

            for offset in range(delay, tech_lifetime + delay):
                total += opex_val / (1 + r) ** offset

            result.loc[(tech, node)] = cap * total

    return result

def calculate_input_carrier_cost(optimization_setup) -> pd.Series:
    """Discounted lifetime input-carrier cost for a hypothetical capacity addition.

    Analogous to ``calculate_revenue`` but for input carriers, using each input
    carrier's nodal shadow price as the fuel/energy price.

    :param optimization_setup: the solved optimization setup
    :return: ``pd.Series`` of discounted cost indexed by
        (set_technologies, set_input_carriers, set_nodes)
    """
    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition_gw = _capacity_addition(optimization_setup)

    input_flow = get_flow_input_carrier(optimization_setup)
    if input_flow.empty:
        print("\n--- Input carrier cost skipped: no input flow data ---\n")
        return pd.Series(dtype=float, name="discounted_input_carrier_cost")

    duals = extract_normalized_dual(optimization_setup)
    if duals is None:
        print("\n--- Input carrier cost skipped: no dual variables available ---\n")
        return pd.Series(dtype=float, name="discounted_input_carrier_cost")

    # stack duals to (set_carriers, set_nodes, set_time_steps_operation)
    carrier_prices = duals.stack().dropna()
    carrier_prices.name = "carrier_price"

    flow_df = input_flow.reset_index()
    prices_df = carrier_prices.reset_index()

    merged = flow_df.merge(
        prices_df,
        left_on=["set_input_carriers", "set_nodes", "set_time_steps_operation"],
        right_on=["set_carriers", "set_nodes", "set_time_steps_operation"],
        how="inner",
    )
    merged["cost"] = merged["specific_input_flow_per_gw"] * merged["carrier_price"]

    annual_cost = (
        merged.groupby(["set_technologies", "set_input_carriers", "set_nodes"])["cost"]
        .sum()
    )

    if annual_cost.empty:
        print("\n--- Input carrier cost skipped: no matching carrier prices ---\n")
        return pd.Series(dtype=float, name="discounted_input_carrier_cost")

    group_index = annual_cost.index
    result = pd.Series(0.0, index=group_index, name="discounted_input_carrier_cost")

    for tech, carrier, node in group_index:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        capacity = float(capacity_addition_gw[tech])
        r = float(discount_rate.loc[(tech, node)])
        val = annual_cost.loc[(tech, carrier, node)]
        if pd.isna(val):
            continue
        total = sum(
            float(val) / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        result.loc[(tech, carrier, node)] = capacity * total

    return result

def get_tech_co2_cost_discounted(optimization_setup) -> pd.Series:
    """Discounted lifetime CO2 emission cost of the technology for the capacity addition.

    Like ``get_variable_opex_discounted`` but multiplies the reference-carrier
    flow by ``carbon_intensity_technology`` and the base-year
    ``price_carbon_emissions``. Conversion technologies only.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` indexed by (set_conversion_technologies, set_nodes)
    """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters

    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    op_level = "set_time_steps_operation"
    base_year = max(sets["set_time_steps_yearly"])

    # carbon intensity [tCO2/MWh] per (tech, node) — no time dimension in ZEN-Garden
    carbon_intensity = parameters.carbon_intensity_technology.to_series().dropna()
    if "set_location" in carbon_intensity.index.names:
        carbon_intensity.index = carbon_intensity.index.rename(
            {"set_location": "set_nodes"}
        )
    carbon_intensity = carbon_intensity[
        carbon_intensity.index.get_level_values("set_technologies").isin(techs)
    ]

    # CO2 price [EUR/tCO2] at the base year
    price_co2 = parameters.price_carbon_emissions.to_series().dropna()
    price_co2_base = float(price_co2.loc[base_year])

    # specific reference flow per GW [MWh/GW per time step]
    ref_flow = get_flow_reference_carrier(optimization_setup)
    if ref_flow.empty:
        return pd.Series(dtype=float, name="co2_cost_discounted")
    ref_flow = ref_flow.droplevel("reference_carrier")

    # [MWh/GW] * [tCO2/MWh] * [EUR/tCO2] = [EUR/GW] per time step, then sum per year
    ref_flow_df = ref_flow.rename("ref_flow").reset_index()
    ci_df = carbon_intensity.rename("ci").reset_index()
    product_df = ref_flow_df.merge(
        ci_df, on=["set_technologies", "set_nodes"], how="inner"
    )
    product_df["cost"] = (
        product_df["ref_flow"] * product_df["ci"] * price_co2_base
    )
    product_df["set_time_steps_yearly"] = product_df[op_level].map(op2year)

    annual_co2_cost = (
        product_df
        .groupby(["set_technologies", "set_nodes", "set_time_steps_yearly"])["cost"]
        .sum()
    )

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    idx = pd.MultiIndex.from_product(
        [techs, nodes],
        names=["set_conversion_technologies", "set_nodes"],
    )
    result = pd.Series(0.0, index=idx, name="co2_cost_discounted")

    for tech in techs:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        cap = float(capacity_addition[tech])

        for node in nodes:
            r = float(discount_rate.loc[(tech, node)])
            try:
                co2_val = float(annual_co2_cost.loc[(tech, node, base_year)])
            except KeyError:
                continue  # tech has no emissions at this node

            total = sum(
                co2_val / (1 + r) ** offset
                for offset in range(delay, tech_lifetime + delay)
            )
            result.loc[(tech, node)] = cap * total

    return result

def get_carrier_co2_cost_discounted(optimization_setup) -> pd.Series:
    """Discounted lifetime CO2 cost from input carriers for the capacity addition.

    Like ``calculate_input_carrier_cost`` but multiplies the input-carrier flow
    by ``carbon_intensity_carrier_import`` and the base-year
    ``price_carbon_emissions``; captures upstream/import emissions.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` indexed by
        (set_technologies, set_input_carriers, set_nodes)
    """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters

    base_year = max(sets["set_time_steps_yearly"])

    # carbon intensity of carrier import [tCO2/MWh] per (carrier, node), base year
    carrier_ci = parameters.carbon_intensity_carrier_import.to_series().dropna()
    if "set_location" in carrier_ci.index.names:
        carrier_ci.index = carrier_ci.index.rename({"set_location": "set_nodes"})
    for year_level in ("set_time_steps_yearly", "year"):
        if year_level in carrier_ci.index.names:
            carrier_ci = carrier_ci.xs(base_year, level=year_level)
            break

    # CO2 price [EUR/tCO2] at the base year
    price_co2 = parameters.price_carbon_emissions.to_series().dropna()
    price_co2_base = float(price_co2.loc[base_year])

    # specific input-carrier flow per GW [MWh/GW per time step]
    input_flow = get_flow_input_carrier(optimization_setup)
    if input_flow.empty:
        return pd.Series(dtype=float, name="carrier_co2_cost_discounted")

    # [MWh/GW] * [tCO2/MWh] * [EUR/tCO2] = [EUR/GW] per time step, then sum over time
    flow_df = input_flow.rename("flow").reset_index()
    ci_df = carrier_ci.rename("ci").reset_index().rename(
        columns={"set_carriers": "set_input_carriers"}
    )
    merged = flow_df.merge(
        ci_df, on=["set_input_carriers", "set_nodes"], how="inner"
    )
    if merged.empty:
        return pd.Series(dtype=float, name="carrier_co2_cost_discounted")

    merged["cost"] = merged["flow"] * merged["ci"] * price_co2_base

    annual_cost = (
        merged
        .groupby(["set_technologies", "set_input_carriers", "set_nodes"])["cost"]
        .sum()
    )

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition_gw = _capacity_addition(optimization_setup)

    group_index = annual_cost.index
    result = pd.Series(0.0, index=group_index, name="carrier_co2_cost_discounted")

    for tech, carrier, node in group_index:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        capacity = float(capacity_addition_gw[tech])
        r = float(discount_rate.loc[(tech, node)])
        val = annual_cost.loc[(tech, carrier, node)]
        if pd.isna(val):
            continue
        total = sum(
            float(val) / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        result.loc[(tech, carrier, node)] = capacity * total

    return result


# subsidies
def _get_subsidies(optimization_setup, subsidy_type=None) -> list[dict]:
    """Validated subsidy entries from the plugin config.

    Reads ``config["subsidies"]`` and returns normalized dicts
    ``{technology, node, type, amount, carrier}``. Malformed entries or entries
    referencing an unknown technology, node or type are dropped with a warning;
    ``remuneration`` entries require ``carrier`` to be an output carrier.

    :param optimization_setup: the optimization setup of the current step
    :param subsidy_type: optional restriction to a single subsidy type
    :return: list of validated subsidy dicts
    """
    try:
        from zen_garden.plugins.investment_decisions.plugin import config
    except Exception:
        return []
    raw = config.get("subsidies", []) or []

    sets = optimization_setup.sets
    techs = set(sets["set_conversion_technologies"])
    nodes = set(sets["set_nodes"])
    valid_types = {"capex", "fixed_opex", "variable_opex", "remuneration"}

    out: list[dict] = []
    for entry in raw:
        try:
            tech = entry["technology"]
            node = entry["node"]
            stype = entry["type"]
            amount = float(entry["amount"])
        except (KeyError, TypeError, ValueError):
            logging.warning(f"Subsidy entry malformed, skipped: {entry!r}")
            continue
        if stype not in valid_types:
            logging.warning(f"Subsidy of unknown type {stype!r}, skipped: {entry!r}")
            continue
        if tech not in techs:
            logging.warning(
                f"Subsidy references unknown technology {tech!r}, skipped: {entry!r}"
            )
            continue
        if node not in nodes:
            logging.warning(
                f"Subsidy references unknown node {node!r}, skipped: {entry!r}"
            )
            continue

        # remuneration applies to one configured output carrier of the tech.
        carrier = entry.get("carrier")
        if stype == "remuneration":
            out_carriers = sets["set_output_carriers"][tech]  # tech already validated
            if carrier is None:
                logging.warning(
                    f"Remuneration subsidy requires an output 'carrier', "
                    f"skipped: {entry!r}"
                )
                continue
            if carrier not in out_carriers:
                logging.warning(
                    f"Remuneration subsidy carrier {carrier!r} is not an output "
                    f"carrier of {tech!r}, skipped: {entry!r}"
                )
                continue

        if subsidy_type is not None and stype != subsidy_type:
            continue
        out.append(
            {
                "technology": tech,
                "node": node,
                "type": stype,
                "amount": amount,
                "carrier": carrier,
            }
        )
    return out

def _subsidy_frame(records: list[dict]) -> pd.DataFrame:
    """Build the standard subsidy DataFrame from per-(tech, node) records.

    Entries for the same (tech, node) are summed; an empty list yields an empty,
    correctly-typed frame.

    :param records: per-(tech, node) subsidy records
    :return: ``pd.DataFrame`` indexed by (set_conversion_technologies,
        set_nodes) with columns ``npv`` and ``annual``
    """
    cols = ["npv", "annual"]
    if not records:
        idx = pd.MultiIndex.from_arrays(
            [[], []], names=["set_conversion_technologies", "set_nodes"]
        )
        return pd.DataFrame(0.0, index=idx, columns=cols)
    df = pd.DataFrame(records)
    # min_count=1 keeps an all-NaN group (e.g. one-time CAPEX subsidy) as NaN, not 0
    return df.groupby(["set_conversion_technologies", "set_nodes"])[cols].sum(
        min_count=1
    )

def _peer_avg_production(spec_prod: pd.Series, tech: str, carrier: str):
    """Average specific-production profile over peer nodes that produce.

    Estimates production at a subsidy node with no flow from the per-time-step
    mean over the (tech, carrier) nodes with positive annual production.

    :param spec_prod: specific production from ``get_specific_production``
    :param tech: technology name
    :param carrier: output carrier name
    :return: averaged profile, or ``None`` if no peer node produces
    """
    try:
        sub = spec_prod.loc[(tech, carrier)]  # indexed by (set_nodes, time step)
    except KeyError:
        return None
    node_sum = sub.groupby(level="set_nodes").sum()
    flow_nodes = node_sum.index[node_sum > 0]
    if len(flow_nodes) == 0:
        return None
    peer = sub[sub.index.get_level_values("set_nodes").isin(flow_nodes)]
    return peer.groupby(level="set_time_steps_operation").mean()

def get_capex_subsidy(optimization_setup) -> pd.DataFrame:
    """CAPEX subsidy: a one-time relief in the decision year, scaled per GW.

    Relief is ``amount * capacity_addition``, paid once and undiscounted
    (mirroring the overnight CAPEX); ``annual`` is therefore ``NaN``.

    :param optimization_setup: the optimization setup of the current step
    :return: subsidy ``pd.DataFrame`` (see ``_subsidy_frame``)
    """
    entries = _get_subsidies(optimization_setup, "capex")
    capacity_addition = _capacity_addition(optimization_setup)

    records = []
    for e in entries:
        tech, node, amount = e["technology"], e["node"], e["amount"]
        value = amount * float(capacity_addition.get(tech, 0.0))
        records.append(
            {
                "set_conversion_technologies": tech,
                "set_nodes": node,
                "npv": value,
                "annual": np.nan,
            }
        )
    return _subsidy_frame(records)

def get_fixed_opex_subsidy(optimization_setup) -> pd.DataFrame:
    """Fixed-OPEX subsidy: an annual lump-sum relief, scaled per GW.

    Annual relief ``amount * capacity_addition`` recurring over the lifetime,
    discounted with the same discount rate and investment delay as the OPEX
    components.

    :param optimization_setup: the optimization setup of the current step
    :return: subsidy ``pd.DataFrame`` (see ``_subsidy_frame``)
    """
    entries = _get_subsidies(optimization_setup, "fixed_opex")
    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    records = []
    for e in entries:
        tech, node, amount = e["technology"], e["node"], e["amount"]
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        annual = amount * float(capacity_addition.get(tech, 0.0))
        r = float(discount_rate.loc[(tech, node)])
        npv = sum(
            annual / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        records.append(
            {
                "set_conversion_technologies": tech,
                "set_nodes": node,
                "npv": npv,
                "annual": annual,
            }
        )
    return _subsidy_frame(records)

def get_variable_opex_subsidy(optimization_setup) -> pd.DataFrame:
    """Variable-OPEX subsidy: a per-MWh relief on the reference-carrier flow.

    ``amount`` [money/MWh] times the base-year specific reference-carrier flow
    per GW and the capacity addition, discounted over the lifetime. Nodes
    without flow fall back to the average flow of peer nodes of the same tech.

    :param optimization_setup: the optimization setup of the current step
    :return: subsidy ``pd.DataFrame`` (see ``_subsidy_frame``)
    """
    entries = _get_subsidies(optimization_setup, "variable_opex")
    if not entries:
        return _subsidy_frame([])

    sets = optimization_setup.sets
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    op_level = "set_time_steps_operation"
    base_year = max(sets["set_time_steps_yearly"])

    # specific reference-carrier flow per GW [MWh/GW per time step], summed per year
    ref_flow = get_flow_reference_carrier(optimization_setup)
    if ref_flow.empty:
        return _subsidy_frame([])
    ref_flow = ref_flow.droplevel("reference_carrier")
    ref_flow_df = ref_flow.rename("ref_flow").reset_index()
    ref_flow_df["set_time_steps_yearly"] = ref_flow_df[op_level].map(op2year)
    annual_ref_flow = (
        ref_flow_df
        .groupby(["set_technologies", "set_nodes", "set_time_steps_yearly"])["ref_flow"]
        .sum()
    )
    # base-year flow per (tech, node), used both directly and for the peer average
    base_flow = annual_ref_flow.xs(base_year, level="set_time_steps_yearly")

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    records = []
    for e in entries:
        tech, node, amount = e["technology"], e["node"], e["amount"]
        if pd.isna(lifetime.get(tech, np.nan)):
            continue

        flow_val = float(base_flow.get((tech, node), 0.0))
        if not (flow_val > 0):
            # node has no reference-carrier flow -> use the average over peer nodes
            tech_flows = (
                base_flow.loc[tech]
                if tech in base_flow.index.get_level_values("set_technologies")
                else pd.Series(dtype=float)
            )
            peers = tech_flows[tech_flows > 0]
            if len(peers) == 0:
                continue
            flow_val = float(peers.mean())

        annual = amount * flow_val * float(capacity_addition.get(tech, 0.0))
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        r = float(discount_rate.loc[(tech, node)])
        npv = sum(
            annual / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        records.append(
            {
                "set_conversion_technologies": tech,
                "set_nodes": node,
                "npv": npv,
                "annual": annual,
            }
        )
    return _subsidy_frame(records)

def get_remuneration_subsidy(optimization_setup) -> pd.DataFrame:
    """Remuneration subsidy: a fixed feed-in price replacing the shadow price.

    Annual extra revenue ``capacity_addition * Σ_t production *
    (amount - shadow_price)`` for the configured output carrier, discounted over
    the lifetime like ``calculate_revenue``. Nodes without production fall back
    to the peer-node average profile; the shadow price stays node-specific.

    :param optimization_setup: the optimization setup of the current step
    :return: subsidy ``pd.DataFrame`` (see ``_subsidy_frame``)
    """
    entries = _get_subsidies(optimization_setup, "remuneration")
    if not entries:
        return _subsidy_frame([])

    spec_prod = get_specific_production(optimization_setup)
    prices = get_shadow_price(optimization_setup)
    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    records = []
    for e in entries:
        tech, node, price, carrier = (
            e["technology"], e["node"], e["amount"], e["carrier"]
        )
        if pd.isna(lifetime.get(tech, np.nan)):
            continue

        key = (tech, carrier, node)
        # production profile at the node; fall back to the peer-node average if absent
        try:
            own = spec_prod.loc[key]
        except KeyError:
            own = None
        if own is not None and float(own.sum()) > 0:
            production = own
        else:
            production = _peer_avg_production(spec_prod, tech, carrier)
            if production is None:
                # no peer node produces this carrier -> cannot estimate
                continue

        try:
            market_price = prices.loc[key]
        except KeyError:
            market_price = pd.Series(0.0, index=production.index)
        diff = (price - market_price).reindex(production.index).fillna(price)
        annual = float((production * diff).sum()) * float(
            capacity_addition.get(tech, 0.0)
        )
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        r = float(discount_rate.loc[(tech, node)])
        npv = sum(
            annual / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        records.append(
            {
                "set_conversion_technologies": tech,
                "set_nodes": node,
                "npv": npv,
                "annual": annual,
            }
        )
    return _subsidy_frame(records)


# profitability calculation
def calculate_profitability(optimization_setup) -> pd.DataFrame:
    """Profitability of the capacity addition as revenue minus costs plus subsidies.

    Costs are CAPEX, fixed/variable OPEX, input-carrier cost and CO2 cost
    (technology and carrier); configured subsidies are added as discounted NPVs.
    Zero-revenue (tech, node) pairs have their raw profitability (before
    subsidies) replaced by the average of revenue-positive peer nodes of the
    same technology and are flagged in ``values_changed`` (their cost/revenue
    components are then set to ``NaN``, the subsidies are kept).

    :param optimization_setup: the solved optimization setup
    :return: ``pd.DataFrame`` indexed by (set_conversion_technologies,
        set_nodes) with the component, subsidy, ``profitability_raw``,
        ``profitability`` and ``values_changed`` columns
    """
    revenue = calculate_revenue(optimization_setup)
    capex = get_capex(optimization_setup)
    fixed_opex = get_fixed_opex_discounted(optimization_setup)
    variable_opex = get_variable_opex_discounted(optimization_setup)
    input_carrier_cost = calculate_input_carrier_cost(optimization_setup)
    tech_co2_cost = get_tech_co2_cost_discounted(optimization_setup)
    carrier_co2_cost = get_carrier_co2_cost_discounted(optimization_setup)

    # subsidies (positive cash flows): discounted ``npv`` plus undiscounted ``annual`` (CSV)
    subsidy_frames = {
        "capex_subsidy": get_capex_subsidy(optimization_setup),
        "fixed_opex_subsidy": get_fixed_opex_subsidy(optimization_setup),
        "variable_opex_subsidy": get_variable_opex_subsidy(optimization_setup),
        "remuneration_subsidy": get_remuneration_subsidy(optimization_setup),
    }

    # sum over carriers and rename to the (set_conversion_technologies, set_nodes) index
    def _by_tech_node(series):
        return (
            series
            .groupby(level=["set_technologies", "set_nodes"])
            .sum()
            .rename_axis(index={"set_technologies": "set_conversion_technologies"})
        )

    revenue_by_tech_node = _by_tech_node(revenue)
    input_carrier_cost_by_tech_node = _by_tech_node(input_carrier_cost)
    carrier_co2_cost_by_tech_node = _by_tech_node(carrier_co2_cost)

    profitability = revenue_by_tech_node.subtract(capex, fill_value=0)
    profitability = profitability.subtract(fixed_opex, fill_value=0)
    profitability = profitability.subtract(variable_opex, fill_value=0)
    profitability = profitability.subtract(input_carrier_cost_by_tech_node, fill_value=0)
    profitability = profitability.subtract(tech_co2_cost, fill_value=0)
    profitability = profitability.subtract(carrier_co2_cost_by_tech_node, fill_value=0)
    profitability.name = "profitability"

    values_changed = pd.Series(False, index=profitability.index, name="values_changed")

    # for 0-revenue pairs, replace the raw profitability with the average of
    # revenue-positive peers of the same tech (before subsidies; unchanged if no peers)
    zero_rev_pairs = [
        idx for idx in profitability.index
        if revenue_by_tech_node.get(idx, 0.0) == 0.0
    ]
    for idx in zero_rev_pairs:
        tech, node = idx
        peer_profits = [
            float(profitability.loc[peer_idx])
            for peer_idx in profitability.index
            if peer_idx[0] == tech
            and peer_idx != idx
            and revenue_by_tech_node.get(peer_idx, 0.0) > 0.0
        ]
        if peer_profits:
            fill_val = float(np.mean(peer_profits))
            profitability.loc[idx] = fill_val
            values_changed.loc[idx] = True
            print(
                f"Profitability fill: ({tech}, {node}) has 0 revenue → "
                f"raw profitability replaced with avg of revenue-positive peers "
                f"= {fill_val:.2f}"
            )

    # raw profitability (after peer-fill, before subsidies), kept as its own component
    raw_profitability = profitability.copy()
    raw_profitability.name = "profitability_raw"

    # add the discounted subsidy NPVs on top of the raw profitability
    for frame in subsidy_frames.values():
        profitability = profitability.add(frame["npv"], fill_value=0)
    profitability.name = "profitability"

    components = pd.DataFrame(
        {
            "revenue": revenue_by_tech_node,
            "capex": capex,
            "fixed_opex": fixed_opex,
            "variable_opex": variable_opex,
            "input_carrier_cost": input_carrier_cost_by_tech_node,
            "tech_co2_cost": tech_co2_cost,
            "carrier_co2_cost": carrier_co2_cost_by_tech_node,
        }
    ).fillna(0.0)

    # subsidies as separate components: discounted ``npv`` and undiscounted ``annual`` (CSV)
    for name, frame in subsidy_frames.items():
        components[name] = frame["npv"].reindex(components.index, fill_value=0.0)
        components[f"{name}_annual"] = frame["annual"].reindex(
            components.index, fill_value=0.0
        )

    components["profitability_raw"] = raw_profitability.reindex(components.index)
    components["profitability"] = profitability.reindex(components.index)
    components["values_changed"] = values_changed.reindex(components.index, fill_value=False)

    # filled entries: cost/revenue components are meaningless → NaN (node-specific subsidies kept)
    component_cols = [
        "revenue", "capex", "fixed_opex", "variable_opex",
        "input_carrier_cost", "tech_co2_cost", "carrier_co2_cost",
    ]
    components.loc[components["values_changed"], component_cols] = np.nan

    # one column per subsidy type, added after the raw profitability
    subsidy_cols = list(subsidy_frames)
    subsidy_labels = {
        "capex_subsidy": "Capex Sub",
        "fixed_opex_subsidy": "FixOpx Sub",
        "variable_opex_subsidy": "VarOpx Sub",
        "remuneration_subsidy": "Remun Sub",
    }
    header = (
        f"{'Tech / Node':<45} {'Revenue':>12} {'CAPEX':>12} "
        f"{'Fixed OPEX':>12} {'Var OPEX':>12} {'Input Cost':>12} "
        f"{'Tech CO2':>12} {'Carrier CO2':>12} {'Raw Profit':>12} "
        + "".join(f"{subsidy_labels.get(c, c):>12} " for c in subsidy_cols)
        + f"{'Profit':>12} {'Changed':>9}"
    )
    print(
        f"\n--- Profitability of Capacity Additions ---\n"
        + header + "\n"
        + "-" * len(header)
    )
    for idx, row in components.iterrows():
        print(
            f"{str(idx):<45} {row['revenue']:>12.2f} {row['capex']:>12.2f} "
            f"{row['fixed_opex']:>12.2f} {row['variable_opex']:>12.2f} "
            f"{row['input_carrier_cost']:>12.2f} {row['tech_co2_cost']:>12.2f} "
            f"{row['carrier_co2_cost']:>12.2f} {row['profitability_raw']:>12.2f} "
            + "".join(f"{row[c]:>12.2f} " for c in subsidy_cols)
            + f"{row['profitability']:>12.2f} "
            f"{'*' if row['values_changed'] else '':>9}"
        )
    print()
    return components



# bias normalization
def get_npc_capex_coefficient(optimization_setup) -> pd.Series:
    """Per-GW coefficient mapping a capacity addition onto its CAPEX share of the
    base objective ``Σ_y NPC_y``, for linear conversion technologies only.

    Reconstructs the model chain ``∂base / ∂Δcapacity`` for a build in the
    decision year (first horizon year) as ``annuity * specific_capex *
    Σ_y discount_factor_y`` over the horizon years whose depreciation range
    still contains the decision year. PWA technologies have no constant per-GW
    value and are returned as ``NaN``.

    :param optimization_setup: the optimization setup of the current step
    :return: ``pd.Series`` indexed by (set_conversion_technologies, set_nodes);
        PWA technologies are ``NaN``
    """
    from zen_garden.model.technology.technology import Technology
    from zen_garden.model.technology.conversion_technology import ConversionTechnology

    sets = optimization_setup.sets
    parameters = optimization_setup.parameters
    energy_system = optimization_setup.energy_system
    system = optimization_setup.system

    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    years = list(sets["set_time_steps_yearly"])
    year_build = years[0]  # decision year: matches apply_profitability_bias_objective

    # scalar discount rate, exactly as used in the model constraints
    dr = float(np.asarray(parameters.discount_rate).item())
    depreciation_time = parameters.depreciation_time.to_series()

    # --- economic discount factor phi_y (mirror constraint_net_present_cost) ---
    entire_horizon = energy_system.set_time_steps_yearly_entire_horizon
    interval = system.interval_between_years
    discount_factor: dict[int, float] = {}
    for year in years:
        interval_between = 1 if year == entire_horizon[-1] else interval
        discount_factor[year] = sum(
            (1 / (1 + dr)) ** (interval * (year - year_build) + i)
            for i in range(0, interval_between)
        )

    # --- specific overnight CAPEX alpha of the decision year (linear only) ---
    capex_specific = parameters.capex_specific_conversion.to_series().dropna()
    for level_name in ("set_time_steps_yearly", "year"):
        if level_name in capex_specific.index.names:
            capex_specific = capex_specific.xs(year_build, level=level_name)
            break

    idx = pd.MultiIndex.from_product(
        [techs, nodes], names=["set_conversion_technologies", "set_nodes"]
    )
    result = pd.Series(np.nan, index=idx, name="npc_capex_coefficient")

    for tech in techs:
        # PWA technologies have no constant per-GW coefficient -> leave as NaN
        if optimization_setup.get_attribute_of_specific_element(
            ConversionTechnology, tech, "capex_is_pwa"
        ):
            continue

        lt = float(depreciation_time[tech])
        if dr != 0:
            annuity = ((1 + dr) ** lt * dr) / ((1 + dr) ** lt - 1)
        else:
            annuity = 1.0 / lt

        # discount factors of horizon years whose depreciation range contains the decision year
        disc_sum = 0.0
        for year in years:
            deprange = Technology.get_lifetime_range(
                optimization_setup, tech, year, use_depreciation_time=True
            )
            if year_build in deprange:
                disc_sum += discount_factor[year]

        for node in nodes:
            if (tech, node) not in capex_specific.index:
                continue
            alpha = float(capex_specific.loc[(tech, node)])
            result.loc[(tech, node)] = annuity * alpha * disc_sum

    return result

def _get_bias_technologies(optimization_setup, output_carriers=None) -> set:
    """Conversion technologies eligible for the profitability bias.

    A technology qualifies when at least one of its output carriers is in
    ``output_carriers``; ``None``/empty selects all conversion technologies. A
    single carrier may be passed as a string.

    :param optimization_setup: the optimization setup of the current step
    :param output_carriers: optional carrier restriction
    :return: ``set`` of technology names (empty, with a warning, if none match)
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    if not output_carriers:
        return set(techs)
    if isinstance(output_carriers, str):
        output_carriers = [output_carriers]
    carriers = set(output_carriers)
    output_carriers_by_tech = sets["set_output_carriers"]
    selected = {
        tech for tech in techs
        if carriers.intersection(output_carriers_by_tech[tech])
    }
    if not selected:
        logging.warning(
            "Profitability bias: no conversion technology produces any of the "
            f"configured output carriers {sorted(carriers)}; bias has no effect."
        )
    return selected

def get_min_coefficient_profitability_ratio(
    optimization_setup, output_carriers=None, coefficient=None
) -> float:
    """Minimal NPC-CAPEX-coefficient / profitability ratio over profitable pairs.

    Computes ``coefficient / profitability`` per (tech, node) and returns the
    smallest finite ratio over the profitable pairs (profitability > 0). Pairs
    with a ``NaN`` coefficient (PWA) or zero coefficient (pass-through techs) are
    excluded. Profitability is read from ``optimization_setup.profitability``
    (set after each solve); ``NaN`` is returned when it is absent.

    :param optimization_setup: the optimization setup of the current step
    :param output_carriers: optional carrier restriction
    :param coefficient: pre-computed ``get_npc_capex_coefficient`` result
    :return: minimal ratio, or ``NaN`` if no profitable pair remains
    """
    profitability = getattr(optimization_setup, "profitability", None)
    if profitability is None:
        # no profitability signal yet -> no ratio available
        print("\n--- Coefficient / profitability ratio: no profitability signal yet ---\n")
        return float("nan")

    if coefficient is None:
        coefficient = get_npc_capex_coefficient(optimization_setup)
    # exclude zero-CAPEX (pass-through) techs: a ratio of 0 would cancel the bias
    coefficient = coefficient[coefficient != 0]

    # restrict to technologies with a configured output carrier
    bias_techs = _get_bias_technologies(optimization_setup, output_carriers)
    profitability = profitability[
        profitability.index.get_level_values("set_conversion_technologies").isin(bias_techs)
    ]

    # restrict to profitable pairs only
    profitability = profitability[profitability > 0]

    ratio = coefficient.div(profitability)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ratio.name = "coefficient_profitability_ratio"

    if ratio.empty:
        print("\n--- Coefficient / profitability ratio: no profitable (tech, node) pairs ---\n")
        return float("nan")

    min_idx = ratio.idxmin()
    min_val = float(ratio.loc[min_idx])

    print(
        f"\n--- CAPEX-coefficient / profitability ratios (profitable pairs) ---\n"
        f"{'Tech / Node':<45} {'Ratio':>12}\n"
        + "-" * 58
    )
    for idx, val in ratio.sort_values().items():
        marker = "  <- min" if idx == min_idx else ""
        print(f"{str(idx):<45} {val:>12.4e}{marker}")
    print()

    return min_val




# objective modification
def apply_profitability_bias_objective(
    optimization_setup, weight: float = 0.5, output_carriers=None
) -> None:
    """Replace the model objective with total NPC penalized by a profitability bias.

    Sets ``J = Σ_y NPC_y - ρ_min * weight * Σ profitability * capacity_addition``
    for the decision-year capacity addition, reading profitability from
    ``optimization_setup.profitability`` (set after each solve). The objective is
    left unchanged when no profitability signal exists yet or ``ρ_min`` is not
    finite and positive. Technologies without a configured output carrier or
    without CAPEX receive a coefficient of 0.

    :param optimization_setup: the optimization setup holding the model
    :param weight: bias weight applied to the profitability term
    :param output_carriers: optional carrier restriction for the bias
    """
    profit = getattr(optimization_setup, "profitability", None)
    if profit is None:
        # no profitability signal yet -> keep the base total-cost objective
        return

    # zero the coefficient of technologies without a configured output carrier
    bias_techs = _get_bias_technologies(optimization_setup, output_carriers)
    profit = profit.where(
        profit.index.get_level_values("set_conversion_technologies").isin(bias_techs),
        0.0,
    )

    # zero the coefficient of (tech, node) pairs without CAPEX (pass-through converters)
    capex_coefficient = get_npc_capex_coefficient(optimization_setup)
    zero_capex = capex_coefficient.index[capex_coefficient == 0]
    if len(zero_capex) > 0:
        profit = profit.where(~profit.index.isin(zero_capex), 0.0)
        print(
            f"Profitability bias: excluded {len(zero_capex)} (tech, node) pairs "
            f"with zero CAPEX coefficient from the bias "
            f"({sorted(set(zero_capex.get_level_values(0)))})"
        )

    ratio_min = get_min_coefficient_profitability_ratio(
        optimization_setup, output_carriers=output_carriers,
        coefficient=capex_coefficient,
    )
    if not np.isfinite(ratio_min) or ratio_min <= 0:
        # no profitable pair with positive CAPEX coefficient -> keep the base objective
        logging.warning(
            "apply_profitability_bias_objective: minimal coefficient/profitability "
            f"ratio ({ratio_min}) is not finite and positive; keeping plain net "
            "present cost objective."
        )
        return

    model = optimization_setup.model
    base = model.variables["net_present_cost"].sum("set_time_steps_yearly")
    year_now = optimization_setup.energy_system.set_time_steps_yearly[0]
    cap_add = model.variables["capacity_addition"].sel(
        set_capacity_types="power", set_time_steps_yearly=year_now,
    )
    coef = (
        profit
        .rename_axis(index={
            "set_conversion_technologies": "set_technologies",
            "set_nodes": "set_location",
        })
        .to_xarray()
        .reindex(
            set_technologies=cap_add.coords["set_technologies"],
            set_location=cap_add.coords["set_location"],
            fill_value=0.0,
        )
    )
    coef_series = coef.to_series()
    print(
        f"\n--- Profitability bias coefficients "
        f"(ratio_min={ratio_min:.4e}, weight={weight}) ---\n"
        f"{'Tech / Node':<45} {'Coef':>14} {'Effective':>14}\n"
        + "-" * 75
    )
    for idx, val in coef_series.items():
        print(f"{str(idx):<45} {val:>14.4f} {ratio_min * weight * val:>14.4e}")
    print()

    objective = base - ratio_min * (weight * coef * cap_add).sum()

    optimization_setup.model.remove_objective()
    optimization_setup.model.add_objective(
        objective, sense=optimization_setup.analysis.sense
    )



#visualization and analysis output
def save_profitability_components(
    optimization_setup,
    output_dir=None,
    components_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append all profitability components of the current step to a per-run CSV.

    Calls ``calculate_profitability`` and writes the components, tagged with the
    decision year and carrier metadata, into the dataset's ``visualization``
    folder. The file name encodes the mode and, under rolling horizon, the bias
    and (if configured) subsidy tags.

    :param optimization_setup: the solved optimization setup of the current step
    :param output_dir: target directory; defaults to the dataset's
        ``visualization`` folder
    :param components_df: pre-computed ``calculate_profitability`` result
    :return: ``pd.DataFrame`` of the rows written for this step
    """
    year = int(max(optimization_setup.sets["set_time_steps_yearly"]))

    if components_df is None:
        components_df = calculate_profitability(optimization_setup)

    df = components_df.copy()
    df.index = df.index.set_names(["set_conversion_technologies", "set_nodes"])
    df = df.reset_index()

    sets = optimization_setup.sets

    def _carriers(mapping, tech):
        try:
            vals = mapping[tech]
        except (KeyError, TypeError):
            return ""
        return ", ".join(str(c) for c in vals)

    techs = df["set_conversion_technologies"]
    df.insert(
        1, "output_carrier",
        techs.map(lambda t: _carriers(sets["set_output_carriers"], t)),
    )
    df.insert(
        2, "input_carrier",
        techs.map(lambda t: _carriers(sets["set_input_carriers"], t)),
    )
    df.insert(
        3, "reference_carrier",
        techs.map(lambda t: _carriers(sets["set_reference_carriers"], t)),
    )
    df.insert(0, "decision_year", year)

    from pathlib import Path

    # file name: <timestamp>_profitability_<mode>[_<bias>][_<subsidy>].csv (bias/subsidy only under rolling horizon)
    timestamp = _get_run_timestamp()
    rolling = bool(getattr(optimization_setup.system, "use_rolling_horizon", False))
    mode = "rolling_horizon" if rolling else "perfect_foresight"

    try:
        from zen_garden.plugins.investment_decisions.plugin import config
    except Exception:
        config = {}

    name_parts = [timestamp, "profitability", mode]
    if rolling:
        if config.get("profitability_bias_enabled", False):
            name_parts.append(f"bias_{config.get('bias_weight', 0.5)}")
        else:
            name_parts.append("no_bias")
        subsidies = _get_subsidies(optimization_setup)
        if subsidies:
            # subsidy label: config ``subsidy_label`` or derived ``<tech>_<type>_<node>``
            label = config.get("subsidy_label")
            if not label:
                label = "__".join(
                    f"{s['technology']}_{s['type']}_{s['node']}" for s in subsidies
                )
            name_parts.append(str(label))
    file_name = "_".join(name_parts) + ".csv"

    # write into the visualization root (one level above the timestamp folder)
    if output_dir is None:
        out = Path(_get_output_dir(optimization_setup))
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / file_name
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)
    print(
        f"Saved profitability components ({len(df)} rows, decision year {year}) "
        f"to {csv_path}"
    )
    return df

def visualization(
    optimization_setup,
    output_dir: str | None = None,
) -> None:
    """Generate and save investment-decision plots, one pair per output carrier.

    Writes ``profitability_breakdown_<carrier>_<year>.png`` (revenue vs. stacked
    costs plus a net-profit marker) and ``profitability_net_<carrier>_<year>.png``
    (net-profit bar chart) into a timestamped subfolder shared across all steps
    of one run.

    :param optimization_setup: the solved optimization setup of the current step
    :param output_dir: output root; defaults to the dataset's ``visualization``
        folder
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path

    # per-run timestamped subfolder, identical for all steps of one program run
    if output_dir is None:
        out = _get_run_output_dir(optimization_setup)
    else:
        out = Path(output_dir) / _get_run_timestamp()
        out.mkdir(parents=True, exist_ok=True)

    year = max(optimization_setup.sets["set_time_steps_yearly"])

    # --- net profitability via central function ---
    pro = calculate_profitability(optimization_setup)["profitability"]

    idx = pro.index
    if idx.empty:
        print("visualization: no profitability data available, skipping plots.")
        return

    # --- individual cost/revenue components for the breakdown chart ---
    def _by_tech_node(series):
        return (
            series
            .groupby(level=["set_technologies", "set_nodes"])
            .sum()
            .rename_axis(index={"set_technologies": "set_conversion_technologies"})
            .reindex(idx, fill_value=0)
        )

    rev = _by_tech_node(calculate_revenue(optimization_setup))
    cap = get_capex(optimization_setup).reindex(idx, fill_value=0)
    fop = get_fixed_opex_discounted(optimization_setup).reindex(idx, fill_value=0)
    vop = get_variable_opex_discounted(optimization_setup).reindex(idx, fill_value=0)
    icc = _by_tech_node(calculate_input_carrier_cost(optimization_setup))
    tco2 = get_tech_co2_cost_discounted(optimization_setup).reindex(idx, fill_value=0)
    cco2 = _by_tech_node(get_carrier_co2_cost_discounted(optimization_setup))

    # one pair of charts per output carrier; all (tech, node) pairs whose
    # technology produces that carrier appear on the x-axis.
    output_carriers_by_tech = optimization_setup.sets["set_output_carriers"]
    technologies = list(dict.fromkeys(idx.get_level_values("set_conversion_technologies")))

    carrier_to_techs: dict[str, list[str]] = {}
    for tech in technologies:
        for carrier in output_carriers_by_tech[tech]:
            carrier_to_techs.setdefault(carrier, []).append(tech)

    if not carrier_to_techs:
        print("visualization: no output carriers found, skipping plots.")
        return

    for carrier, carrier_techs in carrier_to_techs.items():
        carrier_techs_set = set(carrier_techs)
        # rows of the shared index whose technology produces this carrier
        mask = idx.get_level_values("set_conversion_technologies").isin(carrier_techs_set)
        if not mask.any():
            continue

        # slice every component down to the current carrier's (tech, node) pairs
        rev_t = rev[mask]
        cap_t = cap[mask]
        fop_t = fop[mask]
        vop_t = vop[mask]
        icc_t = icc[mask]
        tco2_t = tco2[mask]
        cco2_t = cco2[mask]
        pro_t = pro[mask]

        # zero-revenue techs/nodes have no existing capacity; drop them and annotate
        nonzero_mask = rev_t.values != 0
        zero_pairs = [f"{t} / {n}" for (t, n), keep in zip(rev_t.index, nonzero_mask) if not keep]
        rev_t = rev_t[nonzero_mask]
        cap_t = cap_t[nonzero_mask]
        fop_t = fop_t[nonzero_mask]
        vop_t = vop_t[nonzero_mask]
        icc_t = icc_t[nonzero_mask]
        tco2_t = tco2_t[nonzero_mask]
        cco2_t = cco2_t[nonzero_mask]
        pro_t = pro_t[nonzero_mask]

        no_cap_text = (
            "No existing capacity: " + ", ".join(zero_pairs) if zero_pairs else ""
        )

        if pro_t.empty:
            # nothing left to plot, but still surface the skipped entries
            print(
                f"visualization: carrier '{carrier}' – all technologies without revenue. "
                f"{no_cap_text}"
            )
            continue

        labels = [f"{t}\n{n}" for t, n in pro_t.index]
        x = list(range(len(labels)))
        fig_width = max(7, len(labels) * 1.6)

        # -------------------------------------------------------------- #
        # Plot 1: Revenue vs. costs breakdown + net profit marker        #
        # -------------------------------------------------------------- #
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        w = 0.45

        ax.bar(x, rev_t.values, w, label="Revenue", color="#2ecc71", zorder=3)
        ax.bar(x, -cap_t.values, w, label="CAPEX", color="#e74c3c", zorder=3)
        ax.bar(x, -fop_t.values, w, label="Fixed OPEX", color="#e67e22",
               bottom=-cap_t.values, zorder=3)
        ax.bar(x, -vop_t.values, w, label="Var. OPEX", color="#f39c12",
               bottom=(-cap_t - fop_t).values, zorder=3)
        ax.bar(x, -icc_t.values, w, label="Input Cost", color="#9b59b6",
               bottom=(-cap_t - fop_t - vop_t).values, zorder=3)
        ax.bar(x, -tco2_t.values, w, label="Tech CO2", color="#7f8c8d",
               bottom=(-cap_t - fop_t - vop_t - icc_t).values, zorder=3)
        ax.bar(x, -cco2_t.values, w, label="Carrier CO2", color="#34495e",
               bottom=(-cap_t - fop_t - vop_t - icc_t - tco2_t).values, zorder=3)
        ax.plot(x, pro_t.values, "D", color="#2c3e50", markersize=9,
                label="Net Profit", zorder=4, clip_on=False)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Technology / Node")
        ax.set_ylabel("Discounted value [model money units]")
        ax.set_title(f"Investment Profitability Breakdown – output carrier '{carrier}' ({year})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if no_cap_text:
            fig.text(0.5, 0.01, no_cap_text, ha="center", va="bottom",
                     fontsize=8, style="italic", wrap=True)
            plt.tight_layout(rect=(0, 0.05, 1, 1))
        else:
            plt.tight_layout()
        p1 = out / f"profitability_breakdown_{carrier}_{year}.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"Saved: {p1}")

        # -------------------------------------------------------------- #
        # Plot 2: Net profitability bar chart                            #
        # -------------------------------------------------------------- #
        bar_colors = ["#27ae60" if v >= 0 else "#c0392b" for v in pro_t.values]
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        bars = ax.bar(x, pro_t.values, color=bar_colors, edgecolor="white", zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

        value_range = max(abs(pro_t.values)) if len(pro_t) else 1
        bar_offset = (value_range or 1) * 0.02
        for bar, val in zip(bars, pro_t.values):
            y = val + bar_offset if val >= 0 else val - bar_offset
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{val:,.0f}", ha="center", va=va, fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Technology / Node")
        ax.set_ylabel("Net Discounted Profit [model money units]")
        ax.set_title(f"Net Investment Profitability – output carrier '{carrier}' ({year})")
        handles = [
            mpatches.Patch(color="#27ae60", label="Profitable"),
            mpatches.Patch(color="#c0392b", label="Loss"),
        ]
        ax.legend(handles=handles, fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if no_cap_text:
            fig.text(0.5, 0.01, no_cap_text, ha="center", va="bottom",
                     fontsize=8, style="italic", wrap=True)
            plt.tight_layout(rect=(0, 0.05, 1, 1))
        else:
            plt.tight_layout()
        p2 = out / f"profitability_net_{carrier}_{year}.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"Saved: {p2}")

