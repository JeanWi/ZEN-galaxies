"""Parameter overrides applied after all model parameters have been constructed.

Registered on ``Event.after_construct_params``, which fires once, after every
element class has built its parameters but before any variable or constraint is
constructed. This is the latest point at which a parameter value can still be
changed and have that change flow into the cost terms and constraint bounds built
from it.
"""

import logging


def apply_tight_energy_to_power_ratio(optimization_setup, tight_e2p):
    """Fixes the energy-to-power ratio of storage technologies to a given duration.

    Sets ``energy_to_power_ratio_min == energy_to_power_ratio_max == duration`` for
    every technology named in ``tight_e2p``, so the technology is forced to a single
    storage duration instead of choosing it freely. This also removes the degree of
    freedom that would otherwise decouple the power and energy components of a
    storage technology's reduced cost.

    Args:
        optimization_setup: The OptimizationSetup whose parameters are modified.
        tight_e2p (dict[str, float]): Mapping of storage technology name to the
            fixed duration in hours. An empty dict is a no-op.
    """
    if not tight_e2p:
        return
    parameters = optimization_setup.parameters
    e2p_min = getattr(parameters, "energy_to_power_ratio_min", None)
    e2p_max = getattr(parameters, "energy_to_power_ratio_max", None)
    if e2p_min is None or e2p_max is None:
        logging.warning(
            "reduced_cost plugin: energy_to_power_ratio parameters not found — "
            "tight_e2p override skipped"
        )
        return

    dimension = next(
        (d for d in e2p_min.dims if "technolog" in d), "set_storage_technologies"
    )
    for tech, duration in dict(tight_e2p).items():
        try:
            duration = float(duration)
            e2p_min.loc[{dimension: tech}] = duration
            e2p_max.loc[{dimension: tech}] = duration
            logging.info(
                f"reduced_cost plugin: fixed the duration of '{tech}' to "
                f"{duration} h (energy_to_power_ratio_min = max = {duration})"
            )
        except Exception as error:
            logging.warning(
                f"reduced_cost plugin: could not fix the duration of '{tech}': "
                f"{error}"
            )
