"""Investment decisions plugin.

Adds an investor perspective to ZEN-garden's social-planner optimisation. Over a
rolling horizon, the plugin derives a per-technology, per-node profitability
indicator (the expected net present value per unit of installed capacity) from
each solved step and feeds it back as a scalable bias term into the next step's
objective, rewarding profitable capacity additions while keeping the original
cost optimisation intact.

The plugin hooks into two optimisation events:

* ``event_before_solve`` -- biases the freshly built objective with the
  previous step's profitability (see :func:`before_solve_event`).
* ``event_after_optimization`` -- evaluates the solved model, computes the
  profitability indicator, publishes it for the next step and exports it
  (see :func:`after_optimization_event`).

The ``config`` dictionary is filled by ``plugins.loader.register_plugins()`` and
controls the bias (``profitability_bias_enabled``, ``bias_weight``,
``bias_output_carriers``) and the optional ``subsidies``.
"""

import logging
import sys
import datetime
import contextlib

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugins.investment_decisions.investment_decisions import (
    apply_profitability_bias_objective, calculate_input_carrier_cost, calculate_profitability, extract_average_shadow_prices, calculate_revenue, get_capex, get_fixed_opex_discounted, get_flow_reference_carrier, get_min_coefficient_profitability_ratio, get_variable_opex_discounted, save_profitability_components, visualization
)

config = {}

# --- Global configuration ---------------------------------------------------
# Master switch for diagnostic output. When False, neither the text log file
# nor the visualization is produced. The profitability CSV is always written,
# regardless of this flag.
OUTPUT_ENABLED = True

# Hard-coded output location for the text log.
LOG_FILE_PATH = r"D:\Students\ssambale_jwiegner\ZEN-garden\investment_plugin_output.txt"


class _DualLogger:
    """Helper: tee stdout to both the terminal and an appended, timestamped log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a", encoding="utf-8")
        
        # write a timestamp to the file when the event fires
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"\n=== Event 'after_optimization' triggered at {now} ===\n")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

@contextlib.contextmanager
def _log_block(filename):
    """Helper: run the wrapped block with stdout teed to ``filename``.

    Honors the global ``OUTPUT_ENABLED`` switch: when output is disabled the
    block still runs (so computations and the CSV export proceed) but no log
    file is touched.
    """
    if not OUTPUT_ENABLED:
        yield
        return

    logger = _DualLogger(filename)
    original_stdout = sys.stdout
    sys.stdout = logger
    try:
        yield  # the body of the with-block runs here
    finally:
        sys.stdout = original_stdout  # restore standard output
        logger.log_file.close()       # close the file safely


@EventPublisher.register(Event.event_before_solve)
def before_solve_event(optimization_setup):
    """Inject the profitability bias into the objective before each solve.

    This is the point where the investor perspective enters the optimisation.
    The plugin extends ZEN-garden's total-cost objective by a scalable bias term
    that rewards capacity additions which the previous rolling-horizon step found
    profitable: ``min Σ_y NPC_y - ω · ρ_min · Σ profitability · capacity_addition``.
    The net present cost stays in the objective, so capacity is still only added
    where it benefits the system; the bias merely shifts the optimum towards the
    more profitable technologies, scaled by the user-defined weight ``ω`` and
    normalised by ``ρ_min`` so it stays on the order of magnitude of the cost.

    The bias is applied only when ``profitability_bias_enabled`` is set in the
    plugin config. On the first rolling-horizon step no profitability is
    available yet, so the objective collapses to the plain total net present
    cost. The heavy lifting is delegated to
    :func:`~zen_garden.plugins.investment_decisions.investment_decisions.apply_profitability_bias_objective`.
    """
    with _log_block(LOG_FILE_PATH):
        print("\n--- Applying profitability bias to objective ---")
        if not config.get("profitability_bias_enabled", False):
            return
        apply_profitability_bias_objective(
            optimization_setup,
            weight=config.get("bias_weight", 0.5),
            output_carriers=config.get("bias_output_carriers", []),
        )


@EventPublisher.register(Event.event_after_optimization)
def after_optimization_event(optimization_setup):
    """Derive the profitability indicator from the solved step and export it.

    This closes the rolling-horizon feedback loop. Once a step is solved, the
    plugin reads the flows, parameters and the nodal energy-balance shadow prices
    the solver already produced and condenses them into the profitability
    indicator: the net present value per unit of installed capacity, combining
    market revenue (valued at the shadow prices), CAPEX, fixed and variable OPEX,
    input-carrier cost, the two carbon-cost terms and any configured subsidies.

    The resulting profitability Series is published on the optimisation setup so
    that the next :func:`before_solve_event` can use it as the bias signal. The
    full per-(technology, node) component breakdown is always written to a
    per-run CSV for later analysis, while the text log and the visualization are
    produced only when the global ``OUTPUT_ENABLED`` switch is on.
    """
    with _log_block(LOG_FILE_PATH):

        print("\n--- Calculating and printing discounted fixed OPEX ---")
       
        extract_average_shadow_prices(optimization_setup)        

        # compute once; reuse the result for the bias signal and the CSV export.
        profitability_components = calculate_profitability(optimization_setup)

        # publish profitability Series for use as bias in the next period's
        # objective (consumed by before_solve_event -> apply_profitability_bias_objective).
        optimization_setup.profitability = profitability_components["profitability"]

        # persist all profitability components of this step to a per-run CSV so
        # they can be reused later (e.g. by run_and_visualize) across all steps.
        save_profitability_components(optimization_setup, components_df=profitability_components)

    if OUTPUT_ENABLED:
        visualization(optimization_setup)
