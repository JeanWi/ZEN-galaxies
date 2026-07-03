:orphan:

.. _available_plugins.reduced_cost:

Reduced cost
------------

The ``reduced_cost`` plugin reconstructs, for every unbuilt (or capacity-limited)
conversion, transport, or storage technology, the CAPEX-equivalent reduced cost of
``capacity_addition``: the amount by which CAPEX would have to fall for the
technology to become optimal to build.

The reconstruction is based on the model's LP duals, which are only uniquely
determined where the constraint they come from is not degenerate. The most common
source of that degeneracy is an unbuilt technology whose technology-lifetime
balance reads ``0 = 0``. The plugin offers two optional constraint perturbations
that resolve this without changing the optimal primal solution, and a parameter
override that fixes a storage technology's duration so its power and energy
components share one well-defined reduced cost.

Three events are used, at three different points of the run:

- ``after_construct_params``: optionally fixes the duration (energy-to-power
  ratio) of selected storage technologies.
- ``after_model_construction``: optionally perturbs the technology-lifetime and
  storage-power constraints.
- ``after_postprocessing``: reconstructs and classifies the reduced cost of every
  capacity-addition decision, and optionally renders heatmaps and a diagnostic
  dual decomposition.

Configuration
^^^^^^^^^^^^^

.. code-block:: json

    {
        "plugins": {
            "reduced_cost": {
                "tight_e2p": {"battery": 16, "pumped_hydro": 22},
                "lifetime_rhs_perturbation": 1e-4,
                "storage_power_perturbation": 1e-4,
                "generate_classification": true,
                "generate_heatmaps": true,
                "heatmap_vmax": 150,
                "storage_metric": "proportional",
                "dual_dump": false,
                "dual_dump_targets": []
            }
        }
    }

- ``tight_e2p``: mapping of storage technology name to a fixed duration in hours.
  Empty (default) disables the override.
- ``lifetime_rhs_perturbation`` / ``storage_power_perturbation``: perturbation
  sizes in capacity units (e.g. GW). 0 (default) disables the corresponding
  perturbation.
- ``generate_classification``: whether to split the analysis into classified,
  per-technology-type CSVs (default ``true``).
- ``generate_heatmaps``: whether to render heatmaps from the classified CSVs
  (default ``false``).
- ``heatmap_vmax`` / ``storage_metric``: heatmap rendering options.
- ``dual_dump`` / ``dual_dump_targets``: diagnostic per-constraint dual
  decomposition for the given ``{"tech", "node"[, "capacity_type"]}`` targets
  (default off).

Output
^^^^^^

Written next to the standard ZEN-garden output for each scenario:

- ``capacity_addition_analysis.csv``: the full reduced-cost table.
- ``capacity_addition_analysis_{conversion,transport,storage}.csv``: the
  classified analysis, one row per technology decision, with a ``case`` column
  (``built``, ``buildable_rc``, ``at_limit``, ``not_buildable``,
  ``blocked_profitable``, or ``breakeven_unreliable``) and a ``rc_reliable`` flag.
- ``heatmaps/<year>/<technology_type>/``: rendered heatmaps, if
  ``generate_heatmaps`` is enabled.
- ``rc_dual_decomposition.csv``: the diagnostic dual decomposition, if
  ``dual_dump`` is enabled.

Standalone utilities
^^^^^^^^^^^^^^^^^^^^

Two utilities are shipped with the plugin but are not tied to any event:

- ``zen_garden_plugins.reduced_cost.capex_override.capex_file_override``: a
  context manager that temporarily overrides a technology's CAPEX for a single
  (technology, node, year) in the dataset's input files, restoring the original
  files afterwards. Useful for validating a reported reduced cost by rerunning the
  model with CAPEX just above / below the reported break-even.
- ``python -m zen_garden_plugins.reduced_cost.cli heatmaps <OUTPUT_DIR>``:
  regenerates heatmaps from the classified CSVs of an existing output folder,
  without re-solving the model.

Module documentation
^^^^^^^^^^^^^^^^^^^^

.. automodule:: zen_garden_plugins.reduced_cost.plugin
   :members:
   :undoc-members:
