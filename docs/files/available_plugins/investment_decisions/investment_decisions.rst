:orphan:

.. _available_plugins.investment_decisions:

Investment decisions plugin
----------------------------

The ``investment_decisions`` module adds an investor perspective to ZEN-garden's
social-planner optimisation. Over a rolling horizon it extends the total-cost
objective by a scalable bias term that rewards profitable capacity additions:

.. math::

   \min \sum_{y' \in \mathcal{Y}} \mathrm{NPC}_{y'}
   \;-\; \omega\,\rho_{\min} \sum_{i \in \mathcal{I}} \sum_{n \in \mathcal{N}}
   \Pi_{i,n,y-1}\, \Delta S_{i,n,y}.

The profitability :math:`\Pi_{i,n,y-1}` is the net present value per unit of
capacity, computed from the previous step's flows, shadow prices and parameters.
The weight :math:`\omega` scales the bias (:math:`\omega = 0` recovers the pure
cost optimisation) and the normalisation factor :math:`\rho_{\min}` puts the NPV
onto the order of magnitude of the net present cost.

Configuration
^^^^^^^^^^^^^

The plugin is controlled via the ``config`` dictionary:

* ``profitability_bias_enabled`` (*bool*) -- master switch; when ``False`` no
  bias is added and the objective stays the plain net present cost.
* ``bias_weight`` (*float*) -- the weight :math:`\omega` scaling the bias.
* ``bias_output_carriers`` (*list[str]*) -- optional restriction of the bias to
  technologies producing at least one of these output carriers.
* ``subsidies`` (*list[dict]*) -- optional per-technology, per-node cash flows
  entering the profitability. Each entry has ``technology``, ``node``, ``type``
  (``capex``, ``fixed_opex``, ``variable_opex`` or ``remuneration``) and
  ``amount``; ``remuneration`` additionally requires an output ``carrier``.

.. literalinclude:: ../../../../zen_garden_plugins/investment_decisions/plugin.py
   :language: python

Module documentation
^^^^^^^^^^^^^^^^^^^^

.. automodule:: zen_garden_plugins.investment_decisions.plugin
   :members:
   :undoc-members:
