"""Tests that the plugin exposes its config and registers its handlers correctly."""

import importlib
import sys
import types


def _load_plugin_with_fake_events(monkeypatch):
    """Imports the plugin module against a fake ``zen_garden`` events module.

    Mirrors the ZEN-garden plugin loader: registering a handler records
    ``(event, function)`` in ``calls`` instead of touching the real event system.
    """
    calls = []

    class Event:
        after_construct_params = object()
        after_model_construction = object()
        after_postprocessing = object()

    class EventPublisher:
        @staticmethod
        def register(event):
            def decorator(func):
                calls.append((event, func))
                return func

            return decorator

    zen_garden = types.ModuleType("zen_garden")
    plugin_system = types.ModuleType("zen_garden.plugin_system")
    events = types.ModuleType("zen_garden.plugin_system.events")
    events.Event = Event
    events.EventPublisher = EventPublisher
    zen_garden.plugin_system = plugin_system
    plugin_system.events = events

    monkeypatch.setitem(sys.modules, "zen_garden", zen_garden)
    monkeypatch.setitem(sys.modules, "zen_garden.plugin_system", plugin_system)
    monkeypatch.setitem(sys.modules, "zen_garden.plugin_system.events", events)
    monkeypatch.delitem(
        sys.modules, "zen_garden_plugins.reduced_cost.plugin", raising=False
    )

    module = importlib.import_module("zen_garden_plugins.reduced_cost.plugin")
    return module, Event, calls


def test_plugin_exposes_config_dict(monkeypatch):
    """The plugin exposes a config dict with all documented keys."""
    module, _event, _calls = _load_plugin_with_fake_events(monkeypatch)

    assert isinstance(module.config, dict)
    for key in (
        "tight_e2p",
        "lifetime_rhs_perturbation",
        "storage_power_perturbation",
        "generate_classification",
        "generate_heatmaps",
        "heatmap_vmax",
        "dual_dump",
        "dual_dump_targets",
    ):
        assert key in module.config


def test_plugin_registers_exactly_three_handlers(monkeypatch):
    """The plugin registers exactly one handler per event, on the right event."""
    module, event, calls = _load_plugin_with_fake_events(monkeypatch)

    assert len(calls) == 3
    registered = {registered_event: func for registered_event, func in calls}
    assert (
        registered[event.after_construct_params]
        is module.apply_tight_energy_to_power_ratio
    )
    assert (
        registered[event.after_model_construction]
        is module.perturb_degenerate_constraints
    )
    assert registered[event.after_postprocessing] is module.write_reduced_cost_analysis
