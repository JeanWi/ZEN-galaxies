"""Tests for the tight energy-to-power-ratio parameter override."""

import types

import pytest
import xarray as xr

from zen_garden_plugins.reduced_cost._param_overrides import (
    apply_tight_energy_to_power_ratio,
)


def _make_setup():
    techs = ["battery", "pumped_hydro"]
    coords = {"set_storage_technologies": techs}
    e2p_min = xr.DataArray([1.0, 1.0], coords=coords, dims=["set_storage_technologies"])
    e2p_max = xr.DataArray(
        [float("inf"), float("inf")], coords=coords, dims=["set_storage_technologies"]
    )
    parameters = types.SimpleNamespace(
        energy_to_power_ratio_min=e2p_min, energy_to_power_ratio_max=e2p_max
    )
    return types.SimpleNamespace(parameters=parameters)


def test_fixes_min_and_max_to_the_given_duration():
    setup = _make_setup()

    apply_tight_energy_to_power_ratio(setup, {"battery": 16.0})

    assert setup.parameters.energy_to_power_ratio_min.sel(
        set_storage_technologies="battery"
    ).item() == pytest.approx(16.0)
    assert setup.parameters.energy_to_power_ratio_max.sel(
        set_storage_technologies="battery"
    ).item() == pytest.approx(16.0)
    # untouched technology keeps its original values
    assert setup.parameters.energy_to_power_ratio_min.sel(
        set_storage_technologies="pumped_hydro"
    ).item() == pytest.approx(1.0)


def test_empty_override_is_noop():
    setup = _make_setup()
    original_min = setup.parameters.energy_to_power_ratio_min.values.copy()

    apply_tight_energy_to_power_ratio(setup, {})

    assert (setup.parameters.energy_to_power_ratio_min.values == original_min).all()


def test_unknown_technology_does_not_raise():
    setup = _make_setup()

    # must log a warning and continue, not raise
    apply_tight_energy_to_power_ratio(setup, {"unknown_tech": 5.0})
