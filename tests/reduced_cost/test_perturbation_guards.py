"""Tests for the feasibility guards in the RHS/bound perturbations.

Builds small, hand-crafted stand-ins for ``OptimizationSetup`` out of plain
``xarray`` objects, so the guard logic can be exercised without solving a model.
"""

import types

import numpy as np
import pytest
import xarray as xr

from zen_garden_plugins.reduced_cost._perturbation import (
    perturb_lifetime_rhs,
    perturb_storage_power_addition,
)


class _FakeConstraint:
    def __init__(self, rhs):
        self.rhs = rhs


class _FakeVariable:
    def __init__(self, upper=None, lower=None):
        self.upper = upper
        self.lower = lower


def _make_lifetime_setup():
    """A greenfield technology with head-room, one without, and a brownfield one."""
    techs = ["greenfield_with_room", "greenfield_no_room", "brownfield"]
    coords = {"set_technologies": techs}
    rhs = xr.DataArray([0.0, 0.0, 5.0], coords=coords, dims=["set_technologies"])
    capacity_upper = xr.DataArray(
        [10.0, 0.0, 10.0], coords=coords, dims=["set_technologies"]
    )
    capacity_limit = xr.DataArray(
        [np.inf, np.inf, np.inf], coords=coords, dims=["set_technologies"]
    )
    model = types.SimpleNamespace(
        constraints={"constraint_technology_lifetime": _FakeConstraint(rhs)},
        variables={"capacity": _FakeVariable(upper=capacity_upper)},
    )
    parameters = types.SimpleNamespace(capacity_limit=capacity_limit)
    return types.SimpleNamespace(
        model=model, parameters=parameters, sets={"set_storage_technologies": []}
    )


def test_lifetime_rhs_perturbation_skips_zero_ceiling_and_brownfield():
    setup = _make_lifetime_setup()

    perturb_lifetime_rhs(setup, delta=1.0)

    rhs = setup.model.constraints["constraint_technology_lifetime"].rhs
    # has head-room -> perturbed
    assert rhs.sel(set_technologies="greenfield_with_room").item() == pytest.approx(1.0)
    # ceiling is 0 -> perturbing would make the model infeasible -> skipped
    assert rhs.sel(set_technologies="greenfield_no_room").item() == pytest.approx(0.0)
    # not degenerate (existing capacity != 0) -> skipped
    assert rhs.sel(set_technologies="brownfield").item() == pytest.approx(5.0)


def test_lifetime_rhs_perturbation_is_noop_for_falsy_delta():
    setup = _make_lifetime_setup()
    original = setup.model.constraints[
        "constraint_technology_lifetime"
    ].rhs.values.copy()

    perturb_lifetime_rhs(setup, delta=0.0)

    updated = setup.model.constraints["constraint_technology_lifetime"].rhs.values
    np.testing.assert_array_equal(updated, original)


def _make_storage_setup():
    """A greenfield storage technology, an already-built one, and one without a
    finite duration (energy_to_power_ratio_min)."""
    techs = ["greenfield_battery", "existing_battery", "no_duration_battery"]
    capacity_types = ["power", "energy"]
    coords = {"set_technologies": techs, "set_capacity_types": capacity_types}
    lower = xr.DataArray(
        np.zeros((3, 2)), coords=coords, dims=["set_technologies", "set_capacity_types"]
    )
    capacity_upper = xr.DataArray(
        np.full((3, 2), 10.0),
        coords=coords,
        dims=["set_technologies", "set_capacity_types"],
    )
    capacity_limit = xr.DataArray(
        np.full((3, 2), np.inf),
        coords=coords,
        dims=["set_technologies", "set_capacity_types"],
    )
    existing = xr.DataArray(
        [[0.0, 0.0], [5.0, 5.0], [0.0, 0.0]],
        coords=coords,
        dims=["set_technologies", "set_capacity_types"],
    )
    e2p_min = xr.DataArray(
        [4.0, 4.0, 0.0],
        coords={"set_storage_technologies": techs},
        dims=["set_storage_technologies"],
    )
    model = types.SimpleNamespace(
        constraints={"constraint_technology_lifetime": _FakeConstraint(existing)},
        variables={
            "capacity": _FakeVariable(upper=capacity_upper),
            "capacity_addition": _FakeVariable(lower=lower),
        },
    )
    parameters = types.SimpleNamespace(
        capacity_limit=capacity_limit, energy_to_power_ratio_min=e2p_min
    )
    return types.SimpleNamespace(
        model=model, parameters=parameters, sets={"set_storage_technologies": techs}
    )


def test_storage_power_perturbation_only_applies_to_greenfield_with_duration():
    setup = _make_storage_setup()

    perturb_storage_power_addition(setup, magnitude=1e-4)

    lower = setup.model.variables["capacity_addition"].lower
    # greenfield with a finite duration -> perturbed
    assert lower.sel(
        set_technologies="greenfield_battery", set_capacity_types="power"
    ).item() == pytest.approx(1e-4)
    # already built (existing != 0) -> skipped
    assert lower.sel(
        set_technologies="existing_battery", set_capacity_types="power"
    ).item() == pytest.approx(0.0)
    # no finite duration -> a probe would not force a matching energy addition ->
    # skipped
    assert lower.sel(
        set_technologies="no_duration_battery", set_capacity_types="power"
    ).item() == pytest.approx(0.0)
    # the energy component itself is never perturbed directly, only the power one
    assert lower.sel(
        set_technologies="greenfield_battery", set_capacity_types="energy"
    ).item() == pytest.approx(0.0)


def test_storage_power_perturbation_is_noop_for_falsy_magnitude():
    setup = _make_storage_setup()
    original = setup.model.variables["capacity_addition"].lower.values.copy()

    perturb_storage_power_addition(setup, magnitude=0.0)

    updated = setup.model.variables["capacity_addition"].lower.values
    np.testing.assert_array_equal(updated, original)
