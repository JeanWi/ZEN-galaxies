"""Tests for the pure classification logic in ``_classify.classify``."""

import numpy as np
import pytest

from zen_garden_plugins.reduced_cost._classify import classify


def test_built_when_value_exceeds_tolerance():
    assert (
        classify(
            value=1.0,
            capacity=1.0,
            ceiling=10.0,
            reduced_cost=0.0,
            built_tolerance=1e-6,
        )
        == "built"
    )


def test_not_buildable_when_ceiling_is_zero():
    assert (
        classify(
            value=0.0, capacity=0.0, ceiling=0.0, reduced_cost=5.0, built_tolerance=1e-6
        )
        == "not_buildable"
    )


def test_at_limit_when_capacity_sits_at_ceiling():
    assert (
        classify(
            value=0.0,
            capacity=10.0,
            ceiling=10.0,
            reduced_cost=5.0,
            built_tolerance=1e-6,
        )
        == "at_limit"
    )


def test_buildable_rc_when_reduced_cost_is_positive_with_headroom():
    assert (
        classify(
            value=0.0,
            capacity=0.0,
            ceiling=10.0,
            reduced_cost=5.0,
            built_tolerance=1e-6,
        )
        == "buildable_rc"
    )


def test_blocked_profitable_when_reduced_cost_is_negative():
    assert (
        classify(
            value=0.0,
            capacity=0.0,
            ceiling=10.0,
            reduced_cost=-5.0,
            built_tolerance=1e-6,
        )
        == "blocked_profitable"
    )


def test_breakeven_unreliable_when_reduced_cost_is_near_zero():
    assert (
        classify(
            value=0.0,
            capacity=0.0,
            ceiling=10.0,
            reduced_cost=0.0,
            built_tolerance=1e-6,
        )
        == "breakeven_unreliable"
    )


def test_unbounded_ceiling_never_triggers_not_buildable_or_at_limit():
    # an infinite ceiling can never be "reached" and never equals a zero ceiling
    assert (
        classify(
            value=0.0,
            capacity=1000.0,
            ceiling=np.inf,
            reduced_cost=5.0,
            built_tolerance=1e-6,
        )
        == "buildable_rc"
    )


@pytest.mark.parametrize("nan_value", [np.nan])
def test_nan_inputs_are_treated_as_zero(nan_value):
    # a NaN reduced cost (e.g. an unreconstructable case) must not crash and must
    # fall through to the neutral breakeven_unreliable case
    assert (
        classify(
            value=nan_value,
            capacity=nan_value,
            ceiling=10.0,
            reduced_cost=nan_value,
            built_tolerance=1e-6,
        )
        == "breakeven_unreliable"
    )
