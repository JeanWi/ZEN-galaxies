"""Per-(technology, node, year) CAPEX override via the dataset's input files.

Writes a temporary value into the CAPEX input CSV of a technology, so a ZEN-garden
run reads a modified CAPEX for exactly one (technology, node, year) cell. The
original CSV is restored byte-for-byte afterwards, including on error. This is
useful for validating a reported reduced cost: rerun the model with CAPEX set to
just above / just below the reported break-even and check whether the technology
builds.

Usage as a context manager:

    from zen_garden_plugins.reduced_cost.capex_override import capex_file_override

    overrides = [
        {"tech": "nuclear", "node": "NL", "year": 2050, "value": 4500.0},
        {"tech": "battery", "node": "CH", "year": 2050,
         "capacity_type": "power", "value": 70.0},
    ]
    with capex_file_override(dataset_path, overrides):
        run(config=..., dataset=dataset_path, folder_output=...)
    # the dataset is back in its original state here

``value`` is given in the technology's native CAPEX input units (the same units as
the existing CSV / the ``default_value`` in ``attributes.json``): Euro/kW for power
capacity, Euro/kWh for storage energy capacity, or whatever unit the reference
carrier implies. ``read_capex`` returns the value currently in effect for a
(technology, node, year), e.g. to compute a relative change.

A year-only or attributes-only parameter is expanded into a full node x year CSV in
which only the target cells are overridden; all other nodes/years keep their
original value. A CSV that already varies by node x year is edited in place.
"""

import json
import os
from contextlib import contextmanager

import pandas as pd

# suffix used for the temporary backup of an overridden CSV
BACKUP_SUFFIX = ".rc_capex_override_bak"


def _nodes(dataset):
    """Returns the model's nodes.

    Prefers ``system.json``'s ``set_nodes``, falls back to
    ``energy_system/set_nodes.csv``.

    Args:
        dataset (str): Path to the dataset folder.

    Returns:
        list[str]: Sorted node names.
    """
    try:
        with open(os.path.join(dataset, "system.json")) as f:
            nodes = json.load(f).get("set_nodes")
        if nodes:
            return sorted(str(n) for n in nodes)
    except Exception:
        pass
    try:
        path = os.path.join(dataset, "energy_system", "set_nodes.csv")
        return sorted(pd.read_csv(path)["node"].astype(str).unique())
    except Exception:
        return []


def _reference_years(dataset):
    """Returns the full set of planning years used by the dataset.

    Reads the year column of a conversion-technology CAPEX CSV if one exists,
    otherwise derives the year range from ``system.json``.

    Args:
        dataset (str): Path to the dataset folder.

    Returns:
        list[int]: Sorted planning years.
    """
    base = os.path.join(dataset, "set_technologies", "set_conversion_technologies")
    for dirpath, _, files in os.walk(base):
        for file in files:
            if file.startswith("capex_specific_conversion") and file.endswith(".csv"):
                try:
                    data = pd.read_csv(os.path.join(dirpath, file))
                    if "year" in data.columns:
                        return sorted(int(y) for y in data["year"].unique())
                except Exception:
                    pass
    try:
        with open(os.path.join(dataset, "system.json")) as f:
            system = json.load(f)
        reference_year = int(system.get("reference_year", 0))
        interval = int(system.get("interval_between_years", 1)) or 1
        n_years = int(system.get("optimized_years", 1))
        if n_years >= 1:
            return [reference_year + i * interval for i in range(n_years)]
    except Exception:
        pass
    return []


def _tech_folder(dataset, tech):
    """Returns the input folder of a technology, searching all technology classes."""
    base = os.path.join(dataset, "set_technologies")
    for dirpath, _, files in os.walk(base):
        if os.path.basename(dirpath) == tech and "attributes.json" in files:
            return dirpath
    return None


def _capex_meta(dataset, tech, capacity_type):
    """Returns ``(folder, parameter_name, csv_path)`` for a technology's CAPEX.

    ``parameter_name`` is also the column name used for the default value in
    ``attributes.json``.

    Args:
        dataset (str): Path to the dataset folder.
        tech (str): Technology name.
        capacity_type (str | None): ``"power"`` or ``"energy"`` for storage
            technologies, ``None`` otherwise.

    Returns:
        tuple[str, str, str] | None: Folder, parameter name, and CSV path, or
            ``None`` if the technology folder cannot be found.
    """
    folder = _tech_folder(dataset, tech)
    if folder is None:
        return None
    if "set_storage_technologies" in folder:
        name = (
            "capex_specific_storage_energy"
            if capacity_type == "energy"
            else "capex_specific_storage"
        )
    elif "set_transport_technologies" in folder:
        name = "capex_specific_transport"
    else:  # conversion technologies, including retrofitting technologies
        name = "capex_specific_conversion"
    return folder, name, os.path.join(folder, name + ".csv")


def _attribute_default(folder, parameter_name):
    """Returns the default value of a parameter from ``attributes.json``."""
    try:
        with open(os.path.join(folder, "attributes.json")) as f:
            attributes = json.load(f)
        return float(attributes.get(parameter_name, {}).get("default_value"))
    except Exception:
        return float("nan")


def _value_column(original, parameter_name):
    """Returns the CSV's value column name (parameter- or technology-named)."""
    if original is None:
        return parameter_name
    non_index_columns = [c for c in original.columns if c not in ("node", "year")]
    return non_index_columns[0] if non_index_columns else parameter_name


def _effective_value(original, value_column, default, node, year):
    """Returns the original effective value for (node, year) in input units."""
    if original is None:
        return default
    columns = original.columns
    subset = original
    if "year" in columns:
        subset = subset[subset["year"] == int(year)]
    if "node" in columns:
        subset = subset[subset["node"].astype(str) == str(node)]
    if len(subset):
        return float(subset[value_column].iloc[0])
    if "year" in columns:  # node not listed individually -> year-only value
        year_subset = original[original["year"] == int(year)]
        if len(year_subset):
            return float(year_subset[value_column].iloc[0])
    return default


def read_capex(dataset, tech, node, year, capacity_type=None):
    """Returns the CAPEX value currently in effect, in native input units.

    Args:
        dataset (str): Path to the dataset folder.
        tech (str): Technology name.
        node (str): Node name.
        year (int): Planning year.
        capacity_type (str | None): ``"power"`` or ``"energy"`` for storage
            technologies, ``None`` otherwise.

    Returns:
        float: The CAPEX value, or NaN if it cannot be determined.
    """
    meta = _capex_meta(dataset, tech, capacity_type)
    if meta is None:
        return float("nan")
    folder, parameter_name, path = meta
    original = pd.read_csv(path) if os.path.isfile(path) else None
    return _effective_value(
        original,
        _value_column(original, parameter_name),
        _attribute_default(folder, parameter_name),
        node,
        year,
    )


def restore_leftover_swaps(dataset):
    """Restores any backups left over from an interrupted ``capex_file_override``.

    Args:
        dataset (str): Path to the dataset folder.
    """
    base = os.path.join(dataset, "set_technologies")
    for dirpath, _, files in os.walk(base):
        for file in files:
            if file.endswith(BACKUP_SUFFIX):
                backup_path = os.path.join(dirpath, file)
                original_path = backup_path[: -len(BACKUP_SUFFIX)]
                if os.path.isfile(original_path):
                    os.remove(original_path)
                os.rename(backup_path, original_path)


def _build_grid(folder, parameter_name, path, cells, nodes, reference_years):
    """Returns a full node x year CSV with ``cells`` overriding the target entries.

    Args:
        folder (str): Technology input folder.
        parameter_name (str): CAPEX parameter name.
        path (str): Path to the existing CAPEX CSV, if any.
        cells (dict[tuple[str, int], float]): Mapping of (node, year) to the
            overridden value.
        nodes (list[str]): Model nodes.
        reference_years (list[int]): Model planning years.

    Returns:
        pandas.DataFrame: The new node x year CSV content.
    """
    original = pd.read_csv(path) if os.path.isfile(path) else None
    value_column = _value_column(original, parameter_name)
    default = _attribute_default(folder, parameter_name)
    years = (
        sorted(int(y) for y in original["year"].unique())
        if (original is not None and "year" in original.columns)
        else list(reference_years)
    )
    years = sorted(set(years) | {year for (_, year) in cells})
    node_list = sorted(set(map(str, nodes)) | {node for (node, _) in cells})
    rows = []
    for node in node_list:
        for year in years:
            value = cells.get((node, year))
            rows.append(
                {
                    "node": node,
                    "year": year,
                    value_column: (
                        value
                        if value is not None
                        else _effective_value(
                            original, value_column, default, node, year
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


@contextmanager
def capex_file_override(dataset, overrides, restore_leftover=True):
    """Context manager that overrides CAPEX input CSVs, then restores them.

    Args:
        dataset (str): Path to the dataset folder.
        overrides (list[dict]): Each entry is
            ``{"tech", "node", "year", "value"[, "capacity_type"]}``. Multiple
            overrides targeting the same CSV are merged into one write. An empty
            list is a no-op.
        restore_leftover (bool): If True (default), restore any backups left over
            from a previously interrupted call before applying the new overrides.

    Yields:
        None.

    Raises:
        RuntimeError: If the input folder of an overridden technology cannot be
            found.
    """
    if restore_leftover:
        restore_leftover_swaps(dataset)
    overrides = list(overrides or [])
    if not overrides:
        yield
        return

    nodes = _nodes(dataset)
    reference_years = _reference_years(dataset)
    # group overrides by the CSV they target
    groups = {}  # path -> [folder, parameter_name, {(node, year): value}]
    for override in overrides:
        meta = _capex_meta(dataset, override["tech"], override.get("capacity_type"))
        if meta is None:
            raise RuntimeError(
                f"capex_file_override: could not find the input folder for "
                f"'{override['tech']}'"
            )
        folder, parameter_name, path = meta
        group = groups.setdefault(path, [folder, parameter_name, {}])
        group[2][(str(override["node"]), int(override["year"]))] = float(
            override["value"]
        )

    swapped = []  # (path, backup_path)
    try:
        for path, (folder, parameter_name, cells) in groups.items():
            backup_path = path + BACKUP_SUFFIX
            new_data = _build_grid(
                folder, parameter_name, path, cells, nodes, reference_years
            )
            if os.path.isfile(backup_path):
                os.remove(backup_path)
            if os.path.isfile(path):
                os.rename(path, backup_path)  # back up the original byte-for-byte
            new_data.to_csv(path, index=False)
            swapped.append((path, backup_path))
        yield
    finally:
        for path, backup_path in swapped:
            if os.path.isfile(path):
                os.remove(path)
            if os.path.isfile(backup_path):
                os.rename(backup_path, path)
