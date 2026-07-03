"""Heatmaps of the relative CAPEX reduction needed to build a technology.

Operates purely on the classified analysis CSVs written by
:mod:`zen_garden_plugins.reduced_cost._postprocess`
(``capacity_addition_analysis_{conversion,transport,storage}.csv``); it has no
dependency on ZEN-garden itself and can therefore also be re-run standalone (see
:mod:`zen_garden_plugins.reduced_cost.cli`) on an existing output folder.

Information encoded per cell:

* colour + top number: relative CAPEX reduction needed to build [%] (the relative
  reduced cost)
* bottom number (in parentheses): absolute reduced cost [input CAPEX unit, e.g.
  Euro/kW]
* row label: the technology's own CAPEX [input CAPEX unit]

Which reduced cost drives colour and numbers:

* conversion: the operational reduced cost (degeneracy-robust, dispatch-anchored)
* transport: the lifetime reduced cost (the operational reconstruction is not
  defined for transport)
* storage: the bundle reduced cost (proportional power + energy at the fixed
  duration)

Four visual states per cell:

* real reduced cost (case ``buildable_rc``, reliable) -> colour scale
  green (near) .. red (far)
* correctly built (case ``built``) -> white, annotated with the installed capacity
* at limit / not buildable -> black
* unreliable / other (e.g. breakeven) -> grey

Output (next to the CSVs)::

    <scenario>/heatmaps/<year>/<tech_type>/
        rc_heatmap_<tech_type>_<year>.png             the heatmap
        rc_heatmap_<tech_type>_<year>.csv              matrix (technology x node)
                                                        of the relative reduced cost [%]
        rc_heatmap_<tech_type>_<year>_categories.csv   matrix of the visual states
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

TECH_TYPES = ("conversion", "transport", "storage")
TECH_COLUMN = "set_technologies"
NODE_COLUMN = "set_location"
YEAR_INDEX_COLUMN = "set_time_steps_yearly"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
COLOR_NEAR = "#627313"
COLOR_MID = "#EDE7D3"
COLOR_FAR = "#B7352D"
COLOR_OTHER = "#D9D9D9"
COLOR_TEXT = "#1A1A1A"
COLOR_GRID = "#BFBFBF"
COLOR_CAPEX_LABEL = "#4D4D4D"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "CMU Serif",
            "Latin Modern Roman",
            "Times New Roman",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "axes.edgecolor": "#6F6F6F",
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

# green (near build) -> light sand (mid) -> red (far)
RC_COLORMAP = LinearSegmentedColormap.from_list(
    "reduced_cost", [(0.00, COLOR_NEAR), (0.50, COLOR_MID), (1.00, COLOR_FAR)]
)
RC_COLORMAP.set_over("#6F1410")  # above vmax -> dark red
RC_COLORMAP.set_bad(
    (0, 0, 0, 0)
)  # NaN transparent (the categorical overlay handles these)

CATEGORY_COLOR = {"built": "white", "blocked": "black", "other": COLOR_OTHER}

# per-technology-type source columns: relative metric (colour), absolute reduced
# cost, own CAPEX, and a label for the reduced-cost source
SPEC = {
    "conversion": dict(
        relative="ratio_reduction_operational",
        absolute="rc_capex_equivalent_operational_input_units",
        capex="capex_specific_input_units",
        label="operational reduced cost",
    ),
    "transport": dict(
        relative="ratio_reduction",
        absolute="rc_capex_equivalent_input_units",
        capex="capex_specific_input_units",
        label="lifetime reduced cost",
    ),
    "storage": {
        "proportional": dict(
            relative="ratio_reduction_proportional",
            absolute=None,  # = relative * C_bundle
            capex="C_bundle",
            label="bundle reduced cost (proportional)",
        ),
        "energy": dict(
            relative="ratio_energy_reduction",
            absolute="RC_energy",
            capex="capex_energy",
            label="energy reduced cost",
        ),
        "power": dict(
            relative="ratio_power_reduction",
            absolute="RC_power",
            capex="capex_power",
            label="power reduced cost",
        ),
    },
}


def categorize(case, reliable):
    """Maps a classification case and its reliability flag to a visual state."""
    if case == "buildable_rc" and reliable:
        return "rc"
    if case == "built":
        return "built"
    if case in ("at_limit", "not_buildable"):
        return "blocked"
    return "other"


def find_scenario_dirs(root):
    """Returns all folders under ``root`` that contain a classified analysis CSV."""
    hits = []
    for dirpath, _, files in os.walk(root):
        if any(f"capacity_addition_analysis_{t}.csv" in files for t in TECH_TYPES):
            hits.append(dirpath)
    return sorted(hits)


def year_map(scenario_dir):
    """Returns ``(reference_year, interval_between_years)`` from ``system.json``."""
    try:
        with open(os.path.join(scenario_dir, "system.json")) as f:
            system = json.load(f)
        return int(system.get("reference_year", 0)), int(
            system.get("interval_between_years", 1)
        )
    except Exception:
        return 0, 1


def build_matrices(df, reference_year, interval):
    """Builds per-year (relative, category, absolute, capex, built-value) matrices.

    Args:
        df (pandas.DataFrame): A classified analysis table with the helper columns
            ``__rel``, ``__abs``, ``__capex``, ``__buildval`` already set by
            :func:`run`.
        reference_year (int): The dataset's first planning year.
        interval (int): Years represented by one planning-year index step.

    Returns:
        dict[int, tuple]: Mapping of year to
            ``(relative, category, absolute, capex, built_value)`` matrices
            (technology x node/edge), restricted to rows/columns with at least one
            informative cell.
    """
    if "__rel" not in df.columns or "case" not in df.columns:
        return {}
    data = df.copy()
    reliable = (
        data["rc_reliable"].astype(str).str.lower().isin(["true", "1"])
        if "rc_reliable" in data.columns
        else pd.Series(False, index=data.index)
    )
    data["__cat"] = [
        categorize(c, r) for c, r in zip(data["case"], reliable, strict=True)
    ]
    # an "rc" cell without a relative value cannot be coloured -> demote it
    no_value = data["__cat"].eq("rc") & data["__rel"].isna()
    data.loc[no_value, "__cat"] = "other"
    data["__val"] = np.where(data["__cat"] == "rc", data["__rel"] * 100.0, np.nan)
    data["__absv"] = np.where(data["__cat"] == "rc", data["__abs"], np.nan)
    built_source = data["__buildval"] if "__buildval" in data.columns else np.nan
    data["__built"] = np.where(data["__cat"] == "built", built_source, np.nan)
    data["__year"] = (
        reference_year + data[YEAR_INDEX_COLUMN].astype(int) * interval
        if YEAR_INDEX_COLUMN in data.columns
        else reference_year
    )

    result = {}
    for year, group in data.groupby("__year"):
        category = group.pivot_table(
            index=TECH_COLUMN, columns=NODE_COLUMN, values="__cat", aggfunc="first"
        )
        relative = group.pivot_table(
            index=TECH_COLUMN,
            columns=NODE_COLUMN,
            values="__val",
            aggfunc="first",
            dropna=False,
        )
        absolute = group.pivot_table(
            index=TECH_COLUMN,
            columns=NODE_COLUMN,
            values="__absv",
            aggfunc="first",
            dropna=False,
        )
        capex = group.pivot_table(
            index=TECH_COLUMN,
            columns=NODE_COLUMN,
            values="__capex",
            aggfunc="first",
            dropna=False,
        )
        built = group.pivot_table(
            index=TECH_COLUMN,
            columns=NODE_COLUMN,
            values="__built",
            aggfunc="first",
            dropna=False,
        )
        category = category.sort_index(axis=0).sort_index(axis=1)
        relative = relative.reindex(index=category.index, columns=category.columns)
        absolute = absolute.reindex(index=category.index, columns=category.columns)
        capex = capex.reindex(index=category.index, columns=category.columns)
        built = built.reindex(index=category.index, columns=category.columns)
        informative = category.isin(["rc", "built", "blocked"])
        keep_rows, keep_columns = informative.any(axis=1), informative.any(axis=0)
        relative, category = (
            relative.loc[keep_rows, keep_columns],
            category.loc[keep_rows, keep_columns],
        )
        absolute, capex = (
            absolute.loc[keep_rows, keep_columns],
            capex.loc[keep_rows, keep_columns],
        )
        built = built.loc[keep_rows, keep_columns]
        if category.empty:
            continue
        result[int(year)] = (relative, category, absolute, capex, built)
    return result


def _capex_label(row_values):
    """Returns a technology's own CAPEX as a string (constant value or a range)."""
    values = pd.Series(row_values).dropna()
    if values.empty:
        return ""
    low, high = float(values.min()), float(values.max())
    if high - low <= max(1.0, 0.01 * high):  # ~constant across nodes/edges
        return f"{low:,.0f}".replace(",", " ")
    return f"{low:,.0f}–{high:,.0f}".replace(",", " ")


def render_heatmap(
    relative,
    category,
    absolute,
    capex,
    built,
    png_path,
    vmax,
    annotate,
    square=False,
    legend_y=-0.02,
):
    """Renders one heatmap and writes it to ``png_path``.

    Args:
        relative (pandas.DataFrame): Relative reduced cost matrix [%].
        category (pandas.DataFrame): Visual-state matrix.
        absolute (pandas.DataFrame): Absolute reduced-cost matrix.
        capex (pandas.DataFrame): Own-CAPEX matrix.
        built (pandas.DataFrame): Installed-capacity matrix for built cells.
        png_path (str): Output path for the rendered PNG.
        vmax (float): Upper end of the colour scale, in percent.
        annotate (bool): Whether to write numbers into the cells.
        square (bool): Force roughly square cells, useful for matrices with few
            rows (e.g. storage) that would otherwise look vertically stretched.
        legend_y (float): Figure-fraction y-position of the bottom legend.
    """
    data = relative.values.astype(float)
    absolute_values = absolute.values.astype(float)
    built_values = built.values.astype(float)
    categories = category.values
    n_rows, n_cols = data.shape
    fig_width = max(5.0, 0.60 * n_cols + 2.6)
    fig_height = 0.60 * n_rows + 1.7 if square else max(1.9, 0.46 * n_rows + 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("white")

    norm = Normalize(vmin=0.0, vmax=vmax, clip=False)
    ax.imshow(
        np.ma.masked_invalid(data),
        cmap=RC_COLORMAP,
        norm=norm,
        aspect=("equal" if square else "auto"),
        zorder=2,
    )
    im = ax.images[0]

    for i in range(n_rows):
        for j in range(n_cols):
            if np.isfinite(data[i, j]):
                continue
            color = CATEGORY_COLOR.get(categories[i, j], CATEGORY_COLOR["other"])
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="none",
                    zorder=1.5,
                )
            )

    labels = list(relative.index)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(relative.columns, rotation=0, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([])
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            xy=(0, i),
            xycoords=("axes fraction", "data"),
            xytext=(-8, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=COLOR_TEXT,
            annotation_clip=False,
            zorder=5,
        )
        capex_str = _capex_label(capex.iloc[i].values)
        if capex_str:
            ax.annotate(
                capex_str,
                xy=(0, i),
                xycoords=("axes fraction", "data"),
                xytext=(-8, -6),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=6.8,
                color=COLOR_CAPEX_LABEL,
                style="italic",
                annotation_clip=False,
                zorder=5,
            )
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Node / edge", fontsize=10)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=COLOR_GRID, linewidth=0.7, zorder=3)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#6F6F6F")
        spine.set_linewidth(0.8)

    if annotate:
        largest_dim = max(n_rows, n_cols)
        fontsize = 7.5 if largest_dim <= 14 else (6.5 if largest_dim <= 24 else 5.5)
        for i in range(n_rows):
            for j in range(n_cols):
                value = data[i, j]
                if not np.isfinite(value):
                    if categories[i, j] == "built":
                        built_value = built_values[i, j]
                        if np.isfinite(built_value) and built_value > 1e-6:
                            ax.text(
                                j,
                                i,
                                f"{built_value:.1f}",
                                ha="center",
                                va="center",
                                fontsize=fontsize - 1.5,
                                color=COLOR_TEXT,
                                zorder=4,
                            )
                    continue
                rgba = RC_COLORMAP(norm(min(value, vmax)))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else COLOR_TEXT
                ax.text(
                    j,
                    i - 0.13,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold",
                    color=text_color,
                    zorder=4,
                )
                absolute_value = absolute_values[i, j]
                if np.isfinite(absolute_value):
                    ax.text(
                        j,
                        i + 0.22,
                        f"({absolute_value:,.1f})".replace(",", " "),
                        ha="center",
                        va="center",
                        fontsize=fontsize - 1.2,
                        color=text_color,
                        zorder=4,
                    )

    colorbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.014, extend="max")
    colorbar.set_label(
        "Relative CAPEX reduction needed to build  [%]\n"
        "bracket: absolute reduced cost [EUR/kW]",
        fontsize=9,
    )
    colorbar.outline.set_edgecolor("#6F6F6F")
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(labelsize=8)

    legend = [
        Patch(
            facecolor="white",
            edgecolor="#6F6F6F",
            label="built (number = installed GW)",
        ),
        Patch(facecolor="black", label="at limit / not buildable"),
        Patch(
            facecolor=CATEGORY_COLOR["other"],
            edgecolor="#6F6F6F",
            label="RC unreliable / n.a.",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(root, vmax=150.0, storage_metric="proportional", annotate=True):
    """Renders reduced-cost heatmaps for every scenario folder found under ``root``.

    Args:
        root (str): Output folder to search recursively for classified analysis
            CSVs.
        vmax (float): Upper end of the colour scale, in percent.
        storage_metric (str): Which storage metric to render: ``"proportional"``
            (default), ``"energy"``, or ``"power"``.
        annotate (bool): Whether to write numbers into the cells.
    """
    scenario_dirs = find_scenario_dirs(root)
    if not scenario_dirs:
        return
    for scenario_dir in scenario_dirs:
        reference_year, interval = year_map(scenario_dir)
        for tech_type in TECH_TYPES:
            csv_path = os.path.join(
                scenario_dir, f"capacity_addition_analysis_{tech_type}.csv"
            )
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path)
            spec = (
                SPEC["storage"][storage_metric]
                if tech_type == "storage"
                else SPEC[tech_type]
            )

            if spec["relative"] not in df.columns:
                continue
            df["__rel"] = pd.to_numeric(df[spec["relative"]], errors="coerce")
            if (
                spec["absolute"] is None
            ):  # storage proportional: absolute = relative * bundle CAPEX
                df["__abs"] = df["__rel"] * pd.to_numeric(
                    df["C_bundle"], errors="coerce"
                )
            else:
                df["__abs"] = pd.to_numeric(df[spec["absolute"]], errors="coerce")
            df["__capex"] = pd.to_numeric(df[spec["capex"]], errors="coerce")
            build_value_column = "value_power" if tech_type == "storage" else "value"
            df["__buildval"] = pd.to_numeric(
                df.get(build_value_column, np.nan), errors="coerce"
            )

            matrices = build_matrices(df, reference_year, interval)
            if not matrices:
                continue
            for year, (relative, category, absolute, capex, built) in matrices.items():
                out_dir = os.path.join(scenario_dir, "heatmaps", str(year), tech_type)
                os.makedirs(out_dir, exist_ok=True)
                stem = f"rc_heatmap_{tech_type}_{year}"
                relative.round(4).to_csv(os.path.join(out_dir, stem + ".csv"))
                category.to_csv(os.path.join(out_dir, stem + "_categories.csv"))
                render_heatmap(
                    relative,
                    category,
                    absolute,
                    capex,
                    built,
                    os.path.join(out_dir, stem + ".png"),
                    vmax,
                    annotate,
                    square=(tech_type == "storage"),
                    legend_y=(-0.16 if tech_type == "storage" else -0.02),
                )
