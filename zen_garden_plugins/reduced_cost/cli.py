"""Command-line entry point for regenerating reduced-cost heatmaps.

The classification CSVs (``capacity_addition_analysis_{conversion,transport,
storage}.csv``) are written once, at run time, by
:mod:`zen_garden_plugins.reduced_cost._postprocess`. Heatmap rendering
(:mod:`zen_garden_plugins.reduced_cost._heatmaps`) is a pure function of those
already-written CSVs, so it can be re-run on an existing output folder without
re-solving the model, e.g. after changing the colour scale or the storage metric.

Usage::

    python -m zen_garden_plugins.reduced_cost.cli heatmaps <OUTPUT_DIR> [options]
"""

import argparse

from . import _heatmaps


def _heatmaps_command(args):
    _heatmaps.run(
        args.output_dir,
        vmax=args.vmax,
        storage_metric=args.storage_metric,
        annotate=not args.no_annotate,
    )


def main():
    """Parses command-line arguments and dispatches to the requested subcommand."""
    parser = argparse.ArgumentParser(
        description="Reduced-cost analysis utilities operating on an existing "
        "ZEN-garden output folder."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    heatmaps_parser = subparsers.add_parser(
        "heatmaps", help="Render reduced-cost heatmaps from classified analysis CSVs."
    )
    heatmaps_parser.add_argument(
        "output_dir", help="Output folder to search recursively for classified CSVs."
    )
    heatmaps_parser.add_argument(
        "--vmax",
        type=float,
        default=150.0,
        help="Upper end of the colour scale in %% (default: 150).",
    )
    heatmaps_parser.add_argument(
        "--storage-metric",
        choices=("proportional", "energy", "power"),
        default="proportional",
        help="Storage metric to render (default: proportional).",
    )
    heatmaps_parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Do not write numbers into the cells.",
    )
    heatmaps_parser.set_defaults(func=_heatmaps_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
