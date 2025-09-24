#!/usr/bin/env python
"""Download Camera blob image in a specific directory

Description
-----------
Use calib to compute camera's intrinsic parameters.
"""

# usage:
# python input_dir_snapshots output_dir_calibration 6 4

from pathlib import Path
from typing import Annotated
import sys

import calibrate
import typer


app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    input_directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            help="Input directory containing calibration screenshots",
        ),
    ],
    output_directory: Annotated[
        Path,
        typer.Argument(
            help="Output directory",
        ),
    ],
    chessboard_size: Annotated[
        tuple[int, int],
        typer.Argument(help="Chessboard size"),
    ],
):
    if not input_directory or not isinstance(input_directory, Path):
        raise typer.Exit("Please specify the input directory")

    if not output_directory or not isinstance(output_directory, Path):
        raise typer.Exit("Please specify the output directory")

    if not input_directory.exists():
        raise typer.Exit("Input directory does not exist")

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    try:
        # Run calibration
        # intrinsic = intrinsic_parameters(input_directory, chessboard_size)
        calibrate.main(input_directory, output_directory, chessboard_size)

    except Exception as e:  # noqa: BLE001
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
