import sys
from pathlib import Path
from typing import Annotated

import typer

from calib.cli import calibrate

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
        calibrate.main(input_directory, output_directory, chessboard_size)

    except Exception as e:  # noqa: BLE001
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    app()
