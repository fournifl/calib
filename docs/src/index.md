# Waves'N See - calib
![python versions](https://img.shields.io/badge/python-3.9&nbsp;3.12-blue)
![WNS Team](https://img.shields.io/badge/made&nbsp;by-WNS&nbsp;Team-blue)

Calib stands for the camera calibration procedure.
Distorsion coefficients, focal lengths, and center of optic are computed.

## Installation
```
uv venv --python 3.12
uv sync

or with --all-groups to install the doc deps too:
uv sync --all-groups
```

## Usage
Prior to perform camera calibration, make snapshots of the chessboard in different configurations:

![La team en action](img/animation_chessboard_snapshots.gif)

In this case, chessboard size is (6,4).

To run camera calibration:
```
cd src/calib
python cli/app.py input_dir_snapshots output_dir_calibration  chessboard size_x chessboard size_y
```