# Waves'N See - calib
![python versions](https://img.shields.io/badge/python-3.9&nbsp;3.12-blue)
![WNS Team](https://img.shields.io/badge/made&nbsp;by-WNS&nbsp;Team-blue)

Ce projet permet d'effectuer la calibration de caméra, par calcul des paramètres intrinsèques.

## Installation
```
uv venv --python 3.12
uv sync

or with --all-groups to also install the doc deps:
uv sync --all-groups
```

## Docs
```
cd docs
mkdocs serve
```

## Installation as a uv tool
If user wants to install calib as a uv tool:
```
uv tool install "calib @ git+ssh://git@github.com/wavesnsee/calib"
```

## Usage
Prior to perform camera calibration, make snapshots of the chessboard in different configurations.

**To run camera calibration:**
```
cd src/calib
python cli/app.py inputdir_snapshots outputdir_calibration  chessboard size_x chessboard size_y
```

**To run camera calibration with calib tool:**
```
calib inputdir_snapshots outputdir_calibration  chessboard size_x chessboard size_y
```