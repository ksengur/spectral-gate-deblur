"""Locations of data and outputs. Override with environment variables; defaults are relative to the repo."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATE_OUT = Path(os.environ.get("GATE_OUT", ROOT / "outputs"))                                # gate / calibration CSVs
DIV2K_DIR = Path(os.environ.get("DIV2K_DIR", ROOT / "data" / "DIV2K"))                       # DIV2K_train_HR, DIV2K_valid_HR
LEVIN_DIR = Path(os.environ.get("LEVIN_DIR", ROOT / "data" / "levin09" / "Levin09blurdata")) # imXX_flitYY.mat
KAIR_DIR = Path(os.environ.get("KAIR_DIR", ROOT / "third_party" / "KAIR"))                   # github.com/cszn/KAIR
PRETRAINED_DIR = Path(os.environ.get("PRETRAINED_DIR", ROOT / "pretrained"))                 # usrnet.pth, drunet_color.pth
