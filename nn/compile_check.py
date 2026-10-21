"""Quantize GWaveNet2DModified with random data and export an xmodel for the compiler.

This checks whether the model COMPILES for the KV260 DPU. It says nothing about
accuracy: calibration uses random inputs. For real INT8 accuracy, calibrate with
a few hundred real samples instead (see `calibration_batches`).

Usage (inside the container, from the folder with model_inspector.py):
    python compile_check.py calib
    python compile_check.py test
"""

import sys

import torch
from pytorch_nndct.apis import torch_quantizer

# ---------------------------------------------------------------------------
# 1. Build the model exactly as in model_inspector.py.
#    Copy the import and constructor lines from there and replace this block.
# ---------------------------------------------------------------------------
from models.GWaveNet2DModified import (
    GWaveNet2DModified,
)  # adjust to match model_inspector.py
from models.globals import FILTERS, STRIDE, DILATION, DETECTORS

model = GWaveNet2DModified(
    in_channels=len(DETECTORS),
    out_channels=FILTERS,
    kernel_size=2,
    dilation=DILATION,
    stride=STRIDE,
)  # same constructor arguments as in model_inspector.py
# Optional: load trained weights (not needed for a compile check)
# model.load_state_dict(torch.load("path/to/weights.pt", map_location="cpu"))

# Input shape: (1, 3, 155648) while the model still has the view() at the start.
# Change to (1, 3, 155648, 1) once the reshape has moved into data loading.
INPUT_SHAPE = (1, 3, 155648)

# ---------------------------------------------------------------------------
# 2. Quantize
# ---------------------------------------------------------------------------
TARGET = "DPUCZDX8G_ISA1_B4096"  # KV260 DPU, as in the inspector
OUTPUT_DIR = "quant_check"
device = torch.device("cpu")


def calibration_batches(n=8):
    """Random inputs in [-1, 1], matching the normalised detector data."""
    for _ in range(n):
        yield torch.rand(INPUT_SHAPE) * 2 - 1


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("calib", "test"):
        sys.exit("usage: python compile_check.py calib|test")
    mode = sys.argv[1]

    model.eval()
    dummy = torch.rand(INPUT_SHAPE) * 2 - 1  # batch size 1, required for export

    quantizer = torch_quantizer(
        mode, model, (dummy,), output_dir=OUTPUT_DIR, device=device, target=TARGET
    )
    qmodel = quantizer.quant_model

    with torch.no_grad():
        if mode == "calib":
            for x in calibration_batches():
                qmodel(x)
            quantizer.export_quant_config()
            print(f"Calibration done. Now run: python {sys.argv[0]} test")
        else:
            qmodel(dummy)
            quantizer.export_xmodel(output_dir=OUTPUT_DIR, deploy_check=False)
            print(
                f"Exported xmodel to {OUTPUT_DIR}/. Next: run vai_c_xir on the *_int.xmodel"
            )


if __name__ == "__main__":
    main()
