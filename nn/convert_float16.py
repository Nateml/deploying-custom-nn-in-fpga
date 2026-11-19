"""Copy the parsed data files to float16, which loses nothing and is 4x smaller.

The raw frames on dcache are float16; the parsed .npz files in new_data store them
as float64. Each noise/signal array is converted only if converting it back to
float64 gives exactly the original values, otherwise the script stops. Every written
file is read back and compared again. The test_params files (event parameters, which
do need float64) are copied unchanged.

Files that already exist in the output directory are skipped, so the script can be
stopped and started again. Which files: the first --train-files, --val-files and
--test-files of each folder (default: TRAIN_FILES, VALIDATION_FILES and TEST_FILES
from models/globals.py).

Usage (from nn/):
    vai python convert_float16.py
    vai python convert_float16.py --train-files 100 --workers 4
"""

import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from models.globals import FOLDER, TEST_FILES, TRAIN_FILES, VALIDATION_FILES

SOURCE = "/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_data/"
DESTINATION = "/data/gravwav/nmacdona/earlywarningfpgas/parsed_data/new_data_float16/"


def to_float16(x, name):
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: contains NaN or inf")
    h = x.astype(np.float16)
    if not np.array_equal(h.astype(x.dtype), x):
        raise ValueError(f"{name}: not exactly representable in float16")
    return h


def convert(name, source, destination):
    out = os.path.join(destination, name)
    if os.path.exists(out):
        return name, "exists"
    tmp = f"{out}.tmp.npz"
    if name.startswith("test_params_"):
        shutil.copyfile(os.path.join(source, name), tmp)
    else:
        with np.load(os.path.join(source, name)) as z:
            arrays = {key: to_float16(z[key], f"{name}[{key}]") for key in z.files}
        np.savez(tmp, **arrays)
        with np.load(tmp) as z:
            if set(z.files) != set(arrays) or not all(
                np.array_equal(z[key], arrays[key]) for key in arrays
            ):
                raise ValueError(
                    f"{name}: written file differs from the converted data"
                )
        del arrays
    os.replace(tmp, out)
    return name, "done"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--destination", default=DESTINATION)
    parser.add_argument("--train-files", type=int, default=TRAIN_FILES)
    parser.add_argument("--val-files", type=int, default=VALIDATION_FILES)
    parser.add_argument("--test-files", type=int, default=TEST_FILES)
    parser.add_argument(
        "--workers", type=int, default=3, help="files converted at the same time"
    )
    args = parser.parse_args()

    names = []
    for folder in range(FOLDER + 1):
        for split, n in (
            ("train", args.train_files),
            ("validation", args.val_files),
            ("test", args.test_files),
        ):
            names += [f"{split}_folder{folder}_batch{b}.npz" for b in range(n)]
        names += [
            f"test_params_folder{folder}_batch{b}.npz" for b in range(args.test_files)
        ]
    missing = [n for n in names if not os.path.exists(os.path.join(args.source, n))]
    if missing:
        parser.error(f"{len(missing)} files not in {args.source}, e.g. {missing[0]}")

    os.makedirs(args.destination, exist_ok=True)
    # leftovers of an interrupted run
    for f in os.listdir(args.destination):
        if f.endswith(".tmp.npz"):
            os.remove(os.path.join(args.destination, f))

    print(f"{len(names)} files from {args.source} to {args.destination}", flush=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(
            convert, names, [args.source] * len(names), [args.destination] * len(names)
        )
        for i, (name, status) in enumerate(results, 1):
            elapsed = time.time() - start
            print(
                f"[{i}/{len(names)}, {elapsed / 60:.1f} min] {name}: {status}",
                flush=True,
            )
    print("All files converted and checked.")


if __name__ == "__main__":
    main()
