"""Evaluate a trained checkpoint on the curriculum test set.

Runs the model over every test batch of every curriculum folder, prints accuracy,
loss, confusion matrix and ROC AUC per folder and overall, then accuracy and
true positive rate at the threshold that gives a false alarm probability (FAP) of
--fap (1% by default, the level used in the paper) on all validation noise, and
writes the
probabilities in the "label: probability: PISNR" format that
figures/plotting.ipynb reads (see figures/data/probabilities/).

Usage (inside the container, from nn/):
    python evaluate.py ../models/<checkpoint>.pt
    python evaluate.py ../models/<checkpoint>.pt --model GWaveNet2D --batches 2
"""

import argparse
import os
import re

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn

from data_prep import into_dataloader
from models.config import MODEL_NAMES as MODELS
from models.config import build_model
from models.globals import FOLDER
from training.utils import confusion_matrix

TEST_DATA_DIR = "/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_data/"
# column of the test_params arrays holding the PISNR of each sample
PISNR_COLUMN = 3


def model_from_filename(path):
    # longest name first, so "GWaveNet2DModified" wins over "GWaveNet"
    for name in sorted(MODELS, key=len, reverse=True):
        if re.search(rf"_{name}_", os.path.basename(path)):
            return name
    return None


def n_batches(split, folder):
    pattern = re.compile(rf"{split}_folder{folder}_batch(\d+)\.npz")
    return sum(bool(pattern.fullmatch(f)) for f in os.listdir(TEST_DATA_DIR))


def load_pisnr(folder, batch_n):
    # into_dataloader stacks noise before signal and drops rows with NaNs or
    # values out of [-1, 1]; this assumes no rows were dropped (checked by caller)
    params = np.load(f"{TEST_DATA_DIR}test_params_folder{folder}_batch{batch_n}.npz")
    return np.concatenate(
        (params["noise"][:, PISNR_COLUMN], params["signal"][:, PISNR_COLUMN])
    )


def roc_auc(labels, probabilities):
    # Mann-Whitney U statistic; sklearn.metrics fails to import in the container
    # (libstdc++ too old for its compiled extensions)
    ranks = rankdata(probabilities)
    n_pos = np.sum(labels == 1)
    n_neg = len(labels) - n_pos
    return (np.sum(ranks[labels == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def report(tag, labels, probabilities, loss):
    pred_labels = np.round(probabilities)
    accuracy = 100 * np.mean(pred_labels == labels)
    tp, tn, fp, fn = confusion_matrix(pred_labels, labels)
    auc = roc_auc(labels, probabilities)
    print(
        f"{tag}: accuracy {accuracy:.2f}%, loss {loss:.4f}, AUC {auc:.4f}, "
        f"TP {tp}, TN {tn}, FP {fp}, FN {fn}"
    )


def fap_threshold(model, device, fap, batches, minibatch_size, split="validation"):
    """Threshold above which a fraction fap of the noise samples of split lies,
    over all folders (one threshold, as a deployed detector would use)."""
    noise = []
    with torch.no_grad():
        for folder in range(FOLDER + 1):
            n = n_batches(split, folder)
            if batches is not None:
                n = min(n, batches)
            for batch_n in range(n):
                print(f"Folder {folder}, {split} batch {batch_n + 1}/{n}")
                for X, y in into_dataloader(
                    "pc", "", split, batch_n, minibatch_size, folder
                ):
                    pred = model(X.to(device).float()).squeeze(1)
                    noise.append(torch.sigmoid(pred).cpu().numpy()[y.numpy() == 0])
    return np.quantile(np.concatenate(noise), 1 - fap)


def report_at_threshold(tag, labels, probabilities, threshold):
    detected = probabilities > threshold
    accuracy = 100 * np.mean(detected == (labels == 1))
    tpr = 100 * np.mean(detected[labels == 1])
    fap = 100 * np.mean(detected[labels == 0])
    print(f"{tag}: accuracy {accuracy:.2f}%, TPR {tpr:.2f}%, FAP {fap:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", help="path to a .pt file saved by main.py")
    parser.add_argument(
        "--model",
        choices=MODELS,
        help="model class (default: read from the checkpoint filename)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        help="number of test batches per folder (default: all)",
    )
    parser.add_argument(
        "--filters",
        type=int,
        help="out_channels of a GWaveNet model, e.g. a distilled student "
        "(default: from models/configs/<model>.yaml)",
    )
    parser.add_argument(
        "--fap",
        type=float,
        default=0.01,
        help="false alarm probability that sets the second decision threshold",
    )
    parser.add_argument("--minibatch-size", type=int, default=50)
    parser.add_argument(
        "--output",
        help="probabilities file to write (default: next to the checkpoint)",
    )
    args = parser.parse_args()

    name = args.model or model_from_filename(args.checkpoint)
    if name is None:
        parser.error("could not infer the model from the filename, pass --model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count())
    print(f"Model: {name}, device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    overrides = {} if args.filters is None else {"out_channels": args.filters}
    # distill.py checkpoints carry the architecture they were trained with
    model, pos_weight = build_model(name, checkpoint.get("config"), **overrides)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    output = args.output or os.path.splitext(args.checkpoint)[0] + "_probabilities.txt"
    all_labels, all_probabilities, all_losses = [], [], []

    with open(output, "w") as out, torch.no_grad():
        for folder in range(FOLDER + 1):
            out.write(f"Folder {folder}\n")
            n_test = n_batches("test", folder)
            if args.batches is not None:
                n_test = min(n_test, args.batches)

            labels, probabilities, losses = [], [], []
            for batch_n in range(n_test):
                print(f"Folder {folder}, test batch {batch_n + 1}/{n_test}")
                dataloader = into_dataloader(
                    "pc", "", "test", batch_n, args.minibatch_size, folder
                )
                pisnr = load_pisnr(folder, batch_n)
                if len(pisnr) != len(dataloader.dataset):
                    raise RuntimeError(
                        f"folder {folder} batch {batch_n}: samples were dropped, "
                        "PISNR no longer lines up with the predictions"
                    )

                batch_labels, batch_probabilities = [], []
                for X, y in dataloader:
                    X, y = X.to(device).float(), y.to(device).float()
                    pred = model(X).squeeze(1)
                    losses.append(loss_fn(pred, y).item())
                    batch_probabilities.append(torch.sigmoid(pred).cpu().numpy())
                    batch_labels.append(y.cpu().numpy())

                batch_labels = np.concatenate(batch_labels)
                batch_probabilities = np.concatenate(batch_probabilities)
                for label, probability, snr in zip(
                    batch_labels, batch_probabilities, pisnr
                ):
                    out.write(f"{label}: {probability}: {snr}\n")
                labels.append(batch_labels)
                probabilities.append(batch_probabilities)

            labels = np.concatenate(labels)
            probabilities = np.concatenate(probabilities)
            report(f"Folder {folder}", labels, probabilities, np.mean(losses))
            all_labels.append(labels)
            all_probabilities.append(probabilities)
            all_losses.extend(losses)

    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    report("Overall", labels, probabilities, np.mean(all_losses))

    # set on validation noise, so the threshold isn't tuned on the test set; the
    # FAP measured on the test noise then shows how well it carries over
    threshold = fap_threshold(
        model, device, args.fap, args.batches, args.minibatch_size
    )
    print(
        f"\nAt FAP {100 * args.fap:g}% (threshold {threshold:.4f}, "
        "set on all validation noise):"
    )
    for folder, (folder_labels, folder_probabilities) in enumerate(
        zip(all_labels, all_probabilities)
    ):
        report_at_threshold(
            f"Folder {folder}", folder_labels, folder_probabilities, threshold
        )
    report_at_threshold("Overall", labels, probabilities, threshold)
    print(f"Probabilities written to {output}")


if __name__ == "__main__":
    main()
