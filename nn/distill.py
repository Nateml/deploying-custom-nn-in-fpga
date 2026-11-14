"""Train a student model on the soft outputs of a trained teacher (knowledge distillation).

The teacher is a checkpoint saved by main.py (default: the GWaveNet2D in ../models/),
the student is any model class from evaluate.py, by default the DPU-friendly
GWaveNet2DModified_causal_fix. By default training follows the same curriculum as
training/train_model.py: stage k trains on folders 0..k, with early stopping on the
validation loss per stage. With --no-curriculum there is a single stage on all
folders, and the order of the data files is shuffled every epoch so that no folder
is always seen last.

Loss (Hinton et al. 2015, for a single sigmoid output):
    soft = T^2 * BCE(sigmoid(student / T), sigmoid(teacher / T))
    hard = BCE(sigmoid(student), label)          (pos_weight as in main.py)
    loss = alpha * soft + (1 - alpha) * hard
The soft term never reaches 0: its minimum is the entropy of the teacher outputs.

The teacher is deterministic in eval mode, so its logits are computed once per data
file and cached in memory, so every later epoch and curriculum stage reuses them.

After every epoch the training history is written next to the checkpoint:
    <output>_history.csv      one row per epoch with all metrics (for pandas)
    <output>_learning/{loss,acc}_{train,val}.txt
                              mean and stdev per epoch in the format that
                              figures/utils.py process_model_data reads

Usage (inside the container, from nn/):
    python distill.py
    python distill.py --teacher ../models/<checkpoint>.pt --student GregNet2DModified
    python distill.py --filters 16 --temperature 4 --alpha 0.9
    python distill.py --no-curriculum
    python distill.py --folders 0 --epochs 1 --train-batches 1 --val-batches 1  # smoke test
Then evaluate the student with:
    python evaluate.py <student>.pt --filters <filters>
"""

import argparse
import copy
import csv
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_prep import into_dataloader
from evaluate import MODELS, build_model, model_from_filename, roc_auc
from models.config import load_config
from models.globals import FOLDER

TEACHER = "../models/79.60000038146973_curriculum_GWaveNet2D_20epochs_10000samples_1e-05lr_64filters_16conv_2dilation_50minibatch_AdamWoptimizer.pt"
# training/train_model.py uses 8000 train and 1000 validation samples per folder,
# stored in files of 200 samples
TRAIN_BATCHES = 40
VAL_BATCHES = 5


def distillation_loss(
    student_logits, teacher_logits, y, temperature, alpha, pos_weight
):
    soft_targets = torch.sigmoid(teacher_logits / temperature)
    soft = F.binary_cross_entropy_with_logits(
        student_logits / temperature, soft_targets
    ) * (temperature**2)
    hard = F.binary_cross_entropy_with_logits(student_logits, y, pos_weight=pos_weight)
    return alpha * soft + (1 - alpha) * hard, soft, hard


class TeacherCache:
    """Dataloaders whose batches also carry the teacher logits of each sample."""

    def __init__(self, teacher, device, minibatch_size):
        self.teacher = teacher
        self.device = device
        self.minibatch_size = minibatch_size
        self.logits = {}

    @torch.no_grad()
    def _teacher_logits(self, X):
        out = []
        for i in range(0, len(X), self.minibatch_size):
            x = X[i : i + self.minibatch_size].to(self.device).float()
            out.append(self.teacher(x).squeeze(1).cpu())
        return torch.cat(out)

    def dataloader(self, split, folder, batch_n, minibatch_size):
        X, y = into_dataloader("pc", "", split, batch_n, 1, folder).dataset.tensors
        key = (split, folder, batch_n)
        if key not in self.logits:
            self.logits[key] = self._teacher_logits(X)
        data = TensorDataset(X, y.float(), self.logits[key])
        return DataLoader(data, shuffle=split == "train", batch_size=minibatch_size)


def run_epoch(model, loaders, device, args, pos_weight, optimizer=None):
    """One pass over the loaders. Trains if an optimizer is given."""
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "soft": 0.0, "hard": 0.0}
    n = agree = 0
    # per minibatch and per sample, for the stdevs (as in train_model)
    losses, correct = [], []
    # per sample, for the AUCs
    labels, student_logits, teacher_logits = [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for loader in loaders:
            for X, y, t in loader:
                X, y, t = X.to(device).float(), y.to(device), t.to(device)
                s = model(X).squeeze(1)
                loss, soft, hard = distillation_loss(
                    s, t, y, args.temperature, args.alpha, pos_weight
                )
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                for name, value in zip(totals, (loss, soft, hard)):
                    totals[name] += value.item() * len(y)
                losses.append(loss.item())
                pred = (s > 0).float()
                correct.append((pred == y).cpu().numpy())
                agree += (pred == (t > 0).float()).sum().item()
                n += len(y)
                labels.append(y.cpu().numpy())
                student_logits.append(s.detach().cpu().numpy())
                teacher_logits.append(t.cpu().numpy())
    correct = np.concatenate(correct)
    labels = np.concatenate(labels)
    stats = {name: value / n for name, value in totals.items()}
    stats["loss_std"] = np.std(losses, ddof=1) if len(losses) > 1 else 0.0
    stats["acc"] = 100 * np.mean(correct)
    stats["acc_std"] = 100 * np.std(correct, ddof=1)
    stats["agree"] = 100 * agree / n
    # AUC does not depend on the 0.5 threshold, so it shows progress while every
    # prediction is still on the same side of it
    stats["auc"] = roc_auc(labels, np.concatenate(student_logits))
    stats["teacher_auc"] = roc_auc(labels, np.concatenate(teacher_logits))
    return stats


def save_history(history, stem):
    with open(f"{stem}_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    os.makedirs(f"{stem}_learning", exist_ok=True)
    for metric in ("loss", "acc"):
        for split in ("train", "val"):
            with open(f"{stem}_learning/{metric}_{split}.txt", "w") as f:
                stage = None
                for row in history:
                    if row["stage"] != stage:
                        stage = row["stage"]
                        f.write(f"Folder {stage}\n")
                    mean, std = row[f"{split}_{metric}"], row[f"{split}_{metric}_std"]
                    if metric == "loss":
                        # figures/utils.py strips "tensor(...)" from the losses,
                        # as written by training/generate_outputs.py (its [7:-2]
                        # also cuts the last digit of the mean)
                        f.write(f"tensor({mean:.6f}), tensor({std:.6f})\n")
                    else:
                        f.write(f"{mean}, {std}\n")


def fmt(stats):
    return (
        f"loss {stats['loss']:.4f} (soft {stats['soft']:.4f}, hard {stats['hard']:.4f}), "
        f"acc {stats['acc']:.2f}%, AUC {stats['auc']:.4f} "
        f"(teacher {stats['teacher_auc']:.4f}), teacher agreement {stats['agree']:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--teacher", default=TEACHER, help="teacher checkpoint (.pt)")
    parser.add_argument(
        "--teacher-model",
        choices=MODELS,
        help="teacher class (default: read from the checkpoint filename)",
    )
    parser.add_argument(
        "--student", choices=MODELS, default="GWaveNet2DModified_causal_fix"
    )
    parser.add_argument(
        "--filters",
        type=int,
        help="out_channels of a GWaveNet student, ignored for GregNet "
        "(default: from models/configs/<student>.yaml)",
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="weight of the soft loss (0..1)"
    )
    parser.add_argument("--epochs", type=int, default=20, help="max epochs per stage")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--no-lr-decay",
        action="store_true",
        help="keep lr fixed instead of lr / (stage + 1) as for GWaveNet in main.py",
    )
    parser.add_argument("--minibatch-size", type=int, default=50)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--folders",
        type=int,
        default=FOLDER,
        help="last folder to train on (0 = folder 0 only)",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="train one stage on folders 0..--folders instead of one stage per folder",
    )
    parser.add_argument("--train-batches", type=int, default=TRAIN_BATCHES)
    parser.add_argument("--val-batches", type=int, default=VAL_BATCHES)
    parser.add_argument(
        "--output",
        help="student checkpoint to write (default: ../models/<name built from args>.pt)",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count())

    teacher_name = args.teacher_model or model_from_filename(args.teacher)
    if teacher_name is None:
        parser.error(
            "could not infer the teacher from the filename, pass --teacher-model"
        )
    teacher, _ = build_model(teacher_name)
    checkpoint = torch.load(args.teacher, map_location=device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # teacher and student each use their own models/configs/<name>.yaml
    student_config = load_config(args.student)
    if args.filters is None or args.student.startswith("GregNet"):
        args.filters = student_config.get("out_channels")
    else:
        student_config["out_channels"] = args.filters
    student, pos_weight = build_model(args.student, **student_config)
    student.to(device)
    pos_weight = torch.tensor(pos_weight, device=device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    print(f"Device: {device}")
    print(f"Teacher: {teacher_name}, {n_teacher} parameters")
    print(f"Student: {args.student}, {n_student} parameters, {student_config}")

    architecture = f"{args.filters}filters_"
    if "dilation" in student_config:
        architecture += f"{student_config['dilation']}dilation_"
    filename = (
        f"distilled_{args.student}_{architecture}from_{teacher_name}_"
        f"{args.temperature}T_{args.alpha}alpha_{args.lr}lr_{args.minibatch_size}minibatch"
    )
    if args.no_curriculum:
        filename = f"no_curriculum_{filename}"
        stages = [args.folders]
    else:
        filename = f"curriculum_{filename}"
        stages = range(args.folders + 1)
    output = args.output or f"../models/{filename}.pt"
    stem = os.path.splitext(output)[0]
    cache = TeacherCache(teacher, device, args.minibatch_size)
    rng = np.random.default_rng(42)
    history = []

    for stage_n, stage in enumerate(stages):
        if not args.no_lr_decay:
            for g in optimizer.param_groups:
                g["lr"] = args.lr / (stage_n + 1)
        print(f"Stage {stage_n}: folders 0..{stage}")

        best_loss, best_state, best_epoch, bad_epochs = np.inf, None, 0, 0
        for epoch in range(args.epochs):
            train_files = [
                (f, b) for f in range(stage + 1) for b in range(args.train_batches)
            ]
            if args.no_curriculum:
                rng.shuffle(train_files)
            # loaders are generators so only one data file is in memory at a time
            train_loaders = (
                cache.dataloader("train", f, b, args.minibatch_size)
                for f, b in train_files
            )
            val_loaders = (
                cache.dataloader("validation", f, b, args.minibatch_size)
                for f in range(stage + 1)
                for b in range(args.val_batches)
            )
            train = run_epoch(
                student, train_loaders, device, args, pos_weight, optimizer
            )
            val = run_epoch(student, val_loaders, device, args, pos_weight)
            print(
                f"Epoch {epoch + 1}\n  train: {fmt(train)}\n  val:   {fmt(val)}",
                flush=True,
            )

            improved = val["loss"] < best_loss - args.min_delta
            history.append(
                {
                    "stage": stage_n,
                    "folders": stage,
                    "epoch": epoch + 1,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"train_{k}": v for k, v in train.items()},
                    **{f"val_{k}": v for k, v in val.items()},
                    "best": improved,
                }
            )
            save_history(history, stem)

            if improved:
                best_loss, best_epoch, bad_epochs = val["loss"], epoch + 1, 0
                best_state = copy.deepcopy(student.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs > args.patience:
                    break

        # the next stage starts from the best model of this one, as in train_model
        student.load_state_dict(best_state)
        print(f"Stage {stage_n}: best epoch {best_epoch}, val loss {best_loss:.4f}")
        torch.save(
            {
                "model_state_dict": best_state,
                "model": args.student,
                "filters": args.filters,
                "config": student_config,
                "teacher": args.teacher,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "folders": stage,
                "curriculum": not args.no_curriculum,
            },
            output,
        )

    print(f"Student written to {output}")
    print(f"History written to {stem}_history.csv and {stem}_learning/")
    print(f"Evaluate with: python evaluate.py {output} --filters {args.filters}")


if __name__ == "__main__":
    main()
