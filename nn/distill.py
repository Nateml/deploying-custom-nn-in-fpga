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

--seeds trains one student per seed in the same job. Each data file is loaded once
and used by all of them in turn, so a few seeds take about as long as one (loading
the data is the slow part). The seed sets the student's initial weights and the
order of the minibatches within each file. The order of the files is the same for
all seeds (--data-seed), so a seed's result does not depend on which other seeds
run with it. Each student has its own early stopping; a stage ends when every
student has stopped.

After every epoch the training history is written next to the checkpoint:
    <output>_history.csv      one row per epoch with all metrics (for pandas)
    <output>_learning/{loss,acc}_{train,val}.txt
                              mean and stdev per epoch in the format that
                              figures/utils.py process_model_data reads

After every epoch the training state is saved to <output>_resume.pt (with several
seeds: <output without _seed>_seeds<a-b-c>_resume.pt). If that file exists at start,
training continues from it; it is deleted when training finishes. With --time-limit
the script stops after the last epoch that ends within that many hours, and exits
with code 3, so a chain of short condor jobs (condor/distill_dag.py) can resume it.

Usage (inside the container, from nn/):
    python distill.py
    python distill.py --teacher ../models/<checkpoint>.pt --student GregNet2DModified
    python distill.py --filters 16 --temperature 4 --alpha 0.9
    python distill.py --no-curriculum
    python distill.py --seeds 1 2 3 4
    python distill.py --folders 0 --epochs 1 --train-batches 1 --val-batches 1  # smoke test
Then evaluate the student with:
    python evaluate.py <student>.pt --filters <filters>
"""

import argparse
import copy
import csv
import math
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from data_prep import into_dataloader, prefetched
from evaluate import MODELS, build_model, model_from_filename, roc_auc
from models.config import load_config
from models.globals import FOLDER, TRAIN_FILES, VALIDATION_FILES

# GWaveNet2D retrained as in the thesis (80/13/22 files, lr 2e-5); the earlier teacher
# (40/5/5 files, lr 1e-5) is ../models/79.60000038146973_curriculum_GWaveNet2D_20epochs_10000samples_1e-05lr_64filters_16conv_2dilation_50minibatch_AdamWoptimizer.pt
TEACHER = "/data/gravwav/nmacdona/earlywarningfpgas/output/models/78.58636379241943_curriculum_GWaveNet2D_20epochs_2e-05lr_64filters_16conv_2dilation_50minibatch_AdamWoptimizer_80-13-22files.pt"
# same files per folder as training/train_model.py (files of 200 samples)
TRAIN_BATCHES = TRAIN_FILES
VAL_BATCHES = VALIDATION_FILES
# exit code when --time-limit stops training before the end
EXIT_UNFINISHED = 3
# early stopping state of a Run within a stage
STAGE_STATE = ("best_loss", "best_state", "best_epoch", "bad_epochs", "active")


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
    """Data files together with the teacher logits of each sample."""

    def __init__(self, teacher, device, minibatch_size):
        self.teacher = teacher
        self.device = device
        self.minibatch_size = minibatch_size
        self.logits = {}
        # time spent computing teacher logits, to estimate the next epoch's duration
        self.seconds = 0.0
        self.n_files = 0

    @torch.no_grad()
    def _teacher_logits(self, X):
        out = []
        for i in range(0, len(X), self.minibatch_size):
            x = X[i : i + self.minibatch_size].to(self.device).float()
            out.append(self.teacher(x).squeeze(1).cpu())
        return torch.cat(out)

    def files(self, split, files):
        """(X, y, teacher logits) on the device for each (folder, batch_n) in files.
        The next file is read in the background while the current one is used."""

        def load(folder, batch_n):
            return into_dataloader("pc", "", split, batch_n, 1, folder).dataset.tensors

        for (folder, batch_n), (X, y) in zip(files, prefetched(load, files)):
            key = (split, folder, batch_n)
            if key not in self.logits:
                start = time.time()
                self.logits[key] = self._teacher_logits(X)
                self.seconds += time.time() - start
                self.n_files += 1
            yield (
                X.to(self.device).float(),
                y.to(self.device).float(),
                self.logits[key].to(self.device),
            )


class EpochStats:
    """Metrics of one student over one epoch."""

    def __init__(self):
        self.totals = {"loss": 0.0, "soft": 0.0, "hard": 0.0}
        self.n = self.agree = 0
        # per minibatch and per sample, for the stdevs (as in train_model)
        self.losses, self.correct = [], []
        # per sample, for the AUCs
        self.labels, self.student_logits, self.teacher_logits = [], [], []

    def add(self, s, t, y, loss, soft, hard):
        for name, value in zip(self.totals, (loss, soft, hard)):
            self.totals[name] += value.item() * len(y)
        self.losses.append(loss.item())
        pred = (s > 0).float()
        self.correct.append((pred == y).cpu().numpy())
        self.agree += (pred == (t > 0).float()).sum().item()
        self.n += len(y)
        self.labels.append(y.cpu().numpy())
        self.student_logits.append(s.detach().cpu().numpy())
        self.teacher_logits.append(t.cpu().numpy())

    def result(self):
        correct = np.concatenate(self.correct)
        labels = np.concatenate(self.labels)
        stats = {name: value / self.n for name, value in self.totals.items()}
        stats["loss_std"] = np.std(self.losses, ddof=1) if len(self.losses) > 1 else 0.0
        stats["acc"] = 100 * np.mean(correct)
        stats["acc_std"] = 100 * np.std(correct, ddof=1)
        stats["agree"] = 100 * self.agree / self.n
        # AUC does not depend on the 0.5 threshold, so it shows progress while every
        # prediction is still on the same side of it
        stats["auc"] = roc_auc(labels, np.concatenate(self.student_logits))
        stats["teacher_auc"] = roc_auc(labels, np.concatenate(self.teacher_logits))
        return stats


def run_epoch(runs, files, args, pos_weight, train):
    """One pass over the data files for every run in runs, which train if train is
    set. Returns the stats of each run."""
    stats = [EpochStats() for _ in runs]
    for run in runs:
        run.model.train(train)
    with torch.set_grad_enabled(train):
        for X, y, t in files:
            for run, run_stats in zip(runs, stats):
                if train:
                    order = torch.randperm(len(y), generator=run.generator)
                else:
                    order = torch.arange(len(y))
                for idx in order.to(y.device).split(args.minibatch_size):
                    s = run.model(X[idx]).squeeze(1)
                    loss, soft, hard = distillation_loss(
                        s, t[idx], y[idx], args.temperature, args.alpha, pos_weight
                    )
                    if train:
                        run.optimizer.zero_grad()
                        loss.backward()
                        run.optimizer.step()
                    run_stats.add(s, t[idx], y[idx], loss, soft, hard)
    return [s.result() for s in stats]


class Run:
    """One student (one seed) with its own optimizer, early stopping, checkpoint and
    history."""

    def __init__(self, seed, args, student_config, device, output):
        self.seed = seed
        torch.manual_seed(seed)
        self.model, _ = build_model(args.student, **student_config)
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        # minibatch order within each file
        self.generator = torch.Generator().manual_seed(seed)
        self.output = output
        self.stem = os.path.splitext(output)[0]
        self.history = []

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def start_stage(self, lr):
        self.set_lr(lr)
        self.best_loss, self.best_state, self.best_epoch = np.inf, None, 0
        self.bad_epochs = 0
        self.active = True

    def state(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "history": self.history,
            **{key: getattr(self, key) for key in STAGE_STATE},
        }

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.history = state["history"]
        for key in STAGE_STATE:
            setattr(self, key, state[key])

    def end_epoch(self, stage_n, stage, epoch, train, val, args):
        print(
            f"Epoch {epoch + 1}, seed {self.seed}\n"
            f"  train: {fmt(train)}\n  val:   {fmt(val)}",
            flush=True,
        )
        improved = val["loss"] < self.best_loss - args.min_delta
        self.history.append(
            {
                "stage": stage_n,
                "folders": stage,
                "epoch": epoch + 1,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train.items()},
                **{f"val_{k}": v for k, v in val.items()},
                "best": improved,
            }
        )
        save_history(self.history, self.stem)

        if improved:
            self.best_loss, self.best_epoch, self.bad_epochs = val["loss"], epoch + 1, 0
            self.best_state = copy.deepcopy(self.model.state_dict())
        else:
            self.bad_epochs += 1
            if self.bad_epochs > args.patience:
                self.active = False

    def end_stage(self, stage_n, stage, args, student_config):
        # the next stage starts from the best model of this one, as in train_model
        self.model.load_state_dict(self.best_state)
        print(
            f"Stage {stage_n}, seed {self.seed}: best epoch {self.best_epoch}, "
            f"val loss {self.best_loss:.4f}"
        )
        torch.save(
            {
                "model_state_dict": self.best_state,
                "model": args.student,
                "filters": args.filters,
                "config": student_config,
                "teacher": args.teacher,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "seed": self.seed,
                "data_seed": args.data_seed,
                "folders": stage,
                "curriculum": not args.no_curriculum,
            },
            self.output,
        )


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


def save_resume_state(path, state):
    # write to a temporary file first, so a job killed while saving leaves the
    # previous state intact
    torch.save(state, f"{path}.tmp")
    os.replace(f"{path}.tmp", path)


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
    parser.add_argument(
        "--head-channels",
        type=int,
        help="channels per time position going into fc1, causal_fix student only "
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
    parser.add_argument(
        "--cosine",
        action="store_true",
        help="in the last stage (the only one with --no-curriculum), lower the lr "
        "from the stage's lr to --min-lr along a cosine over --epochs epochs",
    )
    parser.add_argument(
        "--min-lr", type=float, default=0.0, help="final lr of --cosine"
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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="train one student per seed, sharing the data loading",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="seed of the file order with --no-curriculum (the same for all seeds)",
    )
    parser.add_argument("--train-batches", type=int, default=TRAIN_BATCHES)
    parser.add_argument("--val-batches", type=int, default=VAL_BATCHES)
    parser.add_argument(
        "--output",
        help="student checkpoint to write (default: ../models/<name built from args>"
        "_seed<seed>.pt); with several seeds _seed<seed> is added before .pt",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        help="hours; stop (exit code 3, state saved) before an epoch that would end "
        "later than this",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not repeat")

    start_time = time.time()
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
    if args.head_channels is not None:
        student_config["head_channels"] = args.head_channels
    student, pos_weight = build_model(args.student, **student_config)
    pos_weight = torch.tensor(pos_weight, device=device)

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    del student
    print(f"Device: {device}")
    print(f"Teacher: {teacher_name}, {n_teacher} parameters")
    print(f"Student: {args.student}, {n_student} parameters, {student_config}")
    print(f"Seeds: {args.seeds}")

    architecture = f"{args.filters}filters_"
    if "dilation" in student_config:
        architecture += f"{student_config['dilation']}dilation_"
    if student_config.get("residual_fix"):
        architecture += "resfix_"
    if student_config.get("head_channels", 1) != 1:
        architecture += f"{student_config['head_channels']}head_"
    filename = (
        f"distilled_{args.student}_{architecture}from_{teacher_name}_"
        f"{args.temperature}T_{args.alpha}alpha_{args.lr}lr_{args.minibatch_size}minibatch_"
        f"{args.train_batches}-{args.val_batches}files"
    )
    if args.cosine:
        filename += f"_cosine{args.min_lr}minlr"
    if args.no_curriculum:
        filename = f"no_curriculum_{filename}"
        stages = [args.folders]
    else:
        filename = f"curriculum_{filename}"
        stages = range(args.folders + 1)
    stem = (
        f"../models/{filename}"
        if args.output is None
        else os.path.splitext(args.output)[0]
    )
    if len(args.seeds) > 1:
        resume_file = f"{stem}_seeds{'-'.join(map(str, args.seeds))}_resume.pt"
    elif args.output is None:
        resume_file = f"{stem}_seed{args.seeds[0]}_resume.pt"
    else:
        resume_file = f"{stem}_resume.pt"
    runs = []
    for seed in args.seeds:
        if args.output is None:
            output = f"../models/{filename}_seed{seed}.pt"
        elif len(args.seeds) > 1:
            output = f"{os.path.splitext(args.output)[0]}_seed{seed}.pt"
        else:
            output = args.output
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        runs.append(Run(seed, args, student_config, device, output))
    cache = TeacherCache(teacher, device, args.minibatch_size)
    rng = np.random.default_rng(args.data_seed)

    resume = None
    if os.path.exists(resume_file):
        resume = torch.load(resume_file, map_location=device)
        cache.logits = resume["teacher_logits"]
        rng.bit_generator.state = resume["rng"]
        print(
            f"Resuming from {resume_file}: stage {resume['stage_n']}, "
            f"epoch {resume['epoch'] + 1}"
        )

    for stage_n, stage in enumerate(stages):
        if resume is not None and stage_n < resume["stage_n"]:
            continue
        lr = args.lr if args.no_lr_decay else args.lr / (stage_n + 1)
        for run in runs:
            run.start_stage(lr)
        print(f"Stage {stage_n}: folders 0..{stage}")
        first_epoch = 0
        if resume is not None:
            for run, state in zip(runs, resume["runs"]):
                run.load_state(state)
            first_epoch = resume["epoch"]
            resume = None

        cosine = args.cosine and stage_n == len(stages) - 1
        for epoch in range(first_epoch, args.epochs):
            active = [run for run in runs if run.active]
            if not active:
                break
            if cosine:
                # from lr at the first epoch towards min_lr after the last one
                progress = epoch / args.epochs
                epoch_lr = (
                    args.min_lr
                    + (lr - args.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                )
                for run in active:
                    run.set_lr(epoch_lr)
            train_files = [
                (f, b) for f in range(stage + 1) for b in range(args.train_batches)
            ]
            if args.no_curriculum:
                rng.shuffle(train_files)
            val_files = [
                (f, b) for f in range(stage + 1) for b in range(args.val_batches)
            ]
            epoch_start, teacher_start = time.time(), cache.seconds
            # generators, so at most two data files are in memory at a time
            train = run_epoch(
                active, cache.files("train", train_files), args, pos_weight, True
            )
            val = run_epoch(
                active, cache.files("validation", val_files), args, pos_weight, False
            )
            for run, run_train, run_val in zip(active, train, val):
                run.end_epoch(stage_n, stage, epoch, run_train, run_val, args)

            save_resume_state(
                resume_file,
                {
                    "stage_n": stage_n,
                    "epoch": epoch + 1,
                    "runs": [run.state() for run in runs],
                    "teacher_logits": cache.logits,
                    "rng": rng.bit_generator.state,
                },
            )
            if args.time_limit is not None:
                # duration of the next epoch from this one's time per data file,
                # plus the teacher logits of the files a next stage adds
                n_files = len(train_files) + len(val_files)
                per_file = (
                    time.time() - epoch_start - (cache.seconds - teacher_start)
                ) / n_files
                if any(run.active for run in runs) and epoch + 1 < args.epochs:
                    next_epoch = n_files * per_file
                elif stage_n + 1 < len(stages):
                    new_files = args.train_batches + args.val_batches
                    next_epoch = (n_files + new_files) * per_file + new_files * (
                        cache.seconds / max(cache.n_files, 1)
                    )
                else:
                    next_epoch = 0.0
                elapsed = time.time() - start_time
                if elapsed + 1.1 * next_epoch > args.time_limit * 3600:
                    print(
                        f"Stopping after {elapsed / 3600:.2f} h, the next epoch would "
                        f"take about {next_epoch / 60:.0f} min; state saved to "
                        f"{resume_file}",
                        flush=True,
                    )
                    sys.exit(EXIT_UNFINISHED)

        for run in runs:
            run.end_stage(stage_n, stage, args, student_config)

    if os.path.exists(resume_file):
        os.remove(resume_file)
    for run in runs:
        print(f"Student written to {run.output}")
        print(f"History written to {run.stem}_history.csv and {run.stem}_learning/")
    print(f"Evaluate with: python evaluate.py {runs[0].output}")


if __name__ == "__main__":
    main()
