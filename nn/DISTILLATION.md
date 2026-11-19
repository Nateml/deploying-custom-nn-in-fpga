# Knowledge distillation

`distill.py` trains a small **student** model to copy the outputs of a trained **teacher**. The goal is a model that fits the KV260 DPU (the default student is `GWaveNet2DModified_causal_fix`) while keeping as much as possible of the teacher's accuracy.

## Quick start

Run everything from `nn/`, inside the container:

```bash
# defaults: retrained GWaveNet2D teacher (80/13/22 files, 79.10% at FAP 1%), causal_fix student with 64 filters, curriculum
vai python distill.py

# a smaller student with softer targets
vai python distill.py --filters 16 --temperature 4

# one stage on all folders instead of the curriculum
vai python distill.py --no-curriculum

# four seeds in one job (about as fast as one, since loading the data is the slow part)
vai python distill.py --seeds 1 2 3 4

# smoke test (about 2 min on CPU)
vai python distill.py --folders 0 --epochs 1 --train-batches 1 --val-batches 1 --filters 8 \
    --output /tmp/smoke.pt

# full run on a GPU node
condor_submit condor/job.sub 'args=distill.py --filters 16'

# one student per seed as a chain of short (4 h) jobs that resume each other
python3 condor/distill_dag.py kd --seeds 1 2 3 4 5 -- --lr 1e-4
condor_submit_dag /data/gravwav/nmacdona/earlywarningfpgas/output/condor/dags/kd.dag

# short lr sweep (1e-5, 3e-5, 1e-4, 3e-4, 1e-3), about 1 h per job -> ../models/lr_sweep/
condor_submit condor/lr_sweep.sub
```

Teacher and student are each built from their own `models/configs/<name>.yaml` (e.g. the teacher with dilation 2, the causal_fix student with dilation 1); `--filters` overrides the student's `out_channels`, and the full student config is printed and saved in the checkpoint.

The causal_fix config sets `residual_fix: true` (residual connections in Mod1–6; before 2026-11-15 only Mod5 had one, and those checkpoints have no `resfix_` in their name). Checkpoints keep working either way, because `evaluate.py` builds the student from the config saved in its checkpoint.

Then evaluate the student like any other checkpoint. The architecture, including `--filters`, comes from the checkpoint:

```bash
vai python evaluate.py ../models/curriculum_distilled_<...>_seed42.pt
```

If you used a custom `--output` name, also pass `--model GWaveNet2DModified_causal_fix`, because `evaluate.py` reads the model class from the filename.

## What it does

For each minibatch the loss is

```
soft = T² · BCE(sigmoid(student / T), sigmoid(teacher / T))
hard = BCE(sigmoid(student), label)            # pos_weight as in main.py
loss = alpha · soft + (1 − alpha) · hard
```

- A higher temperature `T` flattens the teacher's probabilities, so the student also learns *how confident* the teacher is, not just which class it picked. The `T²` factor keeps the soft term's gradients at a similar scale when you change `T`.
- `alpha` balances copying the teacher (`soft`) against fitting the true labels (`hard`).
- The teacher runs once per data file, and its logits are cached in memory, so later epochs and curriculum stages reuse them. The cache is small (one float per sample).

**Curriculum (default):** the same schedule as `training/train_model.py`. Stage k trains on folders 0..k, with its own early stopping and lr = `--lr / (k + 1)`, and each stage starts from the best model of the previous one.

**`--no-curriculum`:** a single stage on folders 0..`--folders`, at a fixed learning rate. The order of the data files is shuffled every epoch (seeded), but each file holds only one folder, so every minibatch still comes from a single folder.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--teacher` | retrained GWaveNet2D in `OUTPUT_DIR/models/` (before 2026-11-18: the 79.6 checkpoint in `../models/`) | Teacher checkpoint saved by `main.py` |
| `--teacher-model` | from filename | Teacher class, if it can't be read from the filename |
| `--student` | `GWaveNet2DModified_causal_fix` | Any class known to `evaluate.py` |
| `--filters` | from the student's config | Student width (GWaveNet students only; GregNet is fixed) |
| `--head-channels` | from the student's config (1) | Channels per time position going into `fc1` (causal_fix student only). Each extra channel adds about 61k `fc1` weights |
| `--temperature` | 2.0 | `T` |
| `--alpha` | 0.7 | Weight of the soft loss |
| `--epochs` | 20 | Max epochs **per stage** |
| `--lr` | 1e-5 | AdamW learning rate |
| `--no-lr-decay` | off | Keep lr fixed across curriculum stages |
| `--cosine`, `--min-lr` | off, 0 | In the last stage (the only one with `--no-curriculum`), lower the lr per epoch along a cosine, from the stage's lr at the first epoch to `--min-lr` after epoch `--epochs`. Adds `_cosine<min-lr>minlr` to the name. Early stopping still applies, so with the default `--patience` 2 a run usually stops before the lr has come down much; raise `--patience` (e.g. to `--epochs`) to run the whole schedule and keep the best epoch |
| `--patience`, `--min-delta` | 2, 1e-4 | Early stopping on the validation loss |
| `--folders` | 4 | Last folder used (0 = folder 0 only) |
| `--no-curriculum` | off | See above |
| `--seeds` | 42 | One student per seed, trained side by side on the same data files (see below) |
| `--data-seed` | 42 | Order of the data files with `--no-curriculum`, the same for all seeds |
| `--train-batches`, `--val-batches` | 80, 13 | Data files per folder (`TRAIN_FILES`, `VALIDATION_FILES` in `models/globals.py`, as in `train_model.py`). Added to the name as `_<train>-<val>files`. Runs before 2026-11-18 used 40 and 5 |
| `--time-limit` | none | Hours. Stops after the last epoch expected to end within it (state saved, exit code 3), so the next job can resume |
| `--output` | `../models/{curriculum,no_curriculum}_distilled_<args>_seed<seed>.pt` | Where the student is saved. With several seeds, `_seed<seed>` is added before `.pt` |

The checkpoint holds `model_state_dict` (so `evaluate.py` can load it), plus `model`, `filters`, `config`, `teacher`, `temperature`, `alpha`, `seed`, `data_seed`, `folders` and `curriculum`. It is rewritten after every stage, so a stopped run still leaves the last finished stage on disk.

**Seeds.** With `--seeds 1 2 3 4`, every data file is loaded once and each student trains on it in turn, so the seeds share the slow part (reading the data). The GPU part is about 0.5 s per file per student on an L40S (estimated), against 2.7–4 s to load a file. So up to about 4 seeds take roughly as long as one, and each student needs about 14 GB of GPU memory at most while it runs (they take turns). The seed sets the initial weights and the minibatch order within each file. The order of the files comes from `--data-seed` and is the same for every seed in the job, so a seed gives the same result whether it runs alone or with others, apart from the GPU's own run-to-run noise, which is about 0.5% in the metrics after an epoch. Each student has its own early stopping: one that has stopped keeps its best model and sits out, and a stage ends when all have stopped (in the curriculum all students then move to the next stage together). Runs from before `--seeds` (no `_seed` in the name) used one global seed for everything and can't be reproduced exactly.

## Training history

After **every epoch** the full history so far is written next to the checkpoint, so runs that are still going or were stopped can be plotted too:

- **`<output>_history.csv`:** one row per epoch with `stage`, `folders`, `epoch`, `lr`, and for both `train_` and `val_`: `loss`, `soft`, `hard`, `loss_std`, `acc`, `acc_std`, `agree`, `auc`, `teacher_auc`. The `best` column marks epochs that improved the stage's best validation loss (the last `True` in a stage is the model that was kept).

  ```python
  import pandas as pd
  h = pd.read_csv("../models/curriculum_distilled_<...>_history.csv")
  h.plot(y=["train_loss", "val_loss"])           # one point per epoch, all stages in a row
  h.groupby("stage").size().cumsum()             # epoch index where each stage ends
  ```

- **`<output>_learning/{loss,acc}_{train,val}.txt`:** the format that `figures/utils.py` `process_model_data` reads (one `Folder k` block per stage, then `mean, stdev` per epoch). Point the `filename_learning_*` variables in `plotting.ipynb` at these files, or copy them to `figures/data/learning/{loss,acc}/{train,val}/<Name>.txt`.

How to read these files:
- The loss in the plotting files is the total distillation loss (the one early stopping uses). The soft and hard parts are only in the CSV.
- The standard deviations are calculated as in `train_model.py`: loss over minibatches, accuracy over the per-sample 0/1 correctness (so for accuracy it is always about 50).
- `process_model_data` works out its own best epoch per stage as the strict minimum of the validation loss, while `distill.py` requires an improvement of at least `--min-delta`. So the two can differ by an epoch or so when the loss flattens out. The `best` column in the CSV is the one `distill.py` actually used.

## Things to keep in mind

**Reading the logs**
- The soft loss can't reach 0. Its lowest possible value is the entropy of the teacher's outputs, so compare runs on **accuracy, teacher agreement and test AUC**, not on the loss value.
- Losses from runs with different `T` or `alpha` can't be compared directly, and neither can their early-stopping behaviour.
- *Teacher agreement* is how often the student and teacher pick the same class. A high agreement with low accuracy means the student is copying the teacher's mistakes.
- Accuracy and agreement only change when outputs cross the 0.5 threshold. A fresh student often first calls everything noise (`pos_weight` 0.1 makes that cheap) and stays at exactly 50% for several epochs while it is already learning. *AUC* measures how well the outputs rank signals above noise, whatever the threshold, so it shows that progress; *teacher AUC* (on the same samples) is the reference to aim for.

**What limits the student**
- The teacher scores about 79.6% test accuracy. Distillation copies its mistakes as well as its knowledge, so don't expect the student to beat it by much, if at all.
- The teacher was trained on these same training files, so its outputs there are more confident than on unseen data. That makes the soft targets on the training set sharper than ideal; a higher `T` partly makes up for it.
- Weights can't usefully be copied from the teacher into the student: the kernel sizes (16 vs 2), channel layout and activations all differ. Only about 2% of the student's values even have matching shapes. To start from trained weights, first train a full-size causal_fix model and shrink that.

**Running fair experiments**
- Change one thing at a time, and keep the seed, data files (`--train-batches`, `--val-batches`) and evaluation the same.
- Watch the epoch budget. The curriculum gets up to 20 epochs *per stage*, while `--no-curriculum` gets 20 in total. If a run is still improving when it hits the limit, raise `--epochs`.
- Always compare against a student of the same size trained **without** distillation (`--alpha 0`, which keeps only the label loss). Otherwise you can't tell whether distillation helped.
- `main.py`'s "no_curriculum" mode (`FOLDER = 0`) trains on folder 0 only, so it isn't the same as `--no-curriculum` here.
- The default lr (1e-5) comes from the teacher's settings. A freshly initialised small student may need a higher one, so a short sweep (e.g. 1e-5, 1e-4, 1e-3) is worth doing.

**Practicalities**
- Use a GPU node. On CPU one data file of training plus one of validation takes about 2 minutes, and a full curriculum run reads several hundred files per epoch.
- A checkpoint only works with the input length it was trained on (155648). That matters for `compile_check.py`, which uses a shorter input.
- A good float student still has to pass the DPU inspector and the compiler, and its accuracy should be checked again after INT8 quantization. Quantization can cost small models relatively more accuracy.
- `main.py` saves its training state after every epoch (`OUTPUT_DIR/models/<filename>_resume.pt`). If a job is killed, submitting it again with the same settings continues from the last finished epoch and appends to the same log. Delete that file to start from scratch. `distill.py` saves its state after every epoch too, in `<output>_resume.pt` (for several seeds in one job: `..._seeds1-2-3_resume.pt`), and continues from it when started again with the same arguments; delete it to start from scratch. The teacher logits computed so far are saved with it. `condor/distill_dag.py` uses this with `--time-limit 3.5` to train each seed as a chain of short jobs: DAGMan resubmits a node that exits with code 3 or is killed, but not one that crashed in Python (exit code 1).

## Possible next steps

- Initialise smaller students from a trained full-size causal_fix model, keeping each layer's strongest channels.
- Add a feature (hint) loss that matches intermediate block outputs, not just the final output.
- Mix folders within a minibatch, not just across files, for `--no-curriculum`.
