<h1 align="center">Online Batch Selection Methods for Training Acceleration</h1>

A research codebase for studying online batch-selection methods (RhoLoss, DivBS,
GradNorm, …) on a range of datasets, with rich training diagnostics logged to
Weights & Biases and jobs submitted via SLURM.

---

## 1. Environment

The project runs in a conda/mamba environment named `online-bs-p100` (CUDA
12.6, targeting P100 "pascal" GPUs). Create it once from the checked-in spec
(takes ~20 minutes), then activate it:

```bash
mamba env create -f environment-sc.yaml   # or: conda env create -f environment-sc.yaml
mamba activate online-bs-p100
```

All commands below assume this environment is active and that you are in the
repository root.

---

## 2. Running a single experiment

`main.py` takes **one merged YAML config** via `--config`:

```bash
python main.py --config configs/examples/cifar3_template.yaml
```

Common flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--config <path>` | *(required)* | The merged config to run. |
| `--wandb_not_upload` | off | Keep W&B logs local (offline) instead of uploading. |
| `--experiments_dir <dir>` | `.` | Subdirectory of `./experiments` under which run output directories are created (must be a relative path — runs always land under `./experiments`). |
| `--log_file <name>` | auto | Override the logger filename. |

> There is **no `--seed` flag** — the seed is a top-level key in the config (see
> below). To run multiple seeds, sweep `seed` like any other parameter (see
> "Parameter sweeps" below).

Select GPUs with the standard CUDA variable, e.g. `CUDA_VISIBLE_DEVICES=0` or
`CUDA_VISIBLE_DEVICES="0,2"` (only the listed devices are visible to the run).

### Interactive GPU session

To run interactively on a P100 node:

```bash
salloc -C pascal --time=1:00:00 --ntasks=1 --nodes=1 --gpus=1 --mem=8000M
mamba activate online-bs-p100
python main.py --config configs/examples/makeblobs_antipodal.yaml --wandb_not_upload
```

---

## 3. The config system

A config is a **single merged YAML** with these top-level pieces. Required keys
(`run_name_format`, `seed`) raise a clear error if missing.

```yaml
# How the run is named (see "Run naming" below). Plain strings are literals; $dotted.path
# entries are resolved against this config (a bad path raises).
run_name_format:
  - run
  - $dataset.name
  - $method
  - $training_opt.optimizer
  - lr: [$training_opt.optim_params.lr]   # rendered as e.g. lr-0.001
  - $training_opt.loss_type

seed: 1                      # top-level; drives all RNG (was the old --seed flag)

dataset:                     # name must match a loader in data/__init__.py
  name: CIFAR3
  root: ./_CIFAR
  num_classes: 3
  # ... dataset-specific keys; *_Noise datasets also take noise_percent

networks:                    # the model (models/__init__.py)
  type: ResNet_torchvision
  params: { m_type: 'resnet18' }

training_opt:
  num_epochs: 100
  batch_size: 320
  test_batch_size: 512
  num_data_workers: 4
  optimizer: SGD             # SGD | AdamW | ...
  optim_params: { lr: 0.01, weight_decay: 0.0005 }
  scheduler: 'constant'
  scheduler_params: {}
  loss_type: CrossEntropy
  loss_params: {}

method: Uniform              # Uniform | Full | RhoLoss | DivBS | Bayesian | TrainLoss | GradNorm | GradNormIS
method_opt:
  ratio: 1.0                 # fraction of each batch kept (1.0 = no selection)
  balance: False
  ratio_scheduler: constant
  warmup_epochs: 0

diagnostics:                 # what gets logged (see "Diagnostics" below)
  logging_defaults: { log_interval: logarithmic, save_init: 5, save_freq: 4 }
  diagnostics:               # list; bare name or {Name: params} are both fine
    - TrainLoss
    - TrainAcc
    - ValLoss
    - ValAcc
    - Timing
    - Checkpoint

wandb:                       # passed to wandb.init(); --wandb_not_upload overrides mode
  project: "My Project"
  mode: online               # online | offline | disabled
```

Ready-to-run example configs live in `configs/examples/` (e.g. `cifar3_template.yaml`,
`makeblobs_antipodal.yaml`, `mnist_noise.yaml`) — that subfolder is tracked in
git, so it should only hold examples meant to be shared. Your own configs
(personal sweeps, one-offs, scratch edits) go directly in `configs/`: the rest
of that folder is git-ignored, so anything there stays local. Copy an example
in and edit it rather than editing the example in place.

---

## 4. Run naming & output directories

`run_name_format` is rendered by `build_run_name` (in `run_dir.py`) into a run
name, e.g. `run_CIFAR3_RhoLoss_AdamW_lr-0.001_CrossEntropy`. Separators are
configurable (`value_sep="_"`, `kv_sep="-"`).

Each run gets a **self-contained directory** under `./experiments/<experiments_dir>`:

```
experiments/<experiments_dir>/<timestamp>[_<n>]_<run_name>/
  input_config.yaml      # exact, unmodified config that ran (written once, never touched again)
  run_info.yaml           # input_config.yaml plus runtime-derived fields (save_dir, wandb_run_id,
                          # num_gpus, experiments_dir, ...); rewritten every invocation, so it's a
                          # superset of input_config.yaml that also reflects current state
  logs/                  # per-diagnostic logs + SLURM stdout/stderr links
  snapshots/             # checkpoint.pth.tar (rolling) + model_best.pth.tar
  wandb/                 # local W&B files
  labels -> ...          # symlink into the shared noise-label cache (noisy datasets)
```

The directory is claimed **atomically**; if two runs would collide on the
timestamp, a `_<n>` suffix is inserted right after the timestamp. The W&B run
name is the rendered run name (it need not be unique). Noise labels are cached
lazily on first load under `./cache/labels/` — there is no separate "save labels"
step.

---

## 5. Parameter sweeps

Sweeps use a **template config** plus `generate_configs.py`. A template is a
normal merged config with some leaves set to the sentinel `__REQUIRED__`:

```yaml
# configs/examples/cifar3_template.yaml (excerpt)
seed: __REQUIRED__
training_opt:
  optimizer: __REQUIRED__
  optim_params: { lr: __REQUIRED__ }
```

A submission script fills every `__REQUIRED__` leaf over the Cartesian product of
value lists, writing one concrete config per combination into `./.configs-temp/`.
See `run/examples/run_from_config_template.py` for a working example:

```python
from run_utils import run_job, RunType, generate_configs

TEMPLATE = "configs/examples/cifar3_template.yaml"
PARAMS_TO_VARY = {
    "seed": [1, 2, 3],                       # seed is swept like any other key
    "training_opt.optimizer": ["SGD", "AdamW"],
    "training_opt.optim_params.lr": [0.001, 0.01, 0.1],
}
config_paths = generate_configs(TEMPLATE, PARAMS_TO_VARY)
```

Rules: every key in `PARAMS_TO_VARY` must be `__REQUIRED__` in the template, and
every `__REQUIRED__` leaf must be covered by `PARAMS_TO_VARY` (otherwise it
raises). Generated filenames encode the varied values, e.g.
`..._seed1_methodRhoLoss_....yaml`.

---

## 6. Diagnostics

### 6.1 Config format

The `diagnostics.diagnostics` block is a **list** of leaves to log. Each entry
is either a bare class name (`- TrainLoss`) or a single-key dict with params
(`- MinibatchScoresSummary: {statistic: mean}`). Using a list allows the same
class to appear multiple times with different params (a plain YAML mapping
would silently drop duplicate keys).

### 6.2 How the list is built (`create_diagnostics.py`)

`create_diagnostics(method, ...)` reads `config["diagnostics"]["diagnostics"]`
and, for each entry:

1. Looks up the class name in one of three registries
   (`methods/diagnostics/diagnostics.py`): `POST_BATCH_DIAGNOSTICS`,
   `EPOCH_END_DIAGNOSTICS`, or `TRAIN_END_DIAGNOSTICS` — an unknown name
   raises. Each registry feeds a separate `DiagnosticsManager`, one per
   training phase:
   - **`POST_BATCH`** — driven after every training batch
     (`SelectionMethod.after_batch` → `_run_post_batch_diagnostics`), subject
     to a shared logging schedule (below).
   - **`EPOCH_END`** — driven after every epoch (`SelectionMethod.after_epoch`);
     always runs (`should_run=lambda state: True`), not schedule-gated.
   - **`TRAIN_END`** — driven once, after training finishes
     (`SelectionMethod.after_run` → `diagnostics.finalize()`).
2. Instantiates the diagnostic with its config params plus a `log_path`
   (`logs/<log_name>.log`), where `log_name` is the class name, suffixed
   `_0`, `_1`, … if the same class appears more than once in the list.
3. Registers it with its phase's manager.

Diagnostics form a dependency DAG rather than each computing everything from
scratch: a diagnostic's `_run` calls `.run()` on its dependencies (e.g. a
shared `ForwardPass` or `PerSampleLossError`), and `DiagnosticsManager.build`
(`methods/diagnostics/base.py`) deduplicates — two leaves that request an
equal dependency (same class + same constructor args, via `__eq__`) share one
instance, computed once per `TrainState` and cached (`Diagnostic.run`,
`base.py:97`) rather than once per leaf.

### 6.3 Logging schedule

Post-batch diagnostics share one `LogSchedule` (`methods/diagnostics/schedule.py`),
built from `diagnostics.logging_defaults`:

```yaml
diagnostics:
  logging_defaults:
    log_interval: logarithmic   # or "per_epoch"
    save_init: 5                # first N epochs: log densely (see below)
    save_freq: 4                 # subdivisions per epoch during save_init
```

- **`logarithmic`** (default): logs densely for the first `save_init` epochs
  (every `total_batches // save_freq` batches), then once per epoch through
  epoch 25, every 4th epoch through 65, and every 15th epoch after that (plus
  always the final epoch and step 0).
- **`per_epoch`**: logs only on the last batch of each epoch
  (`state.batch_idx == total_batches - 1`), ignoring `save_init`/`save_freq`.

Epoch-end and train-end diagnostics aren't gated by `LogSchedule` at all —
epoch-end always fires at epoch end, train-end fires once at the very end.

### 6.4 Available leaves

- **`EpochSnapshot`** (epoch end) — saves `{epoch, state_dict}` atomically to
  `snapshots/epochs/epochNNN.pth.tar` for interpolation and other analysis.
  Its independent `scheduler` defaults to `logarithmic`: every epoch through
  25, multiples of 4 through 65, and multiples of 15 thereafter. Use
  `per_epoch` to save every completed epoch. These model-only snapshots do
  not replace the full resume checkpoints written by `Checkpoint`.

  ```yaml
  diagnostics:
    diagnostics:
      - EpochSnapshot:
          scheduler: per_epoch
  ```

  A bare `- EpochSnapshot` uses the default logarithmic schedule. Snapshot
  saving is now opt-in through this diagnostic for all methods, including
  `DivBS_RhoLoss`.

- **Snapshots (post batch):** `TrainLoss`, `TrainAcc`, `ValLoss`, `ValAcc`
  (and `TrueLabelTrainLoss`/`TrueLabelTrainAcc` for clean-label metrics on
  noisy data)
- **`TrainProgress`, `ValProgress`** (post batch) — geodesic progress of
  predictions toward the labels
- **`LogitNormL2`, `ParamNorms`, `GradNorms`, `WeightMatrixNorms`** (post batch)
- **`LinearProbe`, `NTK`** (post batch; heavier, take `params`)
- **`Checkpoint`** (post batch) — rolling + best checkpoint (needed to resume
  / track best acc)
- **`TrainingState`** (post batch) — logs `epoch`/`batch_idx`/`total_steps`
- **`MinibatchScoresSummary`** (post batch) — summary of the current
  minibatch's selection scores
- **`PerSampleProgressSummary`, `PerSampleVolatilitySummary`** (post batch,
  DPD diagnostics) — summary of per-sample prediction progress / volatility
  on a held-out loader
- **`Timing`** (epoch end) — wall-clock `total_time` / `time_this_epoch`
- **`SelectedPoints`, `SelectedPointsSummary`** (epoch end) — noisy-selection
  stats
- **`WStarTestAcc`, `WHatTestAcc`, `BayesAccAntipodalGaussian`** (train end)
  — final learned-parameter-recovery metrics (antipodal-Gaussian setups)

A metric's W&B key can be overridden, e.g. on a noisy dataset:

```yaml
- TrainLoss:
    log_key: noisy_train_loss
```

Summary diagnostics (e.g. `MinibatchScoresSummary`, `PerSampleProgressSummary`,
`PerSampleVolatilitySummary`) reduce a wrapped diagnostic's values via a named
`statistic` (`mean`, `median`, `min`, `max`, `std`). `statistic` can be a
single name or a list, logging one key per statistic as `<log_key>_<stat>`:

```yaml
- MinibatchScoresSummary:
    statistic: [mean, median, std, min, max]
```

is equivalent to five separate `MinibatchScoresSummary` entries, one per
statistic, logged as `MinibatchScoresSummary_mean`,
`MinibatchScoresSummary_median`, etc.

---

## 7. Submitting batch jobs to SLURM

Tracked example submission scripts live in **`run/examples/`**; they
generate/select configs and submit one job each via `run_utils.run_job`
(`run/run_utils/`, also tracked since the examples import it). **Copy one into `run/` first**, then run it from there:

```bash
cp run/examples/run_from_configs.py run/
python run/run_from_configs.py
```

Both examples default to `RunType.DRY` (prints what would run without
submitting) — change the `RUN_TYPE` constant at the top of the copied script
to `NORMAL`, `SBATCH`, or `SRUN` to actually run.

For your own ad-hoc/WIP sweeps, likewise copy a config from **`configs/examples/`**
into **`configs/`** itself and edit from there. `run/` and `configs/` are
tracked folders, but everything in them is git-ignored except the `examples/`
(and `run/run_utils/`) subfolders, so personal scripts and configs stay local
while the examples they were copied from stay clean for others to reuse.

Each submitted job requests a GPU and `--requeue`, so a preempted job lands back
in the **same** run directory and resumes from its rolling checkpoint. SLURM
stdout/stderr go to `logs/slurm/%j.{out,err}` and are symlinked into the run
dir.

To resume / extend a finished or interrupted run, either point a config's
`resume.from` at an existing run directory (optionally with
`resume.additional_epochs`) and submit it as usual, or use
`run_utils.extend_job`, which builds that config for you from the run's saved
`input_config.yaml`:

```python
from run_utils import extend_job, RunType

extend_job(
    "experiments/20260728_114119_MNIST_DivBS_LeNet_0.1_1",
    additional_epochs=20,
    overrides={"training_opt.optim_params.lr": 1e-5},  # optional dotted-path tweaks
    run_type=RunType.SBATCH,
    preemptible=True,
)
```

`extend_job` takes the experiment directory in place of `run_job`'s config
path; everything else (`*args`/`**kwargs`) passes straight through to
`run_job`.

---

## 8. Repository layout

- **`main.py`** — entry point: loads the config, claims a run dir, runs training.
- **`methods/SelectionMethod.py`** — base training loop; subclasses implement
  `before_batch` to select/weight each minibatch. Methods: Full, Uniform, DivBS,
  RhoLoss, Bayesian, TrainLoss, GradNorm, GradNormIS.
- **`methods/diagnostics/`** — the diagnostics framework (leaves, managers, NTK,
  probes, model metrics).
- **`data/`** — dataset loaders; `dataset.name` must match a function in
  `data/__init__.py` (CIFAR3/10/100, MNIST/FashionMNIST, TinyImageNet, MakeBlobs,
  Teacher_Generated, and `*_Noise` variants).
- **`models/`** — model definitions (ResNet, LeNet, Linear, DeepLinear, TwoLayer).
- **`configs/examples/`** — tracked ready-to-run configs and sweep templates;
  rest of **`configs/`** — your local/WIP configs (git-ignored). Copy an
  example out of `examples/` before editing.
- **`run_dir.py`** — run-name rendering, atomic run-dir creation, resume plumbing.
- **`run/run_utils/generate_configs.py`** — template → concrete configs for sweeps.
- **`run/examples/`** and **`run/run_utils/`** — tracked example SLURM
  submission scripts and the package they import; rest of **`run/`** — your
  local/WIP submission scripts (git-ignored). Copy an example out of
  `examples/` into `run/` before editing/running it.
- **`experiments/`** — run outputs (git-ignored).

---

## 9. Data preparation

- **CIFAR / MNIST / FashionMNIST** are downloaded automatically on first use
  (into `./_CIFAR`, `./_MNIST`, …).
- **Tiny-ImageNet:** download from
  [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip), unzip into
  `_TINYIMAGENET`, then `cd _TINYIMAGENET && python val_folder.py`.
- **MakeBlobs / Teacher_Generated** are synthetic and generated on demand.

---

## 10. Adding a new diagnostic

### Concepts

The diagnostics system is a **dependency DAG** of `Diagnostic` objects. Each
diagnostic implements `_run()` and returns a `DiagnosticInfo`. Results are
cached per `TrainState` (identified by `(total_steps, phase)`), so shared
intermediate work — e.g. a forward pass — is computed once and reused by any
number of leaf diagnostics that depend on it.

There are two kinds of diagnostics:

| Kind | Logged? | Registered with manager? | Phase |
|------|---------|--------------------------|-------|
| **Compute dependency** (e.g. `ForwardPass`) | No | No | — |
| **Logged leaf** (e.g. `TrainLoss`) | Yes | Yes | `POST_BATCH` or `EPOCH_END` |

Context available inside `_run()` via `self.get_context()`:

| Key | Type | Notes |
|-----|------|-------|
| `model` | `nn.Module` | The current model (in eval or train mode) |
| `device` | `torch.device` | |
| `fixed_train_loader` | DataLoader | Fixed-order train loader for eval passes |
| `test_loader` | DataLoader | Validation loader |
| `num_train_samples` | int | |
| `num_classes` | int | |
| `noisy_indices` | array or None | Set on noisy datasets |
| `true_labels` | Tensor or None | Clean labels on noisy datasets |
| `selected_mask` | bool array | Set each epoch by the training loop (epoch-end only) |
| `total_time` / `time_this_epoch` | float | Set by `after_epoch` (epoch-end only) |
| `checkpoint_state` | dict or None | Full checkpoint dict; set by the training loop |
| `save_dir` | str | Run output directory |

`self.get_state()` returns the current `TrainState` with `.epoch`,
`.batch_idx`, `.total_steps`, `.total_epochs`, `.total_batches`.

### Step 1 — write the class

Place it in `methods/diagnostics/standard.py` (or `model_metrics.py` for
model-weight/gradient diagnostics).

**Minimal logged leaf** (fires post-batch, reads from context):

```python
class MyMetric(Diagnostic):
    def __init__(self, manager, builder, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        # build any dependencies via builder.build(SomeDep, manager, ...)

    def _run(self):
        ctx = self.get_context()
        value = float(ctx["some_key"])          # or run a dep: self.dep.run().info
        return DiagnosticInfo("my_metric", {"my_metric": value})

    def __eq__(self, other):
        return isinstance(other, MyMetric)      # required for deduplication
```

Rules:
- `__init__` signature must be `(self, manager, builder, should_run=None, **params)`.
  `params` carries YAML `params:` keys plus `log_path` (injected automatically).
- `_run` must return a `DiagnosticInfo(name, info)` where `info` is either a
  scalar or a flat `dict` of scalars (both are logged to W&B by the default
  `_log_payload`).
- `__eq__` must be implemented if you intend to create multiple instances of the same diagnostics (with different parameters, for example). Two diagnostics that are equal share a single
  instance (via `DiagnosticsBuilder`), so a dependency created by two different
  leaves is computed only once.

**Compute dependency** (not logged, built via `builder.build`):

```python
class MyExpensivePrep(Diagnostic):
    def __init__(self, manager, some_arg):
        super().__init__(manager)       # no log_path, no should_run
        self.some_arg = some_arg

    def _run(self):
        result = ...                    # heavy computation
        return DiagnosticInfo("my_prep", {"key": result})

    def __eq__(self, other):
        return isinstance(other, MyExpensivePrep) and self.some_arg == other.some_arg
```

Leaves acquire it with `self.example_dependency = builder.build(MyExpensivePrep, manager, some_arg)`
and read it in `_run` with `self.example_dependency.run().info["key"]`.

**Epoch-end diagnostic** — identical to a post-batch leaf but uses
`should_run=(lambda state: True)` (applied automatically by the wiring; see Step
2). Reads epoch-level context keys like `selected_mask`.

### Step 2 — register it

At the bottom of `methods/diagnostics/diagnostics.py`, add your class to the
appropriate registry dict:

```python
POST_BATCH_DIAGNOSTICS = {
    ...
    "MyMetric": MyMetric,          # fires post-batch, on the schedule
}
EPOCH_END_DIAGNOSTICS = {
    ...
    "MyEpochMetric": MyEpochMetric,  # fires every epoch-end
}
```

### Step 3 — enable it in a config

```yaml
diagnostics:
  logging_defaults: { log_interval: logarithmic, save_init: 5, save_freq: 4 }
  diagnostics:
    MyMetric: {}                    # no params needed
    MyMetric: { params: { log_path: logs/my_metric.log } }  # optional override
    MyMetric:                       # custom logging schedule for this leaf only
      logging: { log_interval: linear, save_freq: 1 }
```

Any key under `params:` is forwarded as a keyword argument to `__init__`. The
`log_path` key is always injected automatically (defaults to
`<save_dir>/logs/<DiagnosticName>.log`) and can be overridden here.

---

## 11. Optional: TRAK for projection-NTK diagnostics

The projection NTK variants (`proj-pseudo`, `proj-trace`) need `TRAKer`:

```bash
pip install traker
# or the PNNL implementation:
pip install "git+https://github.com/pnnl/projection_ntk.git"
python -c "from trak import TRAKer; print('TRAKer import OK')"
```
