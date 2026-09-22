"""Concrete diagnostics, decomposed into cached compute dependencies
and small logged leaves wired by the dependency mechanism.

Compute deps, unregistered, never logged:
  ForwardPass(loader)        -- one model pass over a loader, shared by all metrics
  PerSampleLossError(loader, label_source)
Logged leaves, registered with a manager:
  TrainLoss/TrainAcc/ValLoss/ValAcc, TrueLabel*{Loss,Acc}, LogitNormL2,
  Checkpoint, SelectedPoints, TrainingState, Timing
"""

import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from methods.diagnostics.base import Diagnostic, DiagnosticInfo, LogType, Summary, _SUMMARY_STATS
from methods.diagnostics.schedule import EpochSchedule
from run_dir import atomic_save

# --------------------------------------------------------------------------- #
# Layer 1: cached compute dependencies (built via manager.build, never registered)
# --------------------------------------------------------------------------- #

class ForwardPass(Diagnostic):
    """One eval pass over `loader_key` ('train'|'val'); returns per-sample
    log-probs, logit L2 norms, predictions, loader targets, and indices."""

    def __init__(self, manager, loader_key):
        super().__init__(manager)
        self.loader_key = loader_key

    def _run(self):
        model = self.method.model
        device = next(model.parameters()).device
        loader = self.method.fixed_train_loader if self.loader_key == "train" else self.method.test_loader
        model.eval()
        log_probs, logit_l2, preds, targets, indices = [], [], [], [], []
        with torch.no_grad():
            for datas in loader:
                inputs = datas["input"].to(device)
                tgt = datas["target"].to(device)
                logits = model(inputs)
                log_probs.append(F.log_softmax(logits, dim=1).cpu())
                logit_l2.append(torch.norm(logits, p=2, dim=1).cpu())
                preds.append(torch.argmax(logits, dim=1).long().cpu())
                targets.append(tgt.long().cpu())
                idx = datas["index"]
                idx = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx, dtype=torch.long)
                indices.append(idx.long().cpu())
        return DiagnosticInfo(f"forward_{self.loader_key}", {
            "log_probs": torch.cat(log_probs),
            "logit_l2": torch.cat(logit_l2),
            "predictions": torch.cat(preds),
            "targets": torch.cat(targets),
            "indices": torch.cat(indices),
        })

    def __eq__(self, other):
        return isinstance(other, ForwardPass) and self.loader_key == other.loader_key


class PerSampleLossError(Diagnostic):
    """Per-sample loss and 0/1 error for `loader_key` against `label_source`
    ('loader' = the loader's own (possibly noisy) labels, 'true' = clean labels)."""

    def __init__(self, manager, loader_key, label_source):
        super().__init__(manager)
        self.loader_key = loader_key
        self.label_source = label_source
        self.forward_pass = manager.build(ForwardPass, manager, loader_key)

    def _run(self):
        fp = self.forward_pass.run().info
        log_probs, predictions = fp["log_probs"], fp["predictions"]
        if self.label_source == "true":
            true_labels = _to_long_vector(self.method.data_info.get("true_labels"))
            if true_labels is None:
                raise ValueError("PerSampleLossError(label_source='true') needs true_labels in context.")
            labels = true_labels[fp["indices"]]
        else:
            labels = fp["targets"]
        loss = -torch.gather(log_probs, 1, labels.view(-1, 1)).squeeze(1)
        error = (labels != predictions).float()
        return DiagnosticInfo(f"loss_error_{self.loader_key}_{self.label_source}",
                              {"loss": loss, "error": error})

    def __eq__(self, other):
        return (isinstance(other, PerSampleLossError)
                and self.loader_key == other.loader_key
                and self.label_source == other.label_source)


# --------------------------------------------------------------------------- #
# Layer 2: logged leaves
# --------------------------------------------------------------------------- #

class _LossErrorLeaf(Diagnostic):
    """Shared base for the mean-loss / mean-acc leaves. Subclasses MUST set all
    four class attrs below; `log_key` may also be overridden per-run via the
    config's diagnostics `params`."""
    loader_key = None      # 'train' (fixed_train_loader) | 'val' (test_loader)
    label_source = None    # 'loader' (loader's own labels) | 'true' (clean labels)
    metric = None          # 'loss' (mean NLL) | 'acc' (1 - mean 0/1 error)
    log_key = None         # W&B metric name (the info dict key) for this leaf

    _REQUIRED = ("loader_key", "label_source", "metric", "log_key")

    def __init__(self, manager, should_run=None, **params):
        # Config may override the displayed metric name.
        if params.get("log_key") is not None:
            self.log_key = params["log_key"]
        for attr in self._REQUIRED:
            if getattr(self, attr) is None:
                raise TypeError(
                    f"{type(self).__name__} must set class attr '{attr}' "
                    f"(it is still None on the abstract _LossErrorLeaf base)."
                )
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self.dep = manager.build(PerSampleLossError, manager,
                                 self.loader_key, self.label_source)

    def _run(self):
        info = self.dep.run().info
        if self.metric == "loss":
            value = float(info["loss"].mean().item())
        else:
            value = float(1.0 - info["error"].mean().item())
        return DiagnosticInfo(self.log_key, {self.log_key: value})


class TrainLoss(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "train", "loader", "loss", "train_loss"

class TrainAcc(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "train", "loader", "acc", "train_acc"

class ValLoss(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "val", "loader", "loss", "val_loss"

class ValAcc(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "val", "loader", "acc", "val_acc"

class TrueLabelTrainLoss(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "train", "true", "loss", "train_loss_true_labels"

class TrueLabelTrainAcc(_LossErrorLeaf):
    loader_key, label_source, metric, log_key = "train", "true", "acc", "train_acc_true_labels"


class LogitNormL2(Diagnostic):
    """Mean L2 norm of train/val logits."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self.train_fp = manager.build(ForwardPass, manager, "train")
        self.val_fp = manager.build(ForwardPass, manager, "val")

    def _run(self):
        train = float(self.train_fp.run().info["logit_l2"].mean().item())
        val = float(self.val_fp.run().info["logit_l2"].mean().item())
        return DiagnosticInfo("logit_norm_l2", {
            "diagnostics/train_logits_norm_l2_mean": train,
            "diagnostics/test_logits_norm_l2_mean": val,
        })

    def __eq__(self, other):
        return isinstance(other, LogitNormL2)


class _ProgressLeaf(Diagnostic):
    """Shared base for the train/val geodesic-progress leaves. Subclasses
    MUST set loader_key and log_key; log_key may be overridden per-run via
    the config's diagnostics params."""
    loader_key = None      # 'train' (fixed_train_loader) | 'val' (test_loader)
    log_key = None          # W&B metric name

    _REQUIRED = ("loader_key", "log_key")

    def __init__(self, manager, should_run=None, **params):
        if params.get("log_key") is not None:
            self.log_key = params["log_key"]
        for attr in self._REQUIRED:
            if getattr(self, attr) is None:
                raise TypeError(
                    f"{type(self).__name__} must set class attr '{attr}' "
                    f"(it is still None on the abstract _ProgressLeaf base)."
                )
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        self.forward_pass = manager.build(ForwardPass, manager, self.loader_key)

    def _run(self):
        fp = self.forward_pass.run().info
        value = _geodesic_progress(fp["log_probs"], fp["targets"])
        return DiagnosticInfo(self.log_key, {self.log_key: value})

    def __eq__(self, other):
        return isinstance(other, type(self))


class TrainProgress(_ProgressLeaf):
    loader_key, log_key = "train", "train_progress"

class ValProgress(_ProgressLeaf):
    loader_key, log_key = "val", "val_progress"

class LogProbs(Diagnostic):
    """Save train and validation log-probability snapshots locally.

    The saved snapshot has the format:

        {
            "yh":  <train log-probabilities>,
            "yvh": <validation log-probabilities>
        }

    Both values are stored as NumPy arrays.
    """

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run,)
        self.train_forward_pass = manager.build(ForwardPass, manager, "train",)
        self.val_forward_pass = manager.build(ForwardPass, manager, "val",)
        self.training_state = manager.build(TrainingState, manager, **params)
        self.progress_snapshots_dir = os.path.join(self.method.config["save_dir"], "logs/log_probs")
        os.makedirs(self.progress_snapshots_dir, exist_ok=True)
    

    def _run(self):
        train_fp = self.train_forward_pass.run().info
        val_fp = self.val_forward_pass.run().info
        model_state = self.training_state.run().info

        train_log_probs = train_fp["log_probs"]
        val_log_probs = val_fp["log_probs"]
        total_steps = model_state["total_steps"]

        train_log_probs = (train_log_probs.detach().cpu())
        val_log_probs = (val_log_probs.detach().cpu())

        current_method = getattr(self.method, "current_method", type(self.method).__name__)

        snapshot = {"method": current_method,"step": total_steps, "yh": train_log_probs, "yvh": val_log_probs}
        self.checkpoint_path = os.path.join(self.progress_snapshots_dir, f"log_probs_step_{total_steps}.pth.tar")
        atomic_save(lambda p: torch.save(snapshot, p), self.checkpoint_path)

        # I'm not sure if we really need this to return anything, but I set it to 0 and will see if it logs to wandb.
        return DiagnosticInfo("progress_snapshot", 0,)   


class Checkpoint(Diagnostic):
    """Rolling checkpoint + best-model tracking. Writes the rolling
    `snapshots/checkpoint.pth.tar` atomically every time it fires and copies
    `model_best.pth.tar` when val accuracy improves."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, should_run=should_run)
        self.val_acc = manager.build(PerSampleLossError, manager, "val", "loader")
        self.snapshots_dir = os.path.join(self.method.config["save_dir"], "snapshots")
        self.checkpoint_path = os.path.join(self.snapshots_dir, "checkpoint.pth.tar")
        self.best_path = os.path.join(self.snapshots_dir, "model_best.pth.tar")
        self.save_best = bool(params.get("save_best", True))
        self.best_acc = float(self.method.best_acc)
        self.best_epoch = int(self.method.best_epoch)
        self.is_best = False

    def _run(self):
        state = self.get_state()
        val_acc = float(1.0 - self.val_acc.run().info["error"].mean().item())
        self.is_best = val_acc > self.best_acc
        if self.is_best:
            self.best_acc = val_acc
            self.best_epoch = int(state.epoch)

        checkpoint_state = self.method._current_checkpoint_state
        if checkpoint_state is not None:
            checkpoint_state = dict(checkpoint_state)
            checkpoint_state["best_acc"] = self.best_acc
            checkpoint_state["best_epoch"] = self.best_epoch
            atomic_save(lambda p: torch.save(checkpoint_state, p), self.checkpoint_path)
            if self.is_best and self.save_best:
                shutil.copyfile(self.checkpoint_path, self.best_path)

        return DiagnosticInfo("checkpoint", {"best_val_acc": self.best_acc})

    def __eq__(self, other):
        return isinstance(other, Checkpoint)

class EpochSnapshot(Diagnostic):
    """Save epoch/model snapshots for analysis, independently of Checkpoint.

    scheduler accepts 'logarithmic' (default), 'per_epoch', or a callable
    taking TrainState. These are model snapshots, not full resume checkpoints.
    """

    def __init__(self, manager, should_run=None, scheduler="logarithmic", **params):
        self.scheduler = EpochSchedule(scheduler) if isinstance(scheduler, str) else scheduler
        if not callable(self.scheduler):
            raise TypeError("EpochSnapshot scheduler must be a schedule name or callable.")
        super().__init__(
            manager,
            log_path=params.get("log_path"),
            should_run=lambda state: self.scheduler(state) and (should_run is None or should_run(state)),
        )
        self.snapshots_dir = os.path.join(self.method.config["save_dir"], "snapshots", "epochs")

    def _run(self):
        epoch = self.get_state().epoch
        os.makedirs(self.snapshots_dir, exist_ok=True)
        path = os.path.join(self.snapshots_dir, f"epoch{epoch:03d}.pth.tar")
        snapshot = {"epoch": epoch, "state_dict": self.method.model.state_dict()}
        atomic_save(lambda p: torch.save(snapshot, p), path)
        return DiagnosticInfo("epoch_snapshot", {"epoch": epoch, "path": path}, log_type=LogType.FILEONLY)


class SelectedPoints(Diagnostic):
    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, should_run=should_run)
        self.noisy_indices = _to_numpy_indices(self.method.data_info.get("noisy_indices"))
        self.num_train_samples = int(self.method.num_train_samples)
        self._masks = []

    def _run(self):
        mask = self.method._epoch_selected_mask
        if mask is not None:
            m = mask.cpu().numpy().astype(bool) if isinstance(mask, torch.Tensor) else np.asarray(mask, dtype=bool)
            self._masks.append(m)
        return DiagnosticInfo("selected_points_mask", mask, log_type=LogType.FILEONLY)

    def finalize(self):
        if not self._masks:
            return
        arr = np.stack(self._masks, axis=0)  # (num_epochs, num_train_samples)
        np.save(os.path.join(self.method.config["save_dir"], "selected_points.npy"), arr)

    def __eq__(self, other):
        return isinstance(other, SelectedPoints)


class SelectedPointsSummary(Diagnostic):
    """Epoch-end noisy-selection statistics. Reads the epoch's selected-point
    mask from `method._epoch_selected_mask` and the noisy indices / train size
    from `method`."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, should_run=should_run)
        self.noisy_indices = _to_numpy_indices(self.method.data_info.get("noisy_indices"))
        self.num_train_samples = int(self.method.num_train_samples)
        self.selected_points = manager.build(SelectedPoints, manager)

    def _run(self):
        mask = self.selected_points.run().info
        num_noisy_selected = int(mask[self.noisy_indices].sum())
        total_noisy = int(self.noisy_indices.size)
        return DiagnosticInfo("selected_points_statistics", {
            "num noisy points selected": num_noisy_selected,
            "percent of batch with label noise": num_noisy_selected / self.num_train_samples,
            "percent of points with label noise selected":
                (num_noisy_selected / total_noisy) if total_noisy > 0 else 0.0
        })

    def __eq__(self, other):
        return isinstance(other, SelectedPointsSummary)


class Timing(Diagnostic):
    """Epoch-end wall-clock telemetry: cumulative training time and the time
    spent in the most recent epoch, read from the manager's shared context
    (populated by `SelectionMethod.after_epoch`)."""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)

    def _run(self):
        return DiagnosticInfo("timing", {
            "total_time": float(self.method.total_time),
            "time_this_epoch": float(self.method.time_this_epoch),
        })

    def __eq__(self, other):
        return isinstance(other, Timing)

class TrainingState(Diagnostic):
    """Logs the current epoch, current step, and batch index"""

    def __init__(self, manager, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
    
    def _run(self):
        state = self.get_state()
        return DiagnosticInfo("training_state", {
            "epoch": state.epoch,
            "batch_idx": state.batch_idx,
            "total_steps": state.total_steps,
        })
        
    def __eq__(self, other):
        return isinstance(other, TrainingState)

class MinibatchScoreValues(Diagnostic):
    """Dependency-only: exposes `method._last_minibatch_scores` as `info`."""

    def __init__(self, manager):
        super().__init__(manager)

    def _run(self):
        scores = self.method._last_minibatch_scores
        if scores is None:
            epoch = self.manager.current_state.epoch
            if epoch > self.method.start_epoch:
                self.method.logger.info(f"WARNING: scores were found to be None in MinibatchScoreValues diagnostic in epoch {epoch}")
        return DiagnosticInfo("minibatch_scores", scores)

    def __eq__(self, other):
        return isinstance(other, MinibatchScoreValues)


class MinibatchScoresSummary(Summary):
    dependency_cls = MinibatchScoreValues
    info_key = None


def _geodesic_progress(log_probs, labels):
    """
    Geodesic interpolation parameter between the uniform (ignorance) point and
    the one-hot ground truth on the unit sphere of sqrt-probabilities.

    log_probs: tensor of shape (num_samples, num_classes). Each row is the
    output distribution of the model on the corresponding point

    labels: int tensor of shape (num_samples). Contains ground truth labels for
    each data point as indices.
    """
    from scipy import optimize

    if log_probs.numel() == 0:
        return 0.0

    probabilities = torch.exp(log_probs.detach().cpu())
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    predictions_on_sphere = np.sqrt(probabilities.numpy().astype(np.float64, copy=False))

    label_indices = labels.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    num_samples, num_classes = predictions_on_sphere.shape
    if label_indices.size != num_samples:
        raise ValueError(f"labels length {label_indices.size} does not match predictions {num_samples}.")

    valid_labels = (label_indices >= 0) & (label_indices < num_classes)
    if not np.all(valid_labels):
        raise ValueError("Found label outside valid class range when computing progress.")

    one_hot_labels = np.zeros((num_samples, num_classes), dtype=np.float64)
    one_hot_labels[np.arange(num_samples), label_indices] = 1.0
    ground_truth = np.sqrt(one_hot_labels)
    ignorance = np.sqrt(np.full((num_samples, num_classes), 1.0 / float(num_classes), dtype=np.float64))

    eps = 1e-8
    ignorance_ground_truth_cosine = np.clip((ignorance * ground_truth).sum(axis=1), 0.0, 1.0)
    ignorance_predictions_cosine = np.clip((ignorance * predictions_on_sphere).sum(axis=1), 0.0, 1.0)
    ground_truth_predictions_cosine = np.clip((ground_truth * predictions_on_sphere).sum(axis=1), 0.0, 1.0)
    ignorance_ground_truth_angle = np.arccos(ignorance_ground_truth_cosine)
    degenerate_mask = ignorance_ground_truth_angle < eps

    degenerate_distance = float(np.arccos(ignorance_predictions_cosine[degenerate_mask]).sum()) if np.any(degenerate_mask) else 0.0
    if np.all(degenerate_mask):
        return 0.0

    non_degenerate_angle = ignorance_ground_truth_angle[~degenerate_mask]
    non_degenerate_ignorance_predictions_cosine = ignorance_predictions_cosine[~degenerate_mask]
    non_degenerate_ground_truth_predictions_cosine = ground_truth_predictions_cosine[~degenerate_mask]
    non_degenerate_angle_sin = np.sin(non_degenerate_angle)

    def objective_fn(t):
        geodesic_cosine = (
            non_degenerate_ignorance_predictions_cosine * np.sin((1.0 - t) * non_degenerate_angle) / non_degenerate_angle_sin
            + non_degenerate_ground_truth_predictions_cosine * np.sin(t * non_degenerate_angle) / non_degenerate_angle_sin
        )
        geodesic_cosine = np.clip(geodesic_cosine, 0.0, 1.0)
        return degenerate_distance + float(np.arccos(geodesic_cosine).sum())

    lam = optimize.minimize_scalar(objective_fn, bounds=(0.0, 1.0), method="bounded").x
    return float(np.clip(lam, 0.0, 1.0))


def _to_long_vector(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().long().reshape(-1)
    return torch.as_tensor(values, dtype=torch.long).reshape(-1)


def _to_numpy_indices(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    return np.asarray(values, dtype=np.int64).reshape(-1)


