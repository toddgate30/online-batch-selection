"""Logging schedule.

Ports the old `DiagnosticsLogger._build_logarithmic_steps` / `should_log` into
a small object usable as a diagnostic's `should_run(state)` predicate. The
`logarithmic` schedule logs densely early (sub-epoch) and sparsely later;
`per_epoch` logs only at the last batch of each epoch.
"""

import numpy as np


class EpochSchedule:
    """Epoch-end schedule using the same names as LogSchedule.

    Logarithmic snapshots use completed, one-based epochs: every epoch through
    25, every fourth through 65, and every fifteenth thereafter. Unlike the
    post-batch schedule, there are no sub-epoch snapshots or initial snapshot.
    """

    def __init__(self, log_interval="logarithmic"):
        if log_interval not in {"logarithmic", "per_epoch"}:
            raise ValueError(
                f"Unknown epoch scheduler {log_interval!r}; expected 'logarithmic' or 'per_epoch'."
            )
        self.log_interval = log_interval

    def __call__(self, state):
        epoch = state.epoch
        if epoch < 1:
            return False
        if self.log_interval == "per_epoch":
            return True
        return epoch <= 25 or (epoch <= 65 and epoch % 4 == 0) or (epoch > 65 and epoch % 15 == 0)


class LogSchedule:
    def __init__(self, total_batches, num_epochs=None, num_steps=None,
                 log_interval="logarithmic", save_init=5, save_freq=4):
        self.total_batches = int(total_batches)
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.save_init = int(save_init)
        self.save_freq = int(save_freq)
        self.last_batch_idx = self.total_batches - 1
        if log_interval not in {"logarithmic", "per_epoch"}:
            raise NotImplementedError(f"{log_interval=} has not been implement. Please review methods/diagnostics/schedule.py")
        self.log_interval = log_interval
        self.steps = self._build_logarithmic_steps() if log_interval == "logarithmic" else None

    def _build_logarithmic_steps(self):
        total_epochs = self.num_epochs or int(np.ceil(self.num_steps / self.total_batches))
        intra_epoch_stride = max(self.total_batches // self.save_freq, 1)

        t = 0
        steps = [0]
        for epoch in range(total_epochs):
            for batch_idx in range(self.total_batches):
                t += 1
                if epoch < self.save_init and batch_idx % intra_epoch_stride == 0:
                    steps.append(t)
            if self.save_init <= epoch <= 25:
                steps.append(t)
            elif 25 < epoch <= 65 and epoch % 4 == 0:
                steps.append(t)
            elif (epoch > 65 and epoch % 15 == 0) or (epoch == total_epochs - 1):
                steps.append(t)

        if self.num_steps is not None:
            steps = [step for step in steps if step <= self.num_steps]
        return set(steps)

    def __contains__(self, state):
        """Allow `state in schedule` and use as `should_run`."""
        if self.log_interval == "per_epoch":
            return (state.batch_idx % state.total_batches) == self.last_batch_idx
        return state.total_steps in self.steps

    def __call__(self, state):
        return self.__contains__(state)
