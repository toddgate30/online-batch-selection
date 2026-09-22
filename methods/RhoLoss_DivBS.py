import numpy as np
import torch
import torch.nn.functional as F
import time

from methods.method_utils.build_teacher_model import build_teacher_model
from methods.method_utils.loss import *
from methods.method_utils.optimizer import *
from methods.SelectionMethod import MinibatchInfo
from methods.SelectionMethod import SelectionMethod
from methods.DivBS import DivBS
from methods.RhoLoss import RhoLoss
from methods.diagnostics.standard import ValProgress

class RhoLoss_DivBS(RhoLoss, DivBS):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.progress_diag = self.diagnostics.post_batch_manager.build(ValProgress, self.diagnostics.post_batch_manager)
        self.progress_threshold = config['method_opt']['progress_threshold']
        self.current_method = "RhoLoss"

    def after_epoch(self):
        self.diagnostics.run_epoch_end(
            total_steps=self.total_step,
            epoch=self._current_epoch,
            total_epochs=self.epochs,
        )
        if self.current_method == "RhoLoss":
            self.progress_diag.run()
            val_progress = self.progress_diag.last_run_diagnostic.info['val_progress']
            if val_progress >= self.progress_threshold:
                self.current_method = "DivBS"
        return

    def before_batch(self, i, metabatch_inputs, metabatch_targets, metabatch_indexes):
        if self.current_method == "RhoLoss":
            return RhoLoss.before_batch(self, i, metabatch_inputs, metabatch_targets, metabatch_indexes)
        else:
            return DivBS.before_batch(self, i, metabatch_inputs, metabatch_targets, metabatch_indexes)