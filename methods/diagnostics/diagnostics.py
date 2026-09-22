"""Diagnostic registry. Maps config class names to constructors.

Definitions live in standard.py (and model_metrics.py / ntk.py).
"""

from methods.diagnostics.standard import *
from methods.diagnostics.model_metrics import *
from methods.diagnostics.ntk import *
from methods.diagnostics.specialty import *
from methods.diagnostics.delayed_prediction_dynamics import *

POST_BATCH_DIAGNOSTICS = {
    "TrainLoss": TrainLoss,
    "TrainAcc": TrainAcc,
    "ValLoss": ValLoss,
    "ValAcc": ValAcc,
    "TrueLabelTrainLoss": TrueLabelTrainLoss,
    "TrueLabelTrainAcc": TrueLabelTrainAcc,
    "LogitNormL2": LogitNormL2,
    "TrainProgress": TrainProgress,
    "ValProgress": ValProgress,
    "ParamNorms": ParamNorms,
    "GradNorms": GradNorms,
    "WeightMatrixNorms": WeightMatrixNorms,
    "LinearProbe": LinearProbe,
    "NTK": NTK,
    "Checkpoint": Checkpoint,
    "MinibatchScoresSummary": MinibatchScoresSummary,
    "TrainingState": TrainingState,
    "ProjectionProgressSummary": ProjectionProgressSummary,
    "PerSampleVolatilitySummary": PerSampleVolatilitySummary,
    "LogProbs": LogProbs
}
EPOCH_END_DIAGNOSTICS = {
    "EpochSnapshot": EpochSnapshot,
    "SelectedPoints": SelectedPoints,
    "SelectedPointsSummary": SelectedPointsSummary,
    "Timing": Timing,
}
TRAIN_END_DIAGNOSTICS = {
    "WStarTestAcc": WStarTestAcc,
    "WHatTestAcc": WHatTestAcc,
    "BayesAccAntipodalGaussian": BayesAccAntipodalGaussian,
}
