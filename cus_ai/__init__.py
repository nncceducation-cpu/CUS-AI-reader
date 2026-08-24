"""CUS AI Reader research prototype."""

from .aggregation import AggregationConfig, aggregate_labels
from .ai_consensus import grade_prediction
from .calibration import CalibrationSet
from .clinical import classify_study
from .consistency import enforce_consistency

__version__ = "0.6.0"

__all__ = [
    "AggregationConfig",
    "CalibrationSet",
    "aggregate_labels",
    "classify_study",
    "enforce_consistency",
    "grade_prediction",
    "__version__",
]
