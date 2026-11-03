"""facecheck: audit EEG decoders for split leakage and facial-artifact leakage."""
from .core import audit, face_indices, report, verdicts

__all__ = ["audit", "face_indices", "report", "verdicts"]
__version__ = "0.1.0"
