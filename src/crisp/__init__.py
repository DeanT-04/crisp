"""crisp: AI-free upscaler for technical drawings."""

from crisp.batch import process_batch
from crisp.pipeline import process
from crisp.types import Options, TargetSpec

__version__ = "0.1.0"

__all__ = ["Options", "TargetSpec", "__version__", "process", "process_batch"]
