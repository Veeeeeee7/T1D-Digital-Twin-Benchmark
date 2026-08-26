"""Central quiet/verbose logging control for the benchmark runners."""
from __future__ import annotations

import os
import warnings

_VERBOSE = False

# Known-noisy warning message patterns (regex) from the sbi / torch / gym stack and
# the ReplayBG soft ordering-constraint check.
_NOISY = (
    r".*pkg_resources is deprecated.*",
    r".*CUDA initialization.*",
    r".*No prior bounds were passed.*",
    r".*Prior is lacking (mean|variance) attribute.*",
    r".*ka2.*kd.*",                       # ~9.5% of patients trip this soft constraint
    r".*MAP fit violated ordering.*",     # same soft constraint on the dataset-build MAP fit
)


def is_verbose() -> bool:
    """True if the current process is in verbose mode (progress bars on)."""
    return _VERBOSE


def quiet(verbose: bool = False) -> None:
    """Configure verbosity for this process. Call once, early, in a runner's main."""
    global _VERBOSE
    _VERBOSE = bool(verbose)
    if verbose:
        os.environ["T1D_VERBOSE"] = "1"      # children inherit verbose too
        os.environ.pop("T1D_QUIET", None)
        return
    os.environ["T1D_QUIET"] = "1"
    os.environ.pop("T1D_VERBOSE", None)
    for msg in _NOISY:
        warnings.filterwarnings("ignore", message=msg)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sbi\..*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# Subprocesses inherit the parent runner's verbosity via env.
if os.environ.get("T1D_VERBOSE") == "1":
    quiet(True)
elif os.environ.get("T1D_QUIET") == "1":
    quiet(False)
