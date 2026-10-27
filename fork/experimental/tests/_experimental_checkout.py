"""Import surogate.experimental from this checkout, never from an installed copy.

The experimental trainer is a portion of the `surogate` namespace. A regular
`surogate` package on sys.path (the native distribution or a development
checkout) hides that portion, and a stale installed copy of
surogate.experimental would then be tested silently. use_checkout_portion puts
the checkout's portion first and fails if another copy was imported.
"""

import sys
from pathlib import Path


def use_checkout_portion(src):
    src = Path(src).resolve()
    portion = str(src / "surogate")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import surogate

    paths = surogate.__path__
    if portion not in list(paths):
        if isinstance(paths, list):
            paths.insert(0, portion)  # regular package: search the checkout portion first
        else:
            paths.append(portion)
    import surogate.experimental as experimental

    if not Path(experimental.__file__).resolve().is_relative_to(src):
        raise RuntimeError(
            f"stale surogate.experimental at {experimental.__file__}; remove it or use `fork/dev test experimental`"
        )
