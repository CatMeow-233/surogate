from pathlib import Path

from _experimental_checkout import use_checkout_portion

use_checkout_portion(Path(__file__).resolve().parents[1] / "src")
