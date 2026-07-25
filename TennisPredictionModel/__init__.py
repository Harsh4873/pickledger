"""In-house tennis match-winner model (Elo/WElo ratings + calibrated ML)."""

from typing import Any


def __getattr__(name: str) -> Any:
    # Lazy so importing the ratings/feature layer never drags in the serving
    # module: training, tuning and backtesting all run long before any artifact
    # exists, and they only need tennis_data/tennis_core.
    if name == "generate_tennis_picks":
        from .tennis_model import generate_tennis_picks

        return generate_tennis_picks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["generate_tennis_picks"]
