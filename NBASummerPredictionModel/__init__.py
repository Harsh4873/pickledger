"""NBA Summer League in-house model exports."""


def generate_nba_summer_picks(*args, **kwargs):
    from .summer_model import generate_nba_summer_picks as _generate_nba_summer_picks

    return _generate_nba_summer_picks(*args, **kwargs)

__all__ = ["generate_nba_summer_picks"]
