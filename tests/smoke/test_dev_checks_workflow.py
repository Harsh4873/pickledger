"""Keep Dev Checks' partial checkouts aligned with their file dependencies."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_dev_checks_sparse_checkouts_keep_required_inputs():
    workflow = yaml.load((ROOT / ".github/workflows/dev-checks.yml").read_text(), Loader=yaml.BaseLoader)
    configs = {}
    for job in ("frontend", "smoke"):
        checkout = next(step for step in workflow["jobs"][job]["steps"] if step.get("uses") == "actions/checkout@v4")
        configs[job] = checkout["with"]
        assert configs[job]["fetch-depth"] == "1"
        assert configs[job]["lfs"] == "false"
    # Frontend needs few blobs; smoke fetches normal-size blobs up front to
    # avoid a large lazy promisor fetch while retaining the same fixtures.
    assert "filter" not in configs["frontend"]
    assert configs["smoke"]["filter"] == "blob:limit=100m"
    assert configs["frontend"]["sparse-checkout"].splitlines() == ["src", "tests"]
    assert configs["frontend"].get("sparse-checkout-cone-mode", "true") == "true"
    assert configs["smoke"]["sparse-checkout-cone-mode"] == "false"
    assert configs["smoke"]["sparse-checkout"].splitlines() == [
        "/*", "!/data/", "/data/calibration/", "!/data/calibration/team_prop_pregame_ledger/",
        "/data/model_cache/", "/data/player_props_cache/", "/data/player_props_snapshots/",
        "/data/nfl/", "/data/wnba/",
    ]
