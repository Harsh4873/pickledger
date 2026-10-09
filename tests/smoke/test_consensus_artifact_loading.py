"""Exercise deferred artifact IO with payloads from committed outcomes."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import joblib
import pytest

from player_props import consensus


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    recorded = json.loads((root / "tests/fixtures/player_prop_outcomes.json").read_text())
    paths = {}
    expected = {}
    # The contents are real rows; this test concerns storage, not inference.
    for index, key in enumerate(list(consensus.MODEL_PATHS)[:4]):
        path = tmp_path / f"{key[0]}-{key[1]}.joblib"
        payload = recorded[index:index + 3]
        joblib.dump(payload, path)
        paths[key] = path
        expected[":".join(key)] = payload
    monkeypatch.setattr(consensus, "MODEL_PATHS", paths)
    monkeypatch.setattr(consensus, "_BUNDLE", False)
    monkeypatch.delenv("PICKLEDGER_DISABLE_PRECISION_MODEL", raising=False)
    calls = []
    original = joblib.load

    def load(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(joblib, "load", load)
    return paths, expected, calls


def test_only_requested_sport_artifacts_are_deserialized_once(artifacts):
    paths, expected, calls = artifacts
    bundle = consensus.load_consensus_bundle()
    assert calls == [paths[("MLB", "season")]]
    cache = bundle["artifacts"]
    assert cache.get("MLB:history") == expected["MLB:history"]
    assert cache["MLB:history"] == expected["MLB:history"]
    assert "MLB:history" in cache
    assert cache.get("not-present") is None
    assert consensus.load_consensus_bundle() is bundle
    assert calls == [paths[("MLB", "season")], paths[("MLB", "history")]]


@pytest.mark.parametrize("operation", [list, len, dict, lambda c: list(c.items()), lambda c: list(c.values()), lambda c: c.copy()])
def test_enumerating_artifacts_preserves_complete_ordered_mapping(artifacts, operation):
    paths, expected, calls = artifacts
    cache = consensus.load_consensus_bundle()["artifacts"]
    assert cache.get("WNBA:history") == expected["WNBA:history"]
    assert operation(cache) == operation(expected)
    assert sorted(calls) == sorted(paths.values())
    assert cache.copy() == expected
    assert len(calls) == len(paths)


def test_unreadable_optional_artifacts_keep_previous_availability_gate(artifacts, monkeypatch):
    paths, expected, calls = artifacts
    paths[("MLB", "season")].write_bytes(b"invalid joblib")
    bundle = consensus.load_consensus_bundle()
    assert calls == [paths[("MLB", "season")], paths[("MLB", "history")]]
    assert bundle["artifacts"].get("MLB:season") is None
    assert bundle["artifacts"].copy() == {key: value for key, value in expected.items() if key != "MLB:season"}
    for path in paths.values():
        path.write_bytes(b"invalid joblib")
    monkeypatch.setattr(consensus, "_BUNDLE", False)
    assert consensus.load_consensus_bundle() is None


def test_concurrent_lookups_share_one_deserialization(artifacts):
    paths, expected, calls = artifacts
    cache = consensus.load_consensus_bundle()["artifacts"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(cache.get, ["WNBA:history"] * 12))
    assert all(value is results[0] for value in results)
    assert results[0] == expected["WNBA:history"]
    assert calls.count(paths[("WNBA", "history")]) == 1
