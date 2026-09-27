from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType

from scripts.scrapers import scores24_camoufox


def test_camoufox_binary_check_never_fetches(monkeypatch, tmp_path):
    browser = tmp_path / "camoufox"
    browser.write_text("", encoding="utf-8")
    browser.chmod(0o755)
    pkgman = ModuleType("camoufox.pkgman")
    checked = []

    def offline_path(*, download_if_missing=True):
        checked.append(download_if_missing)
        return tmp_path

    pkgman.camoufox_path = offline_path
    pkgman.launch_path = lambda _directory: str(browser)
    package = ModuleType("camoufox")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "camoufox", package)
    monkeypatch.setitem(sys.modules, "camoufox.pkgman", pkgman)

    assert scores24_camoufox.camoufox_binary() == (str(browser), None)
    assert checked == [False]


def test_missing_camoufox_skips_warmup_without_starting_process(monkeypatch):
    monkeypatch.setenv("SCORES24_CAMOUFOX_FALLBACK", "true")
    monkeypatch.setattr(scores24_camoufox, "camoufox_binary", lambda: (None, "missing browser"))
    monkeypatch.setattr(
        scores24_camoufox.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("process started")),
    )

    assert scores24_camoufox.warmup(2) == 1


def test_camoufox_warmup_timeout_kills_browser_process_group(monkeypatch):
    monkeypatch.setenv("SCORES24_CAMOUFOX_FALLBACK", "true")
    monkeypatch.setattr(scores24_camoufox, "camoufox_binary", lambda: ("/browser", None))
    killed = []

    class Process:
        pid = 1234
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("warmup", timeout)
            return -9

    monkeypatch.setattr(scores24_camoufox.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(scores24_camoufox.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    assert scores24_camoufox.warmup(2) == 1
    assert killed == [(1234, scores24_camoufox.signal.SIGKILL)]


def test_camoufox_warmup_launch_failure_is_soft(monkeypatch):
    monkeypatch.setenv("SCORES24_CAMOUFOX_FALLBACK", "true")
    monkeypatch.setattr(scores24_camoufox, "camoufox_binary", lambda: ("/browser", None))

    def fail_launch(*_args, **_kwargs):
        raise OSError("launch unavailable")

    monkeypatch.setattr(scores24_camoufox.subprocess, "Popen", fail_launch)
    assert scores24_camoufox.warmup(2) == 1


def test_camoufox_profile_uses_state_root(monkeypatch, tmp_path):
    monkeypatch.delenv("SCORES24_CAMOUFOX_PROFILE_DIR", raising=False)
    monkeypatch.setenv("SCORES24_STATE_ROOT", str(tmp_path))
    assert scores24_camoufox.camoufox_profile_path() == Path(tmp_path / "camoufox-profile")
