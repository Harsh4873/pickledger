"""The Pages artifact must include the merged model and the daily brief.

harsh.bet serves the dist/ tree uploaded by deploy-pages.yml. Dropping the
merged model cache or the brief files from that publish set 404s on the live
board. These tests fail when those outputs are omitted from the workflow.
"""

from __future__ import annotations

from scripts.pages_publish_set import (
    REQUIRED_OUTPUTS,
    REQUIRED_WORKFLOW_LINES,
    dry_check,
    missing_workflow_lines,
    workflow_text,
)


def test_workflow_publish_set_names_merged_model_and_daily_brief():
    text = workflow_text()
    assert missing_workflow_lines(text) == []
    for line in REQUIRED_WORKFLOW_LINES:
        assert line in text
    for rel in (
        "data/model_cache/latest.json",
        "data/bet_briefs/latest.json",
        "data/personal_ledger.json",
        "harsh.html",
    ):
        assert rel in REQUIRED_OUTPUTS
        assert f"dist/{rel}" in text


def test_omitted_publish_output_fails_the_workflow_check():
    text = workflow_text().replace("cp harsh.html dist/", "cp favicon.svg dist/")
    missing = missing_workflow_lines(text)
    assert "cp harsh.html dist/" in missing
    assert "test -f dist/harsh.html" not in missing

    dropped_brief = workflow_text().replace(
        "cp -R data/bet_briefs dist/data/",
        "cp -R data/parlay_cards dist/data/",
    )
    assert "cp -R data/bet_briefs dist/data/" in missing_workflow_lines(dropped_brief)

    dropped_model = workflow_text().replace(
        "cp -R data/model_cache dist/data/",
        "cp -R data/parlay_cards dist/data/",
    )
    assert "cp -R data/model_cache dist/data/" in missing_workflow_lines(dropped_model)

    dropped_ledger = workflow_text().replace(
        "cp data/personal_ledger.json dist/data/",
        "cp favicon.svg dist/data/",
    )
    assert "cp data/personal_ledger.json dist/data/" in missing_workflow_lines(dropped_ledger)


def test_dry_check_copies_required_outputs_without_deploying():
    report = dry_check()
    assert report["deployed"] is False
    assert report["missingWorkflowLines"] == []
    assert report["missingOutputs"] == []
    assert report["boardLinksHarsh"] is True
    assert report["boardFetchesMergedModel"] is True
