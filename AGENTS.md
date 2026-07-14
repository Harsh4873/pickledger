# PickLedger Maintenance

PickLedger is a standalone repository deployed at `/pickledger/` through GitHub Pages.

## Verification

- Never open the deployed site, a browser, rendered Pages output, or live URLs to verify a change. The user confirms production behavior.
- Review source, run builds/tests, and inspect GitHub Actions and Pages API state.
- Run `npm run upcheck` before declaring the app healthy. If daily data is missing, dispatch the matching refresh workflow and investigate any remaining failure.
- Preserve the committed JSON data pipeline and the shared `pick-cache-writer` concurrency group.

## Publishing

- After coding changes, run focused tests and `npm run upcheck`, commit, push to `main`, and deploy through `.github/workflows/deploy-pages.yml`.
- Commits and pushes must use the currently logged-in GitHub user.
- Never add AI co-author trailers, `Co-authored-by:` lines, or AI/Cursor/Codex taglines to commits.
- Keep GitHub Pages configured for GitHub Actions deployment (`build_type: workflow`).
- Do not add a `CNAME`; the project inherits the `harsh.bet` custom domain from the user-site repository.
- Do not overwrite or revert unrelated user changes.

## Automation

- Scores24 publishing uses `scripts/scrapers/scores24_publish.sh`; see `docs/cursor-automations.md`.
- Scheduled data writers target this repository's `main` branch and dispatch the Pages workflow after authoritative cache updates.
- Model directories remain here because GitHub Actions runs them.
