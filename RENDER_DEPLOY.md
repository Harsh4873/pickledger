# Render Deployment (PickLedger Backend)

This repo includes a Render blueprint file at `render.yaml` for the backend service.

## 1. Push To GitHub

Render deploys from GitHub, so push this project first.

## 2. Create Service In Render

1. In Render, click `New` -> `Blueprint` (recommended).
2. Connect your GitHub repo.
3. Select this repo. Render will read `render.yaml` and create `pickledger-grader`.
4. Deploy.

If you prefer manual setup, choose `Web Service` and use:

- Build Command:
  - `pip install -r requirements.txt && pip install -r NBAPredictionModel/requirements.txt && pip install -r MLBPredictionModel/requirements.txt`
- Start Command:
  - `python pickgrader_server.py`
- Health Check Path:
  - `/health`

## 3. Point Frontend To Render Backend

After deploy, copy your backend URL (example: `https://pickledger-grader.onrender.com`).

Then open your ledger page with:

- `pickledger.html?api=https://pickledger-grader.onrender.com`

Or set once in browser console:

- `localStorage.setItem('pickledger_model_server', 'https://pickledger-grader.onrender.com')`

The frontend now reads this value automatically.

## 4. Lag / Reliability Tips

1. Use at least Render `Starter` plan so the service does not sleep.
2. Keep model runs async (already supported) to avoid browser timeouts.
3. Scores24 scraping can run on Render, but Cloudflare often blocks non-proxied Render egress IPs.
4. For persistent pick storage later, move from local browser storage to Postgres.

## 5. Scores24 On Render (Proxy Required For Stability)

The frontend tries Scores24 sync against your configured backend first, then localhost, then manual feed/cache fallback.

Required Render env vars:

1. `ENABLE_SCORES24_REMOTE=true`
2. `PLAYWRIGHT_PROXY_SERVER=<your proxy url>`
3. Optional auth: `PLAYWRIGHT_PROXY_USERNAME`, `PLAYWRIGHT_PROXY_PASSWORD`

Without a proxy server, NHL (and sometimes other leagues) can fail with Cloudflare 403 errors.

### Manual fallback workflow (still supported)

1. Ask Copilot to run local Scores24 scraping.
2. Copilot updates `scores24_manual_feed.json` with fresh picks.
3. Open the Models tab and click `LOAD SCORES24 FEED`.

Optional: to force manual-only mode on Render, set:

- `ENABLE_SCORES24_REMOTE=false`

## 6. Local Fallback

No change to your local flow is required. If no `api` query/localStorage is set, frontend still uses `http://127.0.0.1:8765`.

## 7. Build Failure: "No module named playwright"

If Render fails with:

- `/opt/render/project/src/.venv/bin/python: No module named playwright`

then your service is still using an older manual build command that runs:

- `python -m playwright install ...`

Fix options:

1. Preferred: Update Render build command to the new one in this doc (no Playwright install step needed).
2. Compatibility: Keep the old build command; this repo includes `playwright` in `requirements.txt` so the module exists during build.
