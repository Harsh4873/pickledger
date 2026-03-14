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
  - `pip install -r requirements.txt && pip install -r NBAPredictionModel/requirements.txt && pip install -r MLBPredictionModel/requirements.txt && python -m playwright install chromium`
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
3. Scores24 scraping is the heaviest path; expect longer runtime there.
4. If traffic grows, split scraping into a separate worker service.
5. For persistent pick storage later, move from local browser storage to Postgres.

## 5. Local Fallback

No change to your local flow is required. If no `api` query/localStorage is set, frontend still uses `http://127.0.0.1:8765`.
