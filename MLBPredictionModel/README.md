# MLB Prediction Model

This project implements the comprehensive MLB Prediction Model detailed in `system_prompt.md`. It eliminates the three biggest failure modes of AI sports analysis by employing a strict verification gate, an explicit three-layer probability build, and rigorous market mechanics (vig removal & Kelly criterion sizing).

## Architecture

- **`data_models.py`**: Contains data classes for Teams, Players, Weather, Venues, and Game Context.
- **`verification.py`**: The Mandatory Verification Gate (Step 0) checks that rosters are verified, lineups are posted, and weather is sourced.
- **`probability_layers.py`**:
  - `Layer 1`: Base Rate generation.
  - `Layer 2`: Situational adjusts (Weather, Park Factor, Rest, Bullpen).
  - `Layer 3`: Pitcher modifier (FIP diff).
  - Extremizer: Market Mechanics extremizing factor (1.3x).
- **`market_mechanics.py`**: Calculates vig-free implied probabilities, edges, and suggests appropriate 1/4 Kelly bet sizing.
- **`mlb_api.py`**: A stub framework designated for fetching JSON data from MLB StatsAPI or Weather platforms.
- **`main.py`**: The CLI executable linking all modules together.

## Execution

Run the main pipeline on simulated data:
```bash
python3 main.py
```

## Player-lineup rating research

`player_rating_features.py` adds a dated, empirical-Bayes batting rating for
an ordered nine-player lineup. It is not a fantasy score or a player Elo, and
it is deliberately not part of the public `new` model artifact or pick cache.

The first evaluation path is intentionally conservative:

```bash
python player_rating_oracle_data.py --seasons 2026
python backtest_player_rating_oracle.py
```

It updates player, starter, and team state only after a game has been scored,
then runs a date-boundary walk-forward comparison between a team/starter
control and the same control plus lineup ratings. Historical StatsAPI feeds do
not preserve a timestamped lineup announcement, so the identity of the final
lineup and starter is explicitly labeled **oracle knowledge**. A positive
oracle result measures upside only; it cannot promote the feature to live
predictions.

The manual **MLB Player-Rating Research** GitHub Actions workflow runs the same
isolated backtest and uploads its dataset, artifact, and metadata for 30 days.
Promotion requires a separate pregame official-lineup snapshot archive,
like-for-like live feature parity, and stable out-of-sample improvement over
the existing MLB New model and market baseline.
