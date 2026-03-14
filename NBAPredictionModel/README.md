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
