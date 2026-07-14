def convert_american_to_implied(odds: int) -> float:
    """Converts American odds to implied probability (raw, with vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    elif odds < 0:
        # e.g. -150 -> 150 / 250
        positive_odds = abs(odds)
        return positive_odds / (positive_odds + 100)
    return 0.5

def remove_vig(odds1: int, odds2: int) -> tuple[float, float]:
    """Removes vig from a two-sided market."""
    impl_1 = convert_american_to_implied(odds1)
    impl_2 = convert_american_to_implied(odds2)
    sum_impl = impl_1 + impl_2
    
    true_1 = impl_1 / sum_impl
    true_2 = impl_2 / sum_impl
    return true_1, true_2

def calculate_edge(model_prob: float, market_implied_prob: float) -> float:
    return model_prob - market_implied_prob

def check_minimum_threshold(edge: float, bet_type: str = 'moneyline') -> bool:
    thresholds = {
        'moneyline': 0.05,
        'prop': 0.06,
        'total': 0.05,
        'parlay_leg': 0.07
    }
    return edge >= thresholds.get(bet_type, 0.05)

def calculate_kelly(odds: int, model_prob: float) -> float:
    """Calculates Kelly Criterion fraction. Returns full Kelly."""
    # b = decimal odds - 1
    # For positive American odds (+150): Decimal = 2.5. b = 1.5
    # For negative American odds (-150): Decimal = 1.667. b = 0.667
    
    if odds > 0:
        b = odds / 100.0
    else:
        b = 100.0 / abs(odds)
        
    p = model_prob
    q = 1.0 - p
    
    if b <= 0:
        return 0.0
        
    f_star = (b * p - q) / b
    return max(0.0, f_star)

def get_recommended_stake(odds: int, model_prob: float, max_bankroll_pct: float = 0.05) -> tuple[float, float]:
    """Returns (full_kelly, recommended_quarter_kelly) as percentages of bankroll."""
    full_kelly_frac = calculate_kelly(odds, model_prob)
    quarter_kelly_frac = full_kelly_frac * 0.25
    
    full_kelly_pct = full_kelly_frac * 100
    quarter_kelly_pct = min(quarter_kelly_frac, max_bankroll_pct) * 100
    
    return full_kelly_pct, quarter_kelly_pct
