from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_frontend_is_static_json_only():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert "from './firebase'" not in main
    assert "auth.currentUser" not in main
    assert "Firestore" not in main
    assert "ADMIN_BACKEND" not in main
    assert "./data/model_cache/index.json" in data
    assert "cannon_mlb_daily" not in data
    assert '<link rel="stylesheet" href="./src/styles/pickledger.css">' in html
    assert "See how every source has performed across the picks and results collected here." in html


def test_frontend_player_mode_is_persisted_isolated_and_team_defaulted():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    settings = (ROOT / "src" / "settings.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert 'data-pick-mode="team"' in html
    assert 'data-pick-mode="player"' in html
    assert "const PICK_MODE_KEY = 'pickledger_pick_mode'" in settings
    assert "const mode: PickMode = stored === 'player' ? 'player' : 'team'" in settings
    assert "pickledger:modechange" in settings

    assert "./data/model_cache/index.json" in data
    assert "cannon_mlb_daily" not in data
    assert "./data/player_props_cache/index.json" in data
    assert "./data/player_props_cache/latest.json" in data
    assert "let teamPicks: Pick[] = []" in data
    assert "let playerPicks: Pick[] = []" in data
    assert "function isPlayerScopedPick(" in data
    assert "if (isPlayerScopedPick(pick)) playerById.set(pick.id, pick)" in data
    assert "return activePickMode === 'player' ? playerPicks : teamPicks" in data
    assert "decision === 'BET' || decision === 'LEAN' || decision === 'PASS'" in data

    assert "const activeFilters = new Set<string>()" in main
    assert "activeFilters.clear()" in main
    assert "selectedDate = ''" in main
    assert "search.value = ''" in main
    assert "function isOpenPick(" in main
    assert "pick.result === 'pending' && !isUnsupportedPendingPick(pick) && isPublishedDailyPick(pick)" in main
    assert "dailyDecision(pick) === 'PASS'" in main
    assert "const pending = getAllPicks().filter(isOpenPick)" in main
    assert "UNTRACKED" in main
    assert "mlbLivePlayerStat(" in main
    assert "espnPlayerStat(" in main
    assert "PLAYER_PROPS_ML_FIRST_SNAPSHOT_AT" in data
    assert "isMlEraPlayerProp(pick)" in data
    assert ".pick-mode-segment" in css
    assert "body.mobile-app-mode .pick-mode-segment" in css
    assert "@media (max-width: 700px)" in css


def test_research_details_use_generator_schema_fields_across_pick_views():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    for field in ("full_kelly", "quarter_kelly", "confidence", "reason", "key_factors"):
        assert f"{field}?" in data
        assert f"pick.{field}" in main
    assert "function researchDetailsHtml(" in main
    assert "Quarter Kelly" in main
    assert "Full Kelly" in main
    assert "Key factors" in main
    assert "expandedResearchPickKeys" in main
    assert "data-research-pick-card" in main
    assert "isPlayer ? '' : `<span class=\"home-feed-row-sport\"" in main
    assert "function bindResearchDetailCards(" in main
    assert "bindPickCards(results)" in main
    assert "function parlayCardHtml(" in main
    assert "Show research details" in main
    assert ".home-player-details" in css
    assert ".home-player-extra" in css
    assert ".home-feed-row.expanded .home-player-extra" in css
    assert ".home-player-factors" in css
    assert 'body[data-pick-mode="player"] .home-feed-row-pick' in css
    player_pick_css = css[css.index('body[data-pick-mode="player"] .home-feed-row-pick'):]
    assert "white-space: normal" in player_pick_css[:350]
    assert "overflow: visible" in player_pick_css[:350]
    assert 'body.mobile-app-mode[data-pick-mode="player"] .home-feed-row' in css


def test_home_team_pick_text_expands_on_hover_focus_and_tap():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert 'data-home-pick-text role="button" tabindex="0" aria-expanded="false"' in main
    assert "function bindHomePickTextExpansion(" in main
    assert "row.classList.toggle('pick-text-expanded')" in main
    assert "bindHomePickTextExpansion(container)" in main
    assert "@media (hover: hover) and (pointer: fine)" in css
    assert ".home-feed-row:hover .home-feed-row-pick" in css
    assert ".home-feed-row:focus-visible .home-feed-row-pick" in css
    assert ".home-feed-row-pick[data-home-pick-text]:focus-visible" in css
    assert ".home-feed-row.pick-text-expanded .home-feed-row-pick" in css
    expanded_css = css[css.index(".home-feed-row:hover .home-feed-row-pick"):]
    assert "white-space: normal" in expanded_css[:500]
    assert "overflow: visible" in expanded_css[:500]
    assert "text-overflow: clip" in expanded_css[:500]


def test_header_brand_and_freshness_copy_are_friendly_and_accurate():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert 'class="brand-home" href="#home" onclick="goHome(event)"' in html
    assert "function goHome(" in main
    assert "function latestPayloadTimestamp(" in data
    assert "Picks updated ${updatedAgoLabel(status.updatedAt)}" in main
    assert "Models refresh each morning and again around 3:30 PM CT" in html
    assert "Scores are checked automatically every 15 minutes" in html
    assert "cache ${status.date}" not in main
    assert ".brand-home" in css


def test_static_viewer_keeps_public_tabs_and_client_grading():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    for tab in ("home", "search", "rankings", "daily", "parlays", "profit"):
        assert f"id=\"tab-{tab}\"" in html
    assert 'id="tab-trends"' not in html
    assert ">TRENDS</button>" not in html
    assert 'onclick="switchTab(\'daily\')">BEST BETS</button>' in html
    assert 'onclick="switchTab(\'parlays\')">PARLAYS</button>' in html
    assert 'onclick="switchTab(\'profit\')">PROFIT DESK</button>' in html
    assert ">YOUR BETS</button>" not in html
    assert html.index(">BEST BETS</button>") < html.index(">PARLAYS</button>")
    assert html.index(">PARLAYS</button>") < html.index(">PROFIT DESK</button>")
    assert "async function refreshAutoGrades()" in main
    assert "async function gradeDate(" in main
    assert "site.api.espn.com" in main
    assert "setLocalResult(pick.id" in main
    assert "await loadAllData();" in main
    assert "DISPLAY_TIME_ZONE = 'America/Chicago'" in main
    assert "function centralDateKey(" in main
    assert "isOpenPick(pick) && pickDateKey(pick) === selectedDate" in main
    assert "window.setInterval(() => void refreshForCentralClock(), AUTO_REFRESH_MS)" in main
    assert "Find a team, matchup, or source in the selected date’s open picks" in html
    assert "embeddedResult === 'pending' ? localResult : embeddedResult" in data
    assert "function isTrackedPick(" in data
    assert "decision === 'BET' || decision === 'LEAN'" in data
    assert "pick && isTrackedPick(pick)" in data
    assert "function renderRankings()" in main
    assert "function renderSearch()" in main


def test_rich_static_viewer_restores_consensus_table_and_scores():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert "function canonicalTrendSignal(" in main
    assert "matching: !group.pass && new Set(group.picks.map(sourceName)).size >= 2" in main
    assert ".trend-market.matching" in css
    assert "function renderDayOfWeekTable()" in main
    assert 'class="dow-table"' in main
    assert 'id="dow-overall-heatmap"' not in html
    assert "async function refreshHomeScores(" in main
    assert "homeScoreChipHtml(" in main
    assert "Open ESPN box score" in main


def test_source_rankings_expand_period_records_and_static_cards_do_not_fake_clicks():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert "function sourceRecordLines(" in main
    for label in ("TODAY", "YESTERDAY", "LAST 7 DAYS", "ALL TIME"):
        assert f"label: '{label}'" in main
    assert "function isSettledPick(" in main
    assert "const PLAYER_PROP_RANKING_START_DATE = '2026-06-23'" in main
    assert "if (activePickMode !== 'player') return picks" in main
    assert "date >= PLAYER_PROP_RANKING_START_DATE" in main
    assert "function rankingWindowLabel(" in main
    assert "function picksForRankingBucket(" in main
    assert "sourceRecordLines(picksForRankingBucket(comparablePicks, item.source), centralDateKey())" in main
    assert 'data-source-card="${escapeHtml(item.source)}"' in main
    assert 'role="button" tabindex="0" aria-expanded="${expanded}"' in main
    assert "function bindSourceCards(" in main
    assert "View period records" in main
    assert 'id="source-rankings-title"' in html
    assert 'id="source-rankings-subtitle"' in html
    assert 'id="dow-subtitle"' in html
    assert "Select a source for today, yesterday, last 7 days, and all-time records." in html
    assert ".source-expand-control" in css
    assert ".source-card.expanded .source-deep-dive" in css
    assert ".trend-game-card:hover" not in css
    assert ".search-card:hover" not in css
    assert ".sport-card:hover" not in css
    assert ".home-game-card:hover" not in css
    assert ".daily-bet-card:hover" not in css


def test_profit_desk_is_its_own_precomputed_decision_first_tab():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    for section in ("Live Card", "Watchlist & Rejections", "How a pick earns promotion"):
        assert section in main
    assert 'id="profit-container"' in html
    assert "type ProfitView = 'card' | 'watchlist' | 'method'" in main
    assert "let profitView: ProfitView = 'card'" in main
    assert "function renderProfit(" in main
    assert "getProfitDeskPayload(key)" in main
    assert "function profitDeskCandidateCard(" in main
    assert "function profitDeskMethodHtml(" in main
    assert "Profit Desk Date" in main
    assert "function toggleProfitDatePicker(" in main
    assert "role=\"tablist\" aria-label=\"Profit Desk views\"" in main
    assert "Sit out" in main
    assert "RESEARCH ONLY • 0U" in main
    assert "No recommendation" in main
    assert "will not improvise a recommendation from raw pick feeds or client-side scoring" in main
    assert "Conservative probability" in main
    assert "Expected value" in main
    assert "Pr(EV &gt; 0)" in main
    assert "distinct dates" in main
    assert "Flat-stake evidence" in main
    assert "LIVE STAKE" in main
    assert "Research context is not live proof" in main
    assert "CLOSING LINE" in main
    assert "record.avgClv" in main
    assert "EDGE clears strict segment-level market-alpha gates" in main
    assert "./data/profit_desk/index.json" in data
    assert "./data/profit_desk/latest.json" in data
    assert "export interface ProfitDeskPayload" in data
    assert "export function getProfitDeskPayload(" in data
    assert "liveRecordToDate" in data
    assert ".profit-decision" in css
    assert ".profit-candidate" in css
    assert ".profit-candidate.tier-edge::before" in css
    assert ".profit-shadow-stake.is-live" in css
    assert ".profit-method-steps" in css
    assert ".profit-sport-filter" in css


def test_best_bets_shortlist_is_fully_restored_on_the_daily_tab():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert 'onclick="switchTab(\'daily\')">BEST BETS</button>' in html
    assert "type DailyView = 'picks' | 'consensus' | 'sources' | 'research'" in main
    assert "let dailyView: DailyView = 'picks'" in main
    assert "function renderDaily(" in main
    assert "The Shortlist" in main
    assert "function dailyPickScore(" in main
    assert "Best Bets Date" in main
    for section in ("Top Picks", "Consensus Signals", "Hot Sources", "Research Queue"):
        assert section in main
    assert "MODEL GREENLIGHT" in main
    assert "PROBABILITY LEADER" in main
    assert "PRICEY FAVORITE" in main
    assert "Quick read, not a blind card." in main
    assert ".daily-hero" in css
    assert ".daily-view-shell" in css
    assert ".daily-bet-card" in css


def test_profit_and_roi_exclude_missing_assumed_and_unverified_prices():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")

    assert "function normalizedPriceProvenance(" in data
    assert "assumed|synthetic|proxy|fallback|default|estimated|model[_ ]price" in data
    assert "raw.market_priced === true && observedMarker" in data
    assert "if (pick.price_verified !== true) return 0" in data
    assert "const pricedPicks = picks.filter(pick => pick.price_verified === true" in main
    assert "P/L untracked" in main


def test_rankings_show_untracked_units_instead_of_zero_when_nothing_is_priced():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    # Buckets with no verified-price settled picks must never render "+0u"
    # as if they broke even; they show an untracked marker instead.
    assert "function trackedUnits(" in main
    assert "return stats.priced ? signedUnits(stats.net) : '—'" in main
    assert "priced: pricedPicks.filter(isSettledPick).length" in main
    assert "P/L untracked — no verified-price picks yet" in main
    assert "'— (no priced picks)'" in main
    assert "Units and ROI count only picks settled at verified sportsbook prices." in html
    assert '<div class="stat-box-val" id="stat-units">&mdash;</div>' in html


def test_parlays_tab_renders_card_level_filters_and_rankings():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")
    builder = (ROOT / "scripts" / "build_parlay_cards.py").read_text(encoding="utf-8")

    for section in ("Edge Double", "Prop Double"):
        assert section in builder
    assert "type ParlayView = string" in main
    assert "let parlayView: ParlayView = 'all'" in main
    assert "let parlayResultMode: ResultMode = 'pending'" in main
    assert "function renderParlays(" in main
    assert "function setParlayView(" in main
    assert "function setParlayResultMode(" in main
    assert "function parlayFilterOptions(" in main
    assert "function parlayCategoryOptions(" in main
    assert "function parlayCardHtml(" in main
    assert "function parlayRankingCardsForDate(" in main
    assert ".filter(payload => !engineVersion || payload.engineVersion === engineVersion)" in main
    assert "dedupeParlayCards(historical.length ? historical : fallbackCards)" in main
    assert "function parlayRankingsPanel(" in main
    assert "Parlay Rankings" in main
    assert "Whole-card records" in main
    assert "Team / Player" in main
    assert "Switch to ${otherMode === 'team' ? 'Team' : 'Player'} mode for this slate" in main
    assert "ENGINE_VERSION = \"parlay_cards_v5_market_excess\"" in builder
    assert "ENGINE_CUTOVER_DATE = \"2026-07-01\"" in builder
    assert "Records count each whole parlay slip once" in main
    assert "No same-game legs, same-player duplicates, or duplicate markets are allowed" in main
    assert "function parlayCardsForMode(" in main
    assert "function parlayCardPickMode(" in main
    assert "Disciplined 2-leg slips from sources with proven trailing edge over market prices." in main
    assert "Disciplined 2-leg slips from consensus-qualified, market-priced player props." in main
    assert "Parlay Date" in main
    assert "role=\"tablist\" aria-label=\"Parlay board filters\"" in main
    assert ".daily-view-nav" in css
    assert ".daily-view-select-wrap" in css
    assert ".parlay-grid" in css
    assert ".parlay-card" in css
    assert ".parlay-metrics" in css
    assert ".parlay-leg-list" in css


def test_soccer_consensus_keeps_lines_and_specialty_markets_distinct():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    signals = main[main.index("function canonicalTrendSignal("):main.index("function trendSignalGroups(")]
    assert "asian-handicap" in signals
    assert "function canonicalTrendLine(" in main
    assert "canonicalTrendLine(namedHandicap[3])" in signals
    assert "canonicalTrendLine(asian[2])" in signals
    assert "spread:${canonicalTeamForPick(pick, spread[1])}:${canonicalTrendLine(spread[2])}" in signals
    assert "(?:[.,]\\d+)?" in signals
    assert "total:${total[1].toLowerCase()}:${total[2]}" in signals
    assert "(?:ML|moneyline|to win|wins?)$/i" in signals


def test_player_mode_keeps_best_bets_available_and_prop_sources_separate():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert 'data-player-hidden-tab="trends"' not in html
    assert "function syncModeTabs(" not in main
    assert 'onclick="switchTab(\'daily\')">BEST BETS</button>' in html
    assert 'onclick="switchTab(\'parlays\')">PARLAYS</button>' in html
    assert "dailyView = 'picks'" in main
    assert "profitView = 'card'" in main
    assert "parlayView = 'all'" in main
    assert "Parlay filter" in main
    assert "const requestedDate = selectedDate || today" in main
    assert "getParlayCardsPayload(requestedDate)" in main
    assert "function playerRankingEpoch(" in main
    assert "function rankingComparablePicks(" in main
    assert "const PLAYER_PROP_RANKING_START_DATE = '2026-06-23'" in main
    assert "if (activePickMode !== 'player') return picks" in main
    assert "function latestAvailableDateKey(" in main
    assert "function playerModelRank(" in main
    assert "return 10000 - modelRank" in main
    assert "function consensusModelPanelHtml(" in main
    assert "function consensusApplicableModelLabels(" in main
    assert "startsWith(`${sportPrefix}_`)" in main
    assert "function playerRankingNames(" in main
    assert "function rankingBucketNames(" in main
    assert "function addPickToRankingBuckets(" in main
    assert "(activePickMode === 'player' ? comparablePicks : rankingPicks).forEach(pick => addPickToRankingBuckets(bySource, pick))" in main
    assert "rankingBucketNames(pick).forEach(source =>" in main
    assert "? 'Model Rankings' : 'Source Rankings'" in main
    assert "? 'Model' : 'Source'" in main
    assert "home-player-model-stack" in main
    assert "Models</strong>" in main
    assert "activePickMode === 'player' ? 'Player Prop' : 'Team'" in main
    for source in ("NBAPlayerProps", "MLBPlayerProps", "WNBAPlayerProps"):
        assert source in data
    assert "playerProp && fallbackSource" in data
    assert ".home-player-model-stack" in (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")


def test_home_filters_prioritize_primary_sports_and_use_more_menu():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert "const PRIMARY_FILTERS = ['ALL', 'MLB', 'WNBA', 'NBA SUMMER', 'FIFA WC']" in main
    assert "const ARCHIVED_SPORTS = new Set(['NBA'])" in data
    assert "!ARCHIVED_SPORTS.has(pick.sport)" in data
    assert "'MLB NEW': 'MLB Model'" in data
    assert "'FIFA WC In-House': 'FIFA Model'" in data
    assert "if (filter === 'NBA SUMMER') return 'SUMMER'" in main
    assert "filter === 'FIFA WC' ? 'FIFA' : filter" in main
    assert "function toggleHomeFilter(" in main
    assert "activeFilters.has(pick.sport)" in main
    assert "activeFilters.has(sourceName(pick))" in main
    assert "aria-pressed=\"${filterActive(filter)}\"" in main
    assert "activeFilterSummary()" in main
    assert 'id="filter-more-btn"' in main
    assert "extraFilters.map(filterButton)" in main
    assert ".filter-more-wrap" in css
    assert ".filter-dropdown.open" in css
    assert "body.mobile-app-mode .filter-more-wrap" in css
    assert "position: static" in css[css.index("body.mobile-app-mode .filter-more-wrap"):][:120]
    mobile_filter = css[css.index("body.mobile-app-mode .filter-bar"):]
    assert "overflow: visible" in mobile_filter[:220]


def test_your_bets_slot_is_replaced_by_parlays_without_clearing_storage_helpers():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert "const YOUR_BETS_STORAGE_KEY = 'pickledger_your_bets_v1'" in main
    assert "localStorage.setItem(YOUR_BETS_STORAGE_KEY" in main
    assert "function addPickToYourBets(" in main
    assert "function updateYourBetUnits(" in main
    assert "function syncYourBetResults(" in main
    assert "const modeBets = yourBets.filter(bet => bet.pickMode === activePickMode)" in main
    assert "function yourBetAddButton(pick: Pick): string" in main
    assert "return ''" in main
    assert ">YOUR BETS</button>" not in html
    assert 'onclick="switchTab(\'parlays\')">PARLAYS</button>' in html
    assert "Locked and graded by PickLedger" in main
    assert "function addCustomYourBet(" not in main
    assert "function updateYourBetResult(" not in main
    assert "function undoYourBetChange(" not in main
    assert "Add A Custom Bet" not in main
    assert "UNDO CHANGE" not in main
    for label in ("TODAY", "YESTERDAY", "ALL TIME"):
        assert f"yourBetSummaryCard('{label}'" in main
    assert 'id="tab-your-bets"' not in html
    assert 'id="tab-parlays"' in html
    assert ".your-bets-shell" in css
    assert ".your-bet-card" in css
    assert ".your-bet-locked-result" in css


def test_phone_toggle_keeps_brand_visible_and_more_menu_unclipped():
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    mobile_header = css[css.index("body.mobile-app-mode header {"):]
    assert "grid-template-columns: minmax(0, 1fr)" in mobile_header[:260]
    assert "body.mobile-app-mode .brand-home" in css
    brand = css[css.index("body.mobile-app-mode .brand-home"):]
    assert "width: max-content" in brand[:180]
    assert "overflow: visible" in brand[:180]


def test_tab_ordering_prioritizes_home_start_time_and_actionable_picks_elsewhere():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")

    for helper in (
        "function pickStartTimestamp(",
        "function gameStartTimestamp(",
        "function compareGameStartAsc(",
        "function startBucket(",
        "function compareActionableStart(",
        "function comparePickActionableStart(",
        "function compareDailyConsensusSignal(",
        "function compareHomePickRows(",
    ):
        assert helper in main
    assert "return timestamp > now ? 0 : 2" in main
    assert "if (leftBucket !== rightBucket) return leftBucket - rightBucket" in main
    assert "return leftBucket === 2 ? right - left : left - right" in main
    assert "const sortedGames = [...groups.entries()].sort((left, right) => compareGameStartAsc(left[1], right[1]))" in main
    assert "const sortedPicks = [...picks].sort(compareHomePickRows)" in main
    assert "homeDecisionRank(left) - homeDecisionRank(right)" in main
    assert "(pickProbability(right) || 0) - (pickProbability(left) || 0)" in main
    assert ".sort(comparePickActionableStart);" in main
    assert "comparePickActionableStart(a.primary, b.primary)" in main
    assert "comparePickActionableStart(left.game, right.game)" in main
    assert ".sort(comparePickActionableStart));" in main


def test_cache_manifest_lists_committed_dated_payloads():
    manifest = json.loads((ROOT / "data" / "model_cache" / "index.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    assert files == sorted(files)
    assert files
    assert "latest.json" not in files
    for filename in files:
        assert (ROOT / "data" / "model_cache" / filename).exists()


def test_auto_grader_updates_nested_model_picks(monkeypatch):
    module = _load_module("auto_grade_picks", ROOT / "scripts" / "auto_grade_picks.py")
    payload = {
        "date": "2026-06-08",
        "models": {
            "mlb_new": {
                "picks": [
                    {
                        "source": "MLB Model",
                        "sport": "MLB",
                        "pick": "Cubs ML (Cubs vs Cardinals)",
                        "decision": "BET",
                        "result": "pending",
                    }
                ]
            }
        },
    }

    def fake_grade(picks, existing, year):
        pick_id = picks[0]["id"]
        return {
            "graded": {pick_id: "win"},
            "startTimes": {pick_id: "2026-06-08T20:00:00Z"},
        }

    monkeypatch.setattr(module.pickgrader_server, "auto_grade", fake_grade)
    assert module.grade_payload(payload) == 2
    pick = payload["models"]["mlb_new"]["picks"][0]
    assert pick["result"] == "win"
    assert pick["start_time"] == "2026-06-08T20:00:00Z"


def test_auto_grader_rechecks_previously_decided_tracked_picks(monkeypatch):
    module = _load_module("auto_grade_recheck_test", ROOT / "scripts" / "auto_grade_picks.py")
    payload = {
        "date": "2026-06-13",
        "models": {
            "wnba_player_props": {
                "picks": [{
                    "id": "aneesah-morrow-rebounds",
                    "source": "PickLedgerPro In-House Player Props",
                    "scope": "player",
                    "sport": "WNBA",
                    "pick": "Aneesah Morrow Over 10.5 Rebounds",
                    "decision": "BET",
                    "result": "win",
                }]
            }
        },
    }

    def fake_grade(picks, existing, year):
        assert picks[0]["result"] == "pending"
        return {"graded": {"aneesah-morrow-rebounds": "loss"}, "startTimes": {}}

    monkeypatch.setattr(module.pickgrader_server, "auto_grade", fake_grade)
    assert module.grade_payload(payload) == 1
    assert payload["models"]["wnba_player_props"]["picks"][0]["result"] == "loss"


def test_auto_grader_only_tracks_bet_and_lean_decisions(monkeypatch):
    module = _load_module("auto_grade_pass_test", ROOT / "scripts" / "auto_grade_picks.py")
    payload = {
        "date": "2026-06-08",
        "models": {
            "mlb_new": {
                "picks": [
                    {
                        "source": "MLB Model",
                        "sport": "MLB",
                        "pick": "Cubs ML (Cubs vs Cardinals)",
                        "decision": "PASS",
                        "result": "pending",
                    },
                    {
                        "source": "MLB Model",
                        "sport": "MLB",
                        "pick": "Cardinals ML (Cubs vs Cardinals)",
                        "decision": "WATCH",
                        "result": "pending",
                    },
                    {
                        "source": "MLB Model",
                        "sport": "MLB",
                        "pick": "Over 8.5 (Cubs vs Cardinals)",
                        "result": "pending",
                    },
                ]
            }
        },
    }

    def fail_if_called(*_args):
        raise AssertionError("PASS decisions must not be sent to the grader")

    monkeypatch.setattr(module.pickgrader_server, "auto_grade", fail_if_called)
    assert module.grade_payload(payload) == 0
    assert all(pick["result"] == "pending" for pick in payload["models"]["mlb_new"]["picks"])


def test_auto_grader_ignores_player_props_from_before_ml_retraining(monkeypatch):
    module = _load_module("auto_grade_ml_cutoff_test", ROOT / "scripts" / "auto_grade_picks.py")
    payload = {
        "date": "2026-06-15",
        "generatedAt": "2026-06-15T22:55:06Z",
        "models": {
            "mlb_player_props": {
                "picks": [{
                    "id": "legacy-prop",
                    "sport": "MLB",
                    "pick": "Player Over 0.5 Hits",
                    "decision": "BET",
                    "result": "pending",
                    "probability_source": "legacy_projection",
                    "ranking_updated_at": "2026-06-15T22:55:06Z",
                }]
            }
        },
    }

    monkeypatch.setattr(module.pickgrader_server, "auto_grade", lambda *_args: (_ for _ in ()).throw(AssertionError("legacy props must not be graded")))
    assert module.grade_payload(payload, ml_player_props_only=True) == 0


def test_scheduled_refreshes_are_json_only_and_use_shared_writer_lock():
    workflow_names = (
        "auto-grade.yml",
        "calibration-refresh.yml",
        "model-cache-refresh.yml",
        "player-props-refresh.yml",
        "external-feed-refresh.yml",
    )
    for name in workflow_names:
        workflow = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "group: pick-cache-writer" in workflow
        assert "cancel-in-progress: false" in workflow

    model = (ROOT / ".github" / "workflows" / "model-cache-refresh.yml").read_text(encoding="utf-8")
    feeds = (ROOT / ".github" / "workflows" / "external-feed-refresh.yml").read_text(encoding="utf-8")
    assert "--skip-firestore" in model
    assert "--skip-firestore" in feeds
    assert "FIREBASE_PROJECT_ID" not in model
    assert "FIREBASE_PROJECT_ID" not in feeds


def test_refresh_timing_and_pages_deploy_are_deterministic():
    workflows = ROOT / ".github" / "workflows"
    model = (workflows / "model-cache-refresh.yml").read_text(encoding="utf-8")
    feeds = (workflows / "external-feed-refresh.yml").read_text(encoding="utf-8")
    grader = (workflows / "auto-grade.yml").read_text(encoding="utf-8")
    calibration = (workflows / "calibration-refresh.yml").read_text(encoding="utf-8")
    props = (workflows / "player-props-refresh.yml").read_text(encoding="utf-8")
    deploy = (workflows / "deploy-pages.yml").read_text(encoding="utf-8")

    assert "cache-gate" not in model
    assert "cron: '*/15 * * * *'" in grader
    assert 'cron: "45 12 * * *"' in model
    assert 'cron: "10,40 14 * * *"' in feeds
    assert "gh workflow run calibration-refresh.yml --ref main" in grader
    assert "decided - last >= 100" in grader
    assert "python scripts/train_pick_calibration.py" in calibration
    assert "gh workflow run player-props-refresh.yml --ref main" in calibration
    for workflow in (model, props, feeds, grader, calibration):
        assert "gh workflow run deploy-pages.yml --ref main" in workflow
        assert "actions: write" in workflow
    assert not (workflows / "cannon-daily-refresh.yml").exists()
    assert "Check daily data readiness" in deploy
    assert "python scripts/site_upcheck.py --data-only" in deploy
    assert "if: needs.readiness.outputs.ready == 'true'" in deploy
    assert "Verify styled Pages artifact" in deploy
    assert "find dist/assets -maxdepth 1 -name '*.js'" in deploy
    assert "! grep -q 'src/main.ts' dist/index.html" in deploy
    assert "python scripts/site_upcheck.py" in deploy
    guard = (workflows / "model-cache-freshness-guard.yml").read_text(encoding="utf-8")
    assert 'CACHE_HEALTHY="$(python - <<\'PY\'' in guard
    assert 'models[key].get("ok") is True for key in required' in guard
    assert 'PLAYER_CACHE_HEALTHY="$(python - <<\'PY\'' in guard
    assert '"nba_player_props"' in guard
    assert '"mlb_player_props"' in guard
    assert '"wnba_player_props"' in guard
    assert 'official_mlb_games = max(' in guard
    assert 'str(pick.get("probability_source") or "").strip() != "player_props_ml_v1"' in guard
    assert 'pick.get("preserved_from_prior_refresh")' in guard
    assert 'key == "mlb_player_props" or bucket.get("abstained") is not True' in guard
    assert 'DISPATCHES_TODAY="$(gh run list' in guard
    assert '--json createdAt,displayTitle,event' in guard
    assert '"Player Props Refresh $TARGET_DATE"' in guard
    assert 'morning automation owns any bounded post-fix rerun' in guard
    assert "run-name: Player Props Refresh ${{ github.event.inputs.date || 'today' }}" in props
    assert "continue-on-error: true" in props
    assert "Enforce player-props publication contract" in props


def test_refresh_workflows_commit_as_triggering_actor():
    for name in (
        "auto-grade.yml",
        "calibration-refresh.yml",
        "model-cache-refresh.yml",
        "player-props-refresh.yml",
        "external-feed-refresh.yml",
    ):
        workflow = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert 'git config user.name  "${GITHUB_ACTOR}"' in workflow
        assert 'git config user.email "${ACTOR_EMAIL}"' in workflow
        assert "github-actions[bot]" not in workflow


def test_model_cache_merge_seeds_external_feeds_on_new_slate_day(tmp_path):
    module = _load_module("merge_model_cache_payload_new_day", ROOT / "scripts" / "merge_model_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    previous = {
        "date": "2026-07-07",
        "models": {
            "wnba": {"ok": True, "picks": [{"pick": "Old"}]},
            "scores24_wnba": {"ok": True, "picks": [{"pick": "S24"}]},
        },
        "external_feeds": {
            "scores24_wnba": {"ok": True, "picks": [{"pick": "S24"}]},
        },
    }
    generated = {
        "date": "2026-07-08",
        "models": {
            "wnba": {"ok": True, "picks": [{"pick": "New"}]},
        },
    }
    (cache_dir / "latest.json").write_text(json.dumps(previous), encoding="utf-8")
    merged = module.merge_payload(generated, cache_dir)
    assert merged["models"]["wnba"]["picks"][0]["pick"] == "New"
    assert merged["external_feeds"]["scores24_wnba"]["picks"][0]["pick"] == "S24"


def test_model_cache_merge_keeps_newer_committed_external_feed_bucket(tmp_path):
    module = _load_module("merge_model_cache_payload_external_race", ROOT / "scripts" / "merge_model_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-07-14",
        "models": {
            "scores24_wnba": {
                "ok": True,
                "date": "2026-07-14",
                "generatedBy": "local:external-feed-refresh",
                "meta": {"from": "local"},
                "picks": [{"pick": "Current local pick"}],
            }
        },
    }
    generated = {
        "date": "2026-07-14",
        "models": {
            "wnba": {"ok": True, "picks": [{"pick": "Fresh team pick"}]},
            "scores24_wnba": {
                "ok": True,
                "date": "2026-07-14",
                "generatedBy": "github-actions:external-feed-refresh",
                "meta": {"from": "github-actions"},
                "picks": [{"pick": "Stale starting-snapshot pick"}],
            },
        },
    }
    (cache_dir / "2026-07-14.json").write_text(json.dumps(current), encoding="utf-8")

    merged = module.merge_payload(generated, cache_dir)

    assert merged["models"]["wnba"]["picks"][0]["pick"] == "Fresh team pick"
    scores = merged["models"]["scores24_wnba"]
    assert scores["generatedBy"] == "local:external-feed-refresh"
    assert scores["meta"]["from"] == "local"
    assert scores["picks"][0]["pick"] == "Current local pick"


def test_model_cache_merge_preserves_other_deployed_buckets(tmp_path):
    module = _load_module("merge_model_cache_payload", ROOT / "scripts" / "merge_model_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-06-08",
        "models": {
            "mlb_new": {"ok": True, "picks": [{"pick": "A", "result": "win"}]},
            "sportytrader_mlb": {"ok": True, "picks": [{"pick": "B"}]},
        },
        "mlb_new": {"ok": True, "picks": [{"pick": "A", "result": "win"}]},
    }
    generated = {
        "date": "2026-06-08",
        "models": {
            "nba": {"ok": True, "picks": [{"pick": "C"}]},
        },
        "mlb_new": {},
        "nba": {"ok": True, "picks": [{"pick": "C"}]},
    }
    (cache_dir / "2026-06-08.json").write_text(json.dumps(current), encoding="utf-8")
    merged = module.merge_payload(generated, cache_dir)
    assert merged["models"]["mlb_new"]["picks"][0]["result"] == "win"
    assert merged["models"]["sportytrader_mlb"]["picks"][0]["pick"] == "B"
    assert merged["models"]["nba"]["picks"][0]["pick"] == "C"
    assert merged["mlb_new"]["picks"][0]["pick"] == "A"
    assert merged["nba"]["picks"][0]["pick"] == "C"


def test_model_cache_merge_preserves_committed_grades(tmp_path):
    module = _load_module("merge_model_cache_payload_grades", ROOT / "scripts" / "merge_model_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-06-08",
        "models": {
            "mlb_new": {
                "picks": [{
                    "source": "X",
                    "sport": "MLB",
                    "pick": "Cubs ML",
                    "result": "win",
                    "pregame_snapshot": {"probability": 0.61},
                }]
            }
        },
    }
    generated = {
        "date": "2026-06-08",
        "models": {
            "mlb_new": {
                "picks": [{"source": "X", "sport": "MLB", "pick": "Cubs ML", "result": "pending"}]
            }
        },
    }
    (cache_dir / "2026-06-08.json").write_text(json.dumps(current), encoding="utf-8")
    merged = module.merge_payload(generated, cache_dir)
    assert merged["models"]["mlb_new"]["picks"][0]["result"] == "win"
    assert merged["models"]["mlb_new"]["picks"][0]["pregame_snapshot"]["probability"] == 0.61


def test_model_cache_merge_keeps_previous_same_date_picks_when_refresh_drops_them(tmp_path):
    module = _load_module("merge_model_cache_payload_keep_dropped", ROOT / "scripts" / "merge_model_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-06-16",
        "models": {
            "fifa_world_cup": {
                "picks": [
                    {"source": "FIFA Model", "sport": "FIFA WC", "date": "2026-06-16", "pick": "France ML", "matchup": "Senegal @ France"},
                    {"source": "FIFA Model", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ]
            }
        },
    }
    generated = {
        "date": "2026-06-16",
        "models": {
            "fifa_world_cup": {
                "picks": [
                    {"source": "FIFA Model", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ]
            }
        },
    }
    (cache_dir / "2026-06-16.json").write_text(json.dumps(current), encoding="utf-8")

    merged = module.merge_payload(generated, cache_dir)
    picks = merged["models"]["fifa_world_cup"]["picks"]

    assert [pick["pick"] for pick in picks] == ["Norway ML", "France ML"]


def test_model_cache_merge_replaces_stale_same_game_market_pick(tmp_path):
    module = _load_module("merge_model_cache_payload_replace_market", ROOT / "scripts" / "merge_model_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-06-22",
        "models": {
            "fifa_world_cup": {
                "picks": [
                    {
                        "source": "FIFA Model",
                        "sport": "FIFA WC",
                        "date": "2026-06-22",
                        "market": "total",
                        "pick": "Over 2.5 (Algeria @ Jordan)",
                        "matchup": "Algeria @ Jordan",
                    },
                    {
                        "source": "FIFA Model",
                        "sport": "FIFA WC",
                        "date": "2026-06-22",
                        "market": "total",
                        "pick": "Under 2.5 (Completed @ Match)",
                        "matchup": "Completed @ Match",
                        "result": "win",
                    },
                ]
            }
        },
    }
    generated = {
        "date": "2026-06-22",
        "models": {
            "fifa_world_cup": {
                "picks": [
                    {
                        "source": "FIFA Model",
                        "sport": "FIFA WC",
                        "date": "2026-06-22",
                        "market": "total",
                        "pick": "Under 2.5 (Algeria @ Jordan)",
                        "matchup": "Algeria @ Jordan",
                    },
                ]
            }
        },
    }
    (cache_dir / "2026-06-22.json").write_text(json.dumps(current), encoding="utf-8")

    merged = module.merge_payload(generated, cache_dir)
    picks = merged["models"]["fifa_world_cup"]["picks"]

    assert [pick["pick"] for pick in picks] == [
        "Under 2.5 (Algeria @ Jordan)",
        "Under 2.5 (Completed @ Match)",
    ]


def test_external_feed_merge_replaces_previous_same_date_picks_when_refresh_drops_them(tmp_path):
    module = _load_module("merge_external_feed_cache_payload_replace_dropped", ROOT / "scripts" / "merge_external_feed_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-06-16",
        "models": {
            "sportytrader_fifa_world_cup": {
                "ok": True,
                "picks": [
                    {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Both teams to score", "matchup": "France vs Senegal"},
                    {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ],
            }
        },
        "external_feeds": {
            "sportytrader_fifa_world_cup": {
                "ok": True,
                "picks": [
                    {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Both teams to score", "matchup": "France vs Senegal"},
                    {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ],
            }
        },
        "sportytrader_fifa_world_cup": {
            "ok": True,
            "picks": [
                {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Both teams to score", "matchup": "France vs Senegal"},
                {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
            ],
        },
    }
    generated = {
        "date": "2026-06-16",
        "models": {
            "sportytrader_fifa_world_cup": {
                "ok": True,
                "picks": [
                    {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ],
            }
        },
        "external_feeds": {
            "sportytrader_fifa_world_cup": {
                "ok": True,
                "picks": [
                    {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ],
            }
        },
        "sportytrader_fifa_world_cup": {
            "ok": True,
            "picks": [
                {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
            ],
        },
    }
    (cache_dir / "2026-06-16.json").write_text(json.dumps(current), encoding="utf-8")

    merged = module.merge_payload(generated, cache_dir)
    picks = merged["models"]["sportytrader_fifa_world_cup"]["picks"]

    assert [pick["pick"] for pick in picks] == ["Norway ML"]
    assert [pick["pick"] for pick in merged["external_feeds"]["sportytrader_fifa_world_cup"]["picks"]] == ["Norway ML"]
    assert [pick["pick"] for pick in merged["sportytrader_fifa_world_cup"]["picks"]] == ["Norway ML"]


def test_external_feed_merge_migrates_legacy_provider_bucket_to_split_key(tmp_path):
    module = _load_module("merge_external_feed_cache_payload_legacy_split", ROOT / "scripts" / "merge_external_feed_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-06-16",
        "models": {
            "sportytrader": {
                "ok": True,
                "picks": [
                    {"source": "SportyTrader", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Both teams to score", "matchup": "France vs Senegal"},
                    {"source": "SportyTrader", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ],
            }
        },
        "external_feeds": {
            "sportytrader": {
                "ok": True,
                "picks": [
                    {"source": "SportyTrader", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Both teams to score", "matchup": "France vs Senegal"},
                    {"source": "SportyTrader", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ],
            }
        },
        "sportytrader": {
            "ok": True,
            "picks": [
                {"source": "SportyTrader", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Both teams to score", "matchup": "France vs Senegal"},
                {"source": "SportyTrader", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
            ],
        },
    }
    generated = {
        "date": "2026-06-16",
        "models": {
            "sportytrader_fifa_world_cup": {
                "ok": True,
                "picks": [
                    {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ],
            }
        },
        "external_feeds": {
            "sportytrader_fifa_world_cup": {
                "ok": True,
                "picks": [
                    {"source": "SportyTraderFIFAWorldCup", "sport": "FIFA WC", "date": "2026-06-16", "pick": "Norway ML", "matchup": "Norway @ Iraq"},
                ],
            }
        },
    }
    (cache_dir / "2026-06-16.json").write_text(json.dumps(current), encoding="utf-8")

    merged = module.merge_payload(generated, cache_dir)
    picks = merged["models"]["sportytrader_fifa_world_cup"]["picks"]

    assert "sportytrader" not in merged["models"]
    assert "sportytrader" not in merged["external_feeds"]
    assert "sportytrader" not in merged
    assert [pick["pick"] for pick in picks] == ["Norway ML"]
    assert {pick["source"] for pick in picks} == {"SportyTraderFIFAWorldCup"}


def test_player_prop_merge_does_not_carry_results_across_rank_epochs(tmp_path):
    module = _load_module("merge_player_props_cache_payload_epochs", ROOT / "scripts" / "merge_player_props_cache_payload.py")
    cache_dir = tmp_path / "data" / "player_props_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-06-16",
        "models": {
            "mlb_player_props": {
                "picks": [{
                    "id": "same-prop",
                    "source": "PickLedgerPro In-House Player Props",
                    "sport": "MLB",
                    "date": "2026-06-16",
                    "pick": "Player Over 0.5 Hits",
                    "matchup": "Away @ Home",
                    "ml_rank_epoch": "MLB:old",
                    "result": "win",
                }]
            }
        },
    }
    generated = {
        "date": "2026-06-16",
        "models": {
            "mlb_player_props": {
                "picks": [{
                    "id": "same-prop",
                    "source": "PickLedgerPro In-House Player Props",
                    "sport": "MLB",
                    "date": "2026-06-16",
                    "pick": "Player Over 0.5 Hits",
                    "matchup": "Away @ Home",
                    "ml_rank_epoch": "MLB:new",
                    "result": "pending",
                }]
            }
        },
    }
    (cache_dir / "2026-06-16.json").write_text(json.dumps(current), encoding="utf-8")

    merged = module.merge_payload(generated, cache_dir)

    assert merged["models"]["mlb_player_props"]["picks"][0]["result"] == "pending"


def test_player_prop_merge_keeps_snapshot_only_props_out_of_latest_board(tmp_path):
    module = _load_module("merge_player_props_cache_payload_current_board", ROOT / "scripts" / "merge_player_props_cache_payload.py")
    cache_dir = tmp_path / "data" / "player_props_cache"
    snapshot_dir = tmp_path / "data" / "player_props_snapshots"
    cache_dir.mkdir(parents=True)
    (snapshot_dir / "2026-06-20").mkdir(parents=True)
    previous = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "picks": [
                    {
                        "id": "old",
                        "scope": "player",
                        "source": "MLBPlayerProps",
                        "sport": "MLB",
                        "date": "2026-06-20",
                        "game_id": "1",
                        "player_id": "10",
                        "stat_key": "hits",
                        "selection": "Over",
                        "line": 0.5,
                        "pick": "Old Over 0.5 Hits",
                        "matchup": "A @ B",
                        "market_priced": True,
                        "probability_source": "player_props_ml_v1",
                        "decision": "LEAN",
                        "ml_model_version": "player_props_consensus_v2.0.0",
                        "ml_probability_mode": "four_model_consensus_gate",
                        "consensus_qualified": True,
                        "result": "pending",
                    }
                ],
            }
        },
    }
    generated = {
        "date": "2026-06-20",
        "models": {
            "mlb_player_props": {
                "ok": True,
                "picks": [
                    {
                        "id": "new",
                        "scope": "player",
                        "source": "MLBPlayerProps",
                        "model_key": "mlb_player_props",
                        "sport": "MLB",
                        "date": "2026-06-20",
                        "game_id": "2",
                        "player_id": "20",
                        "stat_key": "hits_runs_rbis",
                        "selection": "Under",
                        "line": 1.5,
                        "pick": "New Under 1.5 HRR",
                        "matchup": "C @ D",
                        "market_priced": True,
                        "probability_source": "player_props_ml_v1",
                        "decision": "LEAN",
                        "ml_model_version": "player_props_consensus_v2.0.0",
                        "ml_probability_mode": "four_model_consensus_gate",
                        "consensus_qualified": True,
                    }
                ],
            }
        },
    }
    (cache_dir / "2026-06-20.json").write_text(json.dumps({"date": "2026-06-20", "models": {}}), encoding="utf-8")
    (snapshot_dir / "2026-06-20" / "snapshot.json").write_text(json.dumps(previous), encoding="utf-8")

    merged = module.merge_payload(generated, cache_dir, snapshot_dir)
    picks = merged["models"]["mlb_player_props"]["picks"]

    assert {pick["id"] for pick in picks} == {"new"}
    assert {pick["source"] for pick in picks} == {"MLBPlayerProps"}
    assert {pick["model_key"] for pick in picks} == {"mlb_player_props"}
    assert all("carried_forward" not in pick for pick in picks)


def test_player_prop_merge_keeps_legacy_variant_snapshots_archive_only(tmp_path):
    module = _load_module("merge_player_props_cache_payload_legacy_variants", ROOT / "scripts" / "merge_player_props_cache_payload.py")
    cache_dir = tmp_path / "data" / "player_props_cache"
    snapshot_dir = tmp_path / "data" / "player_props_snapshots"
    cache_dir.mkdir(parents=True)
    (snapshot_dir / "2026-06-24").mkdir(parents=True)
    snapshot = {
        "date": "2026-06-24",
        "models": {
            "wnba_player_props_all_time": {
                "ok": True,
                "picks": [
                    {
                        "id": "pp_michaela_all_time",
                        "scope": "player",
                        "source": "WNBA All Time Props",
                        "sport": "WNBA",
                        "date": "2026-06-24",
                        "game_id": "401857018",
                        "player_id": "4281173",
                        "stat_key": "points",
                        "selection": "Over",
                        "line": 9.5,
                        "pick": "Michaela Onyenwere Over 9.5 Points",
                        "matchup": "A @ B",
                        "market_priced": True,
                        "probability_source": "player_props_ml_v1",
                        "decision": "LEAN",
                        "ml_model_version": "player_props_consensus_v2.0.0",
                        "ml_probability_mode": "four_model_consensus_gate",
                        "consensus_qualified": True,
                        "model_variant": "all_time",
                        "result": "pending",
                    }
                ],
            }
        },
    }
    generated = {
        "date": "2026-06-24",
        "models": {
            "wnba_player_props": {
                "ok": True,
                "ranking_epoch": "WNBA:player_props_consensus_v2.0.0:published:test",
                "picks": [],
            }
        },
    }
    (cache_dir / "2026-06-24.json").write_text(json.dumps({"date": "2026-06-24", "models": {}}), encoding="utf-8")
    (snapshot_dir / "2026-06-24" / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    merged = module.merge_payload(generated, cache_dir, snapshot_dir)
    picks = merged["models"]["wnba_player_props"]["picks"]

    assert picks == []


def test_external_feed_merge_does_not_promote_partial_cache_to_latest(tmp_path):
    module = _load_module("merge_external_feed_cache_payload", ROOT / "scripts" / "merge_external_feed_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    previous = {"date": "2026-06-14", "models": {"mlb_new": {"ok": True, "picks": []}}}
    partial = {
        "date": "2026-06-15",
        "models": {"scores24_mlb": {"ok": True, "picks": [{"pick": "Cubs ML"}]}},
    }
    (cache_dir / "latest.json").write_text(json.dumps(previous), encoding="utf-8")

    merged = module.merge_payload(partial, cache_dir)
    latest_updated = module.write_merged_payload(merged, cache_dir)

    assert latest_updated is False
    assert json.loads((cache_dir / "latest.json").read_text(encoding="utf-8"))["date"] == "2026-06-14"
    assert json.loads((cache_dir / "2026-06-15.json").read_text(encoding="utf-8"))["models"]["scores24_mlb"]["ok"] is True


def test_external_feed_merge_promotes_complete_cache_to_latest(tmp_path):
    module = _load_module("merge_external_feed_cache_payload_complete", ROOT / "scripts" / "merge_external_feed_cache_payload.py")
    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    complete = {
        "date": "2026-06-15",
        "models": {
            key: {"ok": True, "picks": []}
            for key in module.REQUIRED_TEAM_MODEL_KEYS
        },
    }

    latest_updated = module.write_merged_payload(complete, cache_dir)

    assert latest_updated is True
    assert json.loads((cache_dir / "latest.json").read_text(encoding="utf-8"))["date"] == "2026-06-15"


def test_rankings_tab_renders_profit_desk_qualification_board():
    main = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "src" / "styles" / "pickledger.css").read_text(encoding="utf-8")

    assert 'id="profit-qualification-section"' in html
    assert 'id="profit-qualification-board"' in html
    assert "Profit Desk Qualification" in html
    assert "function renderProfitQualificationBoard(" in main
    assert "renderProfitQualificationBoard();" in main
    assert "QUALIFICATION_GATE_LABELS" in main
    assert "ON THE CARD" in main
    assert "GATES CLEARED" in main
    assert "export interface ProfitDeskSourceCard" in data
    assert "sources?: ProfitDeskSourceCard[]" in data
    assert ".qual-card" in css
    assert ".qual-gates" in css
    assert ".qual-card.is-qualified::before" in css
