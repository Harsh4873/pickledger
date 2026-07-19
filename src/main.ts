import { initMobileMode, initPickMode, initSettingsUI, initTheme, type PickMode } from './settings';
import {
  getAllPicks,
  getCacheStatus,
  getParlayCardsPayload,
  getParlayCardPayloads,
  getProfitDeskPayload,
  loadAllData,
  setPickMode as setDataPickMode,
  setLocalGameTime,
  setLocalResult,
  type ParlayCard,
  type ParlayCardsPayload,
  type ParlayRanking,
  type Pick,
  type PickResult,
  type ProfitDeskBlocker,
  type ProfitDeskCandidate,
  type ProfitDeskModeSummary,
  type ProfitDeskPayload,
} from './data';

type Stats = {
  total: number;
  wins: number;
  losses: number;
  pushes: number;
  pending: number;
  net: number;
  risk: number;
  priced: number;
  winRate: number | null;
  roi: number | null;
};

type HomeScoreInfo = {
  eventId: string;
  sport: string;
  tone: 'pregame' | 'live' | 'final' | 'delayed';
  text: string;
  startTime: string;
};

type TrendSignalGroup = {
  key: string;
  label: string;
  picks: Pick[];
  matching: boolean;
  pass: boolean;
};

type DailySourceForm = {
  source: string;
  recentStats: Stats;
  lastStats: Stats;
  recentDates: string[];
  todayCalls: Pick[];
  score: number;
};

type WeekdaySourceForm = {
  source: string;
  stats: Stats;
  decided: number;
  todayCalls: Pick[];
  score: number;
};

type DailyView = 'picks' | 'consensus' | 'sources' | 'research' | 'dayform';
type ProfitView = 'card' | 'watchlist' | 'method';
type ParlayView = string;
type DailySort = 'time' | 'percentage';
type ResultMode = 'pending' | 'all' | 'settled';
type ViewOption<Key extends string> = {
  key: Key;
  label: string;
  count: number;
  description: string;
};
type DailyViewOption = ViewOption<DailyView>;
type ParlayViewOption = ViewOption<ParlayView>;

type DailyPickGroup = {
  key: string;
  picks: Pick[];
  primary: Pick;
  tags: string[];
  score: number;
};

const ESPN_ENDPOINTS: Record<string, [string, string]> = {
  MLB: ['baseball', 'mlb'],
  NBA: ['basketball', 'nba'],
  'NBA SUMMER': ['basketball', 'nba-summer'],
  WNBA: ['basketball', 'wnba'],
  'FIFA WC': ['soccer', 'fifa.world'],
  NHL: ['hockey', 'nhl'],
};

const activeFilters = new Set<string>();
let activePickMode: PickMode = 'team';
let homeMode: ResultMode = 'pending';
let dailyView: DailyView = 'picks';
let profitView: ProfitView = 'card';
let profitDeskSport = 'ALL';
let parlayView: ParlayView = 'all';
let parlayResultMode: ResultMode = 'pending';
let dailySort: DailySort = 'time';
let selectedDate = '';
let followCentralToday = true;
let calendarMonth = '';
let calendarOpen = false;
let dailyCalendarOpen = false;
let profitCalendarOpen = false;
let parlayCalendarOpen = false;
let filterMoreOpen = false;
let refreshInFlight = false;
const homeScores = new Map<string, HomeScoreInfo>();
const homeScoreFetches = new Map<string, number>();
const expandedSourceKeys = new Set<string>();
const expandedResearchPickKeys = new Set<string>();
let homeScoreRefreshKey = '';
let latestPicksUpdatedAt = '';
const HOME_SCORE_TTL_MS = 45_000;
const DISPLAY_TIME_ZONE = 'America/Chicago';
const AUTO_REFRESH_MS = 5 * 60_000;
const PLAYER_PROP_RANKING_START_DATE = '2026-06-23';
const MLB_TEAM_CONSENSUS_EPOCH_PREFIX = 'MLB:mlb_team_consensus_v1';
const MLB_TEAM_CONSENSUS_SOURCES = new Set([
  'MLB Model', 'MLB ML', 'MLB Total',
  'MLB First Five', 'MLB F5', 'MLB F5 Total',
  'MLB Inning', 'MLB Team Total',
]);
// The 2026-07-19 board rebuild: rankings restart from this date so stale
// records don't carry into the redesigned source split. The proven MLB
// moneyline/total split (formerly "MLB Model") keeps its full
// consensus-era history — that record is the one worth preserving.
const TEAM_RANKING_START_DATE = '2026-07-19';
// "MLB Model" stays listed as a safety net: any mlb_new row whose market
// tag fails the ML/Total split must never drop out of the record.
const LEGACY_RECORD_SOURCES = new Set(['MLB ML', 'MLB Total', 'MLB Model']);
// WNBA redesign (2026-07-19): the proven moneyline record carries over as
// WNBA ML; the rebuilt spread/total variants (and any stray legacy label)
// restart from the redesign date.
const WNBA_RESET_SOURCES = new Set(['WNBA Model', 'WNBA Spread', 'WNBA Total']);
const PRIMARY_FILTERS = ['ALL', 'MLB', 'WNBA'];
let lastCentralDate = '';

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function calendarDateKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function centralDateKey(date = new Date()): string {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: DISPLAY_TIME_ZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function parseDateKey(value: string): Date | null {
  const match = String(value || '').match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return null;
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return Number.isNaN(date.getTime()) ? null : date;
}

function pickDateKey(pick: Pick): string {
  const raw = String(pick.date || '').trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? '' : centralDateKey(parsed);
}

function dateLabel(key: string, long = false): string {
  const date = parseDateKey(key);
  if (!date) return key;
  return date.toLocaleDateString('en-US', long
    ? { weekday: 'long', month: 'long', day: 'numeric' }
    : { month: 'short', day: 'numeric' });
}

function formatStart(value: unknown): string {
  const date = new Date(String(value || ''));
  if (Number.isNaN(date.getTime())) return 'TBD';
  return date.toLocaleTimeString('en-US', {
    timeZone: DISPLAY_TIME_ZONE,
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short',
  });
}

function formatOdds(pick: Pick): string {
  if (pick.odds != null) return pick.odds > 0 ? `+${pick.odds}` : String(pick.odds);
  const probability = Number(pick.probability);
  return Number.isFinite(probability)
    ? `${Math.round((probability <= 1 ? probability * 100 : probability))}%`
    : '';
}

function detailValues(value: unknown): string[] {
  if (Array.isArray(value)) return value.flatMap(detailValues);
  if (value == null || value === '') return [];
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return detailValues(record.reason || record.factor || record.label || record.name);
  }
  return [String(value).trim()].filter(Boolean);
}

function consensusModelLabels(pick: Pick): string[] {
  const raw = pick.consensus_models;
  if (Array.isArray(raw)) {
    return raw.map(value => String(value || '').trim()).filter(Boolean);
  }
  const count = Number(pick.consensus_model_count);
  return Number.isFinite(count) && count > 0 ? [`${count} model consensus`] : [];
}

function consensusApplicableModelLabels(pick: Pick): string[] {
  for (const field of ['consensus_applicable_models', 'consensus_record_models']) {
    const raw = pick[field];
    if (Array.isArray(raw)) {
      const labels = raw.map(value => String(value || '').trim()).filter(Boolean);
      if (labels.length) return labels;
    }
  }
  const sportPrefix = String(pick.sport || '').trim().toLowerCase();
  const labels = consensusModelLabels(pick);
  if (!sportPrefix) return labels;
  const applicable = labels.filter(label => label.toLowerCase().startsWith(`${sportPrefix}_`));
  return applicable.length ? applicable : labels;
}

function consensusModelName(label: string): string {
  return label.split(':', 1)[0].replace(/_/g, ' ').trim().toUpperCase();
}

function playerRankingNames(pick: Pick): string[] {
  return [sourceName(pick)];
}

function rankingBucketNames(pick: Pick): string[] {
  return activePickMode === 'player' ? playerRankingNames(pick) : [sourceName(pick)];
}

function picksForRankingBucket(picks: Pick[], bucketName: string): Pick[] {
  return uniquePlayerRankingPicks(picks.filter(pick => rankingBucketNames(pick).includes(bucketName)));
}

function addPickToRankingBuckets(buckets: Map<string, Pick[]>, pick: Pick): void {
  rankingBucketNames(pick).forEach(name => {
    buckets.set(name, [...(buckets.get(name) || []), pick]);
  });
}

function consensusModelPanelHtml(pick: Pick): string {
  const models = consensusModelLabels(pick);
  if (!models.length) return '';
  return `<div class="home-player-model-stack">
    <div class="home-player-model-stack-title">${models.length} ACTIVE MODELS</div>
    <div class="home-player-model-list">${models.map(model => `<span title="${escapeHtml(model)}">${escapeHtml(consensusModelName(model))}</span>`).join('')}</div>
  </div>`;
}

function formatPlayerMeasure(value: unknown, percent = false): string {
  if (value == null || value === '') return '';
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  if (percent) {
    const percentage = Math.abs(number) <= 1 ? number * 100 : number;
    return `${Number(percentage.toFixed(1))}%`;
  }
  return String(Number(number.toFixed(2)));
}

function pickExpandableDetails(pick: Pick): { reason: string; factors: string[] } {
  return {
    reason: detailValues(pick.reason ?? pick.rationale)[0] || '',
    factors: detailValues(pick.key_factors),
  };
}

function researchDetailsHtml(pick: Pick, expanded: boolean): string {
  const decision = String(pick.decision || 'PASS').trim().toUpperCase();
  const kellyUnits = pick.kelly_units ?? pick.recommended_units;
  const kelly = kellyUnits ?? pick.kelly;
  const fullKelly = formatPlayerMeasure(pick.full_kelly, true);
  const quarterKelly = formatPlayerMeasure(pick.quarter_kelly, true);
  const kellyText = kellyUnits != null
    ? `${formatPlayerMeasure(kellyUnits)}u`
    : formatPlayerMeasure(kelly, true);
  const confidence = formatPlayerMeasure(pick.confidence, true);
  const { reason, factors } = pickExpandableDetails(pick);
  const modelPanel = activePickMode === 'player' ? consensusModelPanelHtml(pick) : '';
  const modelCount = consensusModelLabels(pick).length;
  const hasExtra = Boolean(reason || factors.length || modelPanel);
  if (!hasExtra && activePickMode !== 'player') return '';
  return `<div class="home-player-details">
    ${activePickMode === 'player' ? `<div class="home-player-metrics">
      <span class="home-player-decision decision-${escapeHtml(decision.toLowerCase())}">${escapeHtml(decision)}</span>
      ${modelCount ? `<span><strong>Models</strong>${modelCount}</span>` : ''}
      ${quarterKelly ? `<span><strong>Quarter Kelly</strong>${escapeHtml(quarterKelly)}</span>` : ''}
      ${fullKelly ? `<span><strong>Full Kelly</strong>${escapeHtml(fullKelly)}</span>` : ''}
      ${!quarterKelly && !fullKelly && kellyText ? `<span><strong>Kelly</strong>${escapeHtml(kellyText)}</span>` : ''}
      ${confidence ? `<span><strong>Confidence</strong>${escapeHtml(confidence)}</span>` : ''}
    </div>` : ''}
    ${hasExtra ? `<div class="home-player-expand-control"><span data-player-expand-label>${expanded ? 'Hide research details' : 'Show research details'}</span><span class="home-player-expand-icon" aria-hidden="true">&#9662;</span></div>` : ''}
    ${hasExtra ? `<div class="home-player-extra">
      ${modelPanel}
      ${reason ? `<div class="home-player-reason"><strong>Reason</strong><span>${escapeHtml(reason)}</span></div>` : ''}
      ${factors.length ? `<div class="home-player-factors"><strong>Key factors</strong><div>${factors.map(factor => `<span>${escapeHtml(factor)}</span>`).join('')}</div></div>` : ''}
    </div>` : ''}
  </div>`;
}

function sourceName(pick: Pick): string {
  return String(pick.source || 'Unknown').trim();
}

function playerRankingEpoch(pick: Pick): string {
  return String(pick.ml_rank_epoch || pick.ranking_epoch || pick.model_epoch || '').trim();
}

function playerRankingPickIdentity(pick: Pick): string {
  return [
    pick.sport,
    pickDateKey(pick),
    pick.game_id || pick.event_id || gameName(pick),
    pick.player_id || pick.market_athlete_id || pick.player_name || pick.player,
    pick.stat_key || pick.market_type || pick.market,
    pick.selection || pick.pick,
    pick.line ?? pick.market_line ?? '',
  ].map(value => String(value ?? '').trim().toLowerCase()).join('::');
}

function uniquePlayerRankingPicks(picks: Pick[]): Pick[] {
  if (activePickMode !== 'player') return picks;
  const byMarket = new Map<string, Pick>();
  picks.forEach(pick => {
    const key = playerRankingPickIdentity(pick);
    const current = byMarket.get(key);
    if (!current || (playerModelRank(pick) ?? 9999) < (playerModelRank(current) ?? 9999)) {
      byMarket.set(key, pick);
    }
  });
  return [...byMarket.values()];
}

function activePlayerRankingEpochs(): Map<string, string> {
  const latest = new Map<string, { date: string; epoch: string; updatedAt: string }>();
  getAllPicks().forEach(pick => {
    const epoch = playerRankingEpoch(pick);
    if (!epoch) return;
    const date = pickDateKey(pick);
    const updatedAt = String(pick.ranking_updated_at || '').trim();
    const current = latest.get(pick.sport);
    if (!current || date > current.date || (date === current.date && updatedAt >= current.updatedAt)) {
      latest.set(pick.sport, { date, epoch, updatedAt });
    }
  });
  return new Map([...latest.entries()].map(([sport, value]) => [sport, value.epoch]));
}

function isSettledPick(pick: Pick): boolean {
  return pick.result !== 'pending';
}

function isUnsupportedPendingPick(pick: Pick): boolean {
  return pick.result === 'pending' && pick.grade_supported === false;
}

function isOpenPick(pick: Pick): boolean {
  return pick.result === 'pending' && !isUnsupportedPendingPick(pick) && isPublishedDailyPick(pick);
}

function rankingComparablePicks(picks: Pick[]): Pick[] {
  if (activePickMode !== 'player') {
    return picks.filter(pick => {
      const source = sourceName(pick);
      const isConsensusSource = String(pick.sport || '').toUpperCase() === 'MLB'
        && MLB_TEAM_CONSENSUS_SOURCES.has(source);
      // The 2026-07-19 record reset applies ONLY to the in-house MLB and
      // WNBA model variants. External feeds and every other source keep
      // their full history.
      if (WNBA_RESET_SOURCES.has(source)) {
        return pickDateKey(pick) >= TEAM_RANKING_START_DATE;
      }
      if (!isConsensusSource) return true;
      const epoch = String(pick.ml_rank_epoch || pick.ranking_epoch || pick.model_epoch || '').trim();
      if (!epoch.startsWith(MLB_TEAM_CONSENSUS_EPOCH_PREFIX)) return false;
      if (LEGACY_RECORD_SOURCES.has(source)) return true;
      return pickDateKey(pick) >= TEAM_RANKING_START_DATE;
    });
  }
  return uniquePlayerRankingPicks(picks.filter(pick => {
    const date = pickDateKey(pick);
    return date >= PLAYER_PROP_RANKING_START_DATE;
  }));
}

function latestAvailableDateKey(picks = getAllPicks()): string {
  return [...new Set(picks.map(pickDateKey).filter(Boolean))].sort().at(-1) || centralDateKey();
}

function playerModelRank(pick: Pick): number | null {
  const rank = Number(pick.ml_rank ?? pick.model_rank ?? pick.rank);
  return Number.isFinite(rank) && rank > 0 ? rank : null;
}

function rankingWindowLabel(): string {
  return activePickMode === 'player'
    ? `SINCE ${dateLabel(PLAYER_PROP_RANKING_START_DATE).toUpperCase()}`
    : 'ALL TIME';
}

function rankingBucketScopeLabel(bucketName: string): string {
  if (activePickMode === 'player') return rankingWindowLabel();
  if (LEGACY_RECORD_SOURCES.has(bucketName)) {
    return 'ALL TIME | MLB TEAM CONSENSUS V1';
  }
  if (MLB_TEAM_CONSENSUS_SOURCES.has(bucketName) || WNBA_RESET_SOURCES.has(bucketName)) {
    return `SINCE ${dateLabel(TEAM_RANKING_START_DATE).toUpperCase()}`;
  }
  return 'ALL TIME';
}

function gameName(pick: Pick): string {
  const explicit = String(pick.matchup || pick.game || '').trim();
  if (explicit) return explicit;
  if (pick.away_team && pick.home_team) return `${pick.away_team} vs ${pick.home_team}`;
  if (pick.team) return String(pick.team);
  if (pick.player) return String(pick.player);
  const parenthetical = pick.pick.match(/\(([^)]+(?:vs|@)[^)]+)\)/i);
  return parenthetical?.[1] || pick.pick;
}

function gameKey(pick: Pick): string {
  const teams = teamsForPick(pick);
  const matchup = teams
    ? teams.map(canonicalTeamToken).sort().join('|')
    : normalizeTeam(gameName(pick));
  return `${pick.sport}::${pickDateKey(pick)}::${matchup}`;
}

function statsFor(picks: Pick[]): Stats {
  const wins = picks.filter(pick => pick.result === 'win').length;
  const losses = picks.filter(pick => pick.result === 'loss').length;
  const pushes = picks.filter(pick => pick.result === 'push').length;
  const pending = picks.filter(isOpenPick).length;
  const decided = wins + losses;
  const pricedPicks = picks.filter(pick => pick.price_verified === true && pick.odds != null && pick.odds !== 0);
  const net = Number(pricedPicks.reduce((sum, pick) => sum + pick.pl, 0).toFixed(2));
  const risk = Number(pricedPicks.filter(pick => pick.result !== 'pending' && pick.result !== 'push')
    .reduce((sum, pick) => sum + pick.units, 0).toFixed(2));
  return {
    total: picks.length,
    wins,
    losses,
    pushes,
    pending,
    net,
    risk,
    priced: pricedPicks.filter(isSettledPick).length,
    winRate: decided ? wins / decided : null,
    roi: risk ? net / risk : null,
  };
}

function signedUnits(value: number): string {
  return `${value >= 0 ? '+' : ''}${Number(value.toFixed(2))}u`;
}

// P/L only counts picks settled at verified sportsbook prices. When a bucket
// has none, "+0u" would read as break-even, so show an untracked marker instead.
function trackedUnits(stats: Stats): string {
  return stats.priced ? signedUnits(stats.net) : '—';
}

function trackedUnitsClass(stats: Stats): string {
  if (!stats.priced || stats.net === 0) return 'neutral';
  return stats.net > 0 ? 'positive' : 'negative';
}

function shiftedDateKey(key: string, days: number): string {
  const date = parseDateKey(key);
  if (!date) return key;
  date.setDate(date.getDate() + days);
  return calendarDateKey(date);
}

function sourceRecordText(picks: Pick[]): string {
  const stats = statsFor(picks);
  const record = `${stats.wins}-${stats.losses}${stats.pushes ? `-${stats.pushes}` : ''}`;
  return [
    record,
    stats.priced ? signedUnits(stats.net) : '',
    stats.winRate == null ? '' : `${(stats.winRate * 100).toFixed(1)}%`,
    stats.pending ? `${stats.pending} open` : '',
  ].filter(Boolean).join(' | ');
}

function sourceRecordLines(picks: Pick[], anchorDate = centralDateKey()): Array<{ label: string; text: string }> {
  const today = anchorDate || centralDateKey();
  const yesterday = shiftedDateKey(today, -1);
  const lastSevenStart = shiftedDateKey(today, -6);
  const forDate = (key: string): Pick[] => picks.filter(pick => pickDateKey(pick) === key);
  return [
    { label: 'TODAY', text: sourceRecordText(forDate(today)) },
    { label: 'YESTERDAY', text: sourceRecordText(forDate(yesterday)) },
    {
      label: 'LAST 7 DAYS',
      text: sourceRecordText(picks.filter(pick => {
        const key = pickDateKey(pick);
        return key >= lastSevenStart && key <= today;
      })),
    },
    { label: 'ALL TIME', text: sourceRecordText(picks) },
  ];
}

function resultBadge(result: PickResult): string {
  return `<span class="badge badge-${result}">${result === 'pending' ? 'OPEN' : result.toUpperCase()}</span>`;
}

function pickResultBadge(pick: Pick): string {
  if (isUnsupportedPendingPick(pick)) return '<span class="badge badge-pending">UNTRACKED</span>';
  if (pick.result === 'pending' && dailyDecision(pick) === 'PASS') return '<span class="badge badge-pending">PASS</span>';
  return resultBadge(pick.result);
}

function statusClass(picks: Pick[]): string {
  if (picks.some(isOpenPick)) return 'live';
  const results = new Set(picks.map(pick => pick.result));
  if (results.size > 1) return 'mixed';
  return picks[0]?.result || 'live';
}

function pickStartTimestamp(pick: Pick): number | null {
  const value = String(pick.start_time || pick.game_start_time || '').trim();
  if (!value) return null;
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : null;
}

function gameStartTimestamp(picks: Pick[]): number | null {
  const timestamps = picks.map(pickStartTimestamp).filter((value): value is number => value != null);
  return timestamps.length ? Math.min(...timestamps) : null;
}

function compareStartAsc(left: number | null, right: number | null): number {
  if (left == null && right == null) return 0;
  if (left == null) return 1;
  if (right == null) return -1;
  return left - right;
}

function startBucket(timestamp: number | null, now = Date.now()): number {
  if (timestamp == null) return 1;
  return timestamp > now ? 0 : 2;
}

function compareActionableStart(left: number | null, right: number | null, now = Date.now()): number {
  const leftBucket = startBucket(left, now);
  const rightBucket = startBucket(right, now);
  if (leftBucket !== rightBucket) return leftBucket - rightBucket;
  if (left == null && right == null) return 0;
  if (left == null) return 1;
  if (right == null) return -1;
  return leftBucket === 2 ? right - left : left - right;
}

function compareGameStartAsc(left: Pick[], right: Pick[]): number {
  return compareStartAsc(gameStartTimestamp(left), gameStartTimestamp(right))
    || (left[0] ? gameName(left[0]) : '').localeCompare(right[0] ? gameName(right[0]) : '');
}

function comparePickActionableStart(left: Pick, right: Pick): number {
  return compareActionableStart(pickStartTimestamp(left), pickStartTimestamp(right))
    || gameName(left).localeCompare(gameName(right))
    || left.pick.localeCompare(right.pick);
}

function homeDecisionRank(pick: Pick): number {
  const decision = dailyDecision(pick);
  if (decision === 'BET') return 0;
  if (decision === 'LEAN') return 1;
  if (decision === 'PASS') return 2;
  return 3;
}

function compareHomePickRows(left: Pick, right: Pick): number {
  if (activePickMode === 'player') {
    const leftRank = playerModelRank(left) ?? 9999;
    const rightRank = playerModelRank(right) ?? 9999;
    if (leftRank !== rightRank) return leftRank - rightRank;
  }
  return homeDecisionRank(left) - homeDecisionRank(right)
    || (pickProbability(right) || 0) - (pickProbability(left) || 0)
    || (pickEdgePercent(right) || 0) - (pickEdgePercent(left) || 0)
    || sourceName(left).localeCompare(sourceName(right))
    || left.pick.localeCompare(right.pick);
}

function filterLabel(filter: string): string {
  if (filter === 'NBA SUMMER') return 'SUMMER';
  return filter === 'FIFA WC' ? 'FIFA' : filter;
}

function filterActive(filter: string): boolean {
  return filter === 'ALL' ? activeFilters.size === 0 : activeFilters.has(filter);
}

function toggleHomeFilter(filter: string): void {
  if (filter === 'ALL') {
    activeFilters.clear();
    return;
  }
  if (activeFilters.has(filter)) activeFilters.delete(filter);
  else activeFilters.add(filter);
}

function activeFilterSummary(): string {
  const labels = [...activeFilters].map(filter => filterLabel(filter).toUpperCase());
  if (!labels.length) return 'ALL SOURCES';
  if (labels.length <= 2) return labels.join(' + ');
  return `${labels[0]} + ${labels.length - 1} MORE`;
}

function ensureSelection(): void {
  const dates = [...new Set(getAllPicks().map(pickDateKey).filter(Boolean))].sort();
  const today = centralDateKey();
  // Stay pinned to Central "today" while following the live slate — even when the
  // active mode is empty (e.g. no player props on All-Star day). Falling back to a
  // prior date made Team Home look like Monday while Summer/WNBA/FIFA lived on today.
  if (followCentralToday) selectedDate = today;
  else if (!selectedDate || !dates.includes(selectedDate)) selectedDate = dates.at(-1) || today;
  if (!calendarMonth) calendarMonth = selectedDate.slice(0, 7);
}

function filteredPicks(): Pick[] {
  return getAllPicks().filter(pick => (
    activeFilters.size === 0 ||
    activeFilters.has(pick.sport) ||
    activeFilters.has(sourceName(pick))
  ));
}

function boardPicks(): Pick[] {
  return filteredPicks().filter(pick => {
    if (pickDateKey(pick) !== selectedDate) return false;
    if (homeMode === 'pending') return isOpenPick(pick);
    if (homeMode === 'settled') return pick.result !== 'pending';
    return true;
  });
}

function setRefreshStatus(message: string, state = ''): void {
  const status = document.getElementById('sync-status');
  if (status) {
    status.textContent = message;
    status.classList.toggle('ok', state === 'ok');
    status.classList.toggle('error', state === 'error');
  }
}

function renderFilters(): void {
  const container = document.getElementById('filter-bar');
  if (!container) return;
  const picks = getAllPicks();
  const available = [...new Set([
    ...picks.map(pick => pick.sport),
    ...picks.map(sourceName),
  ])];
  const extraFilters = available.filter(filter => !PRIMARY_FILTERS.includes(filter)).sort((a, b) => a.localeCompare(b));
  const filterButton = (filter: string): string => (
    `<button type="button" class="filter-btn ${filterActive(filter) ? 'active' : ''}" data-filter="${escapeHtml(filter)}" aria-pressed="${filterActive(filter)}">${escapeHtml(filterLabel(filter))}</button>`
  );
  const extraSelected = extraFilters.some(filterActive);
  container.innerHTML = `${PRIMARY_FILTERS.map(filterButton).join('')}
    <div class="filter-more-wrap" id="filter-more-wrap">
      <button type="button" class="filter-more-btn ${extraSelected ? 'has-selection' : ''}" id="filter-more-btn" aria-label="Show more sports and sources" aria-expanded="${filterMoreOpen}">+</button>
      <div class="filter-dropdown ${filterMoreOpen ? 'open' : ''}" id="filter-dropdown">
        ${extraFilters.length ? extraFilters.map(filterButton).join('') : '<div class="filter-dropdown-empty">No other sources in this view</div>'}
      </div>
    </div>`;
  container.querySelectorAll<HTMLButtonElement>('[data-filter]').forEach(button => {
    button.addEventListener('click', event => {
      event.stopPropagation();
      toggleHomeFilter(button.dataset.filter || 'ALL');
      render();
    });
  });
  document.getElementById('filter-more-btn')?.addEventListener('click', event => {
    event.stopPropagation();
    filterMoreOpen = !filterMoreOpen;
    renderFilters();
  });
}

function calendarHtml(): string {
  const monthDate = parseDateKey(`${calendarMonth}-01`) || parseDateKey(selectedDate) || new Date();
  const gridStart = new Date(monthDate.getFullYear(), monthDate.getMonth(), 1 - monthDate.getDay());
  const counts = new Map<string, number>();
  filteredPicks().forEach(pick => counts.set(pickDateKey(pick), (counts.get(pickDateKey(pick)) || 0) + 1));
  const days = Array.from({ length: 42 }, (_, index) => {
    const date = new Date(gridStart.getFullYear(), gridStart.getMonth(), gridStart.getDate() + index);
    const key = calendarDateKey(date);
    const count = counts.get(key) || 0;
    return `<button class="home-calendar-day ${date.getMonth() !== monthDate.getMonth() ? 'is-outside' : ''} ${key === centralDateKey() ? 'is-today' : ''} ${key === selectedDate ? 'is-selected' : ''} ${count ? 'has-picks' : ''}" data-date="${key}">
      <span class="home-calendar-day-num">${date.getDate()}</span>
      <span class="home-calendar-day-count">${count || '&middot;'}</span>
    </button>`;
  }).join('');
  return `<div class="home-date-popover-top">
    <div><div class="home-date-popover-label">Calendar View</div><div class="home-date-popover-month">${monthDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</div></div>
    <div class="home-date-nav-wrap"><button class="home-date-nav" data-month-shift="-1">&#8249;</button><button class="home-date-nav" data-month-shift="1">&#8250;</button></div>
  </div>
  <div class="home-date-weekdays">${['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => `<div class="home-date-weekday">${day}</div>`).join('')}</div>
  <div class="home-calendar-grid">${days}</div>`;
}

function bindCalendar(): void {
  const popover = document.getElementById('home-date-popover');
  if (!popover) return;
  popover.querySelectorAll<HTMLButtonElement>('[data-date]').forEach(button => {
    button.addEventListener('click', () => {
      selectedDate = button.dataset.date || selectedDate;
      followCentralToday = selectedDate === centralDateKey();
      calendarMonth = selectedDate.slice(0, 7);
      calendarOpen = false;
      render();
    });
  });
  popover.querySelectorAll<HTMLButtonElement>('[data-month-shift]').forEach(button => {
    button.addEventListener('click', event => {
      event.stopPropagation();
      const current = parseDateKey(`${calendarMonth}-01`) || new Date();
      current.setMonth(current.getMonth() + Number(button.dataset.monthShift || 0));
      calendarMonth = calendarDateKey(current).slice(0, 7);
      render();
    });
  });
}

function inlineDatePickerHtml(prefix: 'daily' | 'parlay' | 'profit', open: boolean, label: string): string {
  const toggle = prefix === 'daily' ? 'toggleDailyDatePicker' : prefix === 'profit' ? 'toggleProfitDatePicker' : 'toggleParlayDatePicker';
  const selectedAll = filteredPicks().filter(pick => pickDateKey(pick) === selectedDate);
  return `<div class="home-date-wrap inline-date-wrap" id="${prefix}-date-wrap">
    <button
      id="${prefix}-date-trigger"
      class="home-date-trigger"
      type="button"
      onclick="${toggle}(event)"
      aria-haspopup="dialog"
      aria-expanded="${open}"
    >
      <span class="home-date-trigger-copy">
        <span class="home-date-trigger-label">${escapeHtml(label)}</span>
        <span class="home-date-trigger-value">${escapeHtml(dateLabel(selectedDate, true))}</span>
      </span>
      <span class="home-date-trigger-meta">${selectedDate === centralDateKey() ? 'Today | CT' : `${selectedAll.length} picks`}</span>
      <span class="home-date-trigger-caret">&#9662;</span>
    </button>
    <div id="${prefix}-date-popover" class="home-date-popover ${open ? 'open' : ''}" role="dialog" aria-label="Select board date">${calendarHtml()}</div>
  </div>`;
}

function bindInlineDatePicker(prefix: 'daily' | 'parlay' | 'profit'): void {
  const popover = document.getElementById(`${prefix}-date-popover`);
  if (!popover) return;
  popover.querySelectorAll<HTMLButtonElement>('[data-date]').forEach(button => {
    button.addEventListener('click', () => {
      selectedDate = button.dataset.date || selectedDate;
      followCentralToday = selectedDate === centralDateKey();
      calendarMonth = selectedDate.slice(0, 7);
      calendarOpen = false;
      dailyCalendarOpen = false;
      profitCalendarOpen = false;
      parlayCalendarOpen = false;
      render();
    });
  });
  popover.querySelectorAll<HTMLButtonElement>('[data-month-shift]').forEach(button => {
    button.addEventListener('click', event => {
      event.stopPropagation();
      const current = parseDateKey(`${calendarMonth}-01`) || new Date();
      current.setMonth(current.getMonth() + Number(button.dataset.monthShift || 0));
      calendarMonth = calendarDateKey(current).slice(0, 7);
      render();
    });
  });
}

function renderHome(): void {
  ensureSelection();
  renderFilters();
  const picks = boardPicks();
  const stats = statsFor(picks);
  const selectedAll = filteredPicks().filter(pick => pickDateKey(pick) === selectedDate);
  const groups = new Map<string, Pick[]>();
  picks.forEach(pick => groups.set(gameKey(pick), [...(groups.get(gameKey(pick)) || []), pick]));
  const sortedGames = [...groups.entries()].sort((left, right) => compareGameStartAsc(left[1], right[1]));

  const title = document.getElementById('home-title');
  const eyebrow = document.getElementById('home-eyebrow');
  const sub = document.getElementById('home-sub');
  const triggerValue = document.getElementById('home-date-trigger-value');
  const triggerMeta = document.getElementById('home-date-trigger-meta');
  const itemLabel = activePickMode === 'player' ? 'props' : 'picks';
  const itemTitle = activePickMode === 'player' ? 'Props' : 'Picks';
  if (eyebrow) eyebrow.textContent = activePickMode === 'player' ? 'PLAYER PROPS' : 'TEAM PICKS';
  if (title) title.textContent = activePickMode === 'player'
    ? `${dateLabel(selectedDate, true)} Player Props`
    : `${dateLabel(selectedDate, true)} Picks`;
  if (sub) sub.textContent = `${selectedAll.length} ${activePickMode === 'player' ? 'player props' : 'picks'} from ${new Set(selectedAll.map(sourceName)).size} sources, organized by matchup.`;
  if (triggerValue) triggerValue.textContent = dateLabel(selectedDate, true);
  if (triggerMeta) triggerMeta.textContent = selectedDate === centralDateKey() ? 'Today | CT' : `${selectedAll.length} picks`;
  document.querySelectorAll<HTMLElement>('[data-home-mode]').forEach(button => button.classList.toggle('active', button.dataset.homeMode === homeMode));

  const summary = document.getElementById('home-summary-grid');
  if (summary) summary.innerHTML = [
    [stats.total, homeMode === 'pending' ? `Open ${itemTitle}` : itemTitle],
    [groups.size, 'Matchups'],
    [new Set(picks.map(sourceName)).size, 'Sources'],
    [stats.pending, 'Open'],
    [trackedUnits(stats), 'Net Units'],
  ].map(([value, label]) => `<div class="home-summary-card"><div class="home-summary-value">${escapeHtml(value)}</div><div class="home-summary-label">${label}</div></div>`).join('');

  const popover = document.getElementById('home-date-popover');
  if (popover) {
    popover.innerHTML = calendarHtml();
    popover.classList.toggle('open', calendarOpen);
  }
  document.getElementById('home-date-trigger')?.setAttribute('aria-expanded', String(calendarOpen));
  bindCalendar();

  const feed = document.getElementById('pick-feed');
  if (!feed) return;
  if (!picks.length) {
    const modeLabel = homeMode === 'pending' ? 'open' : homeMode === 'settled' ? 'finished' : 'available';
    feed.innerHTML = `<div class="pick-feed-empty"><div class="home-empty-kicker">${homeMode === 'pending' ? `OPEN ${itemLabel.toUpperCase()}` : homeMode === 'settled' ? 'RESULTS' : `ALL ${itemLabel.toUpperCase()}`} | ${escapeHtml(dateLabel(selectedDate).toUpperCase())}</div><div class="home-empty-title">No ${modeLabel} ${itemLabel} in this view</div><div class="home-empty-sub">Try another date, sport, source, or result view.</div></div>`;
    return;
  }
  const bySport = new Map<string, Array<[string, Pick[]]>>();
  sortedGames.forEach(entry => {
    const sport = entry[1][0]?.sport || 'OTHER';
    bySport.set(sport, [...(bySport.get(sport) || []), entry]);
  });
  feed.innerHTML = [...bySport.entries()]
    .sort((left, right) => compareGameStartAsc(left[1][0]?.[1] || [], right[1][0]?.[1] || []))
    .map(([sport, games]) => `
    <section class="home-feed-section">
      <div class="home-feed-section-head"><div><div class="home-feed-section-title">${escapeHtml(sport)}</div><div class="home-feed-section-meta">${games.reduce((sum, game) => sum + game[1].length, 0)} ${itemLabel} | ${games.length} matchups</div></div></div>
      <div class="home-feed-grid">${games.map(([, gamePicks]) => renderGameCard(gamePicks)).join('')}</div>
    </section>`).join('');
  bindPickCards(feed);
  void refreshHomeScores(selectedDate, picks);
}

function renderGameCard(picks: Pick[]): string {
  const sortedPicks = [...picks].sort(compareHomePickRows);
  const stats = statsFor(picks);
  const pending = stats.pending > 0;
  const start = picks.map(pick => pick.start_time).filter(Boolean).sort()[0];
  const scoreChip = homeScoreChipHtml(homeScores.get(gameKey(picks[0])), start, gameName(picks[0]));
  const itemLabel = activePickMode === 'player' ? 'props' : 'picks';
  return `<article class="home-game-card status-${statusClass(picks)}">
    <div class="home-game-top">
      <div class="home-game-kicker"><span class="home-sport-pill">${escapeHtml(picks[0]?.sport)}</span><span class="home-status-pill ${statusClass(picks)}">${pending ? 'OPEN' : `${stats.wins}-${stats.losses}${stats.pushes ? `-${stats.pushes}` : ''}`}</span></div>
      <div class="home-game-right-stack">${scoreChip}<div class="home-game-pl ${trackedUnitsClass(stats)}">${pending ? `${stats.pending} open` : trackedUnits(stats)}</div><div class="home-game-caption">${formatStart(start)}</div></div>
    </div>
    <div><div class="home-game-title">${escapeHtml(gameName(picks[0]))}</div><div class="home-game-meta">${escapeHtml(dateLabel(pickDateKey(picks[0])))} | ${picks.length} ${itemLabel} | ${new Set(picks.map(sourceName)).size} sources</div></div>
    <div class="home-game-picks">${sortedPicks.map(renderPickRow).join('')}</div>
  </article>`;
}

// Quarter-Kelly is a display-only stake suggestion on a 100u bankroll; the
// tracked flat records elsewhere stay on their own units.
function quarterKellyLabel(pick: Pick): string {
  if (pick.result !== 'pending') return '';
  const decision = String(pick.decision || '').trim().toUpperCase();
  if (decision !== 'BET' && decision !== 'LEAN') return '';
  const recommended = Number(pick.recommended_units ?? (pick.quarter_kelly != null ? Number(pick.quarter_kelly) * 100 : NaN));
  if (!Number.isFinite(recommended) || recommended <= 0) return '';
  return `¼K ${Number(recommended.toFixed(1))}u`;
}

function renderPickRow(pick: Pick): string {
  const decision = String(pick.decision || '').trim().toUpperCase();
  const isPlayer = activePickMode === 'player';
  const details = pickExpandableDetails(pick);
  const hasResearch = Boolean(details.reason || details.factors.length);
  const expanded = hasResearch && expandedResearchPickKeys.has(pick.id);
  const researchAttrs = hasResearch
    ? ` data-research-pick-card="${escapeHtml(pick.id)}" role="button" tabindex="0" aria-expanded="${expanded}"`
    : '';
  const pickTextAttrs = isPlayer
    ? ''
    : hasResearch
      ? ' data-home-pick-text'
      : ' data-home-pick-text role="button" tabindex="0" aria-expanded="false"';
  return `<div class="home-feed-row result-${pick.result}${isPlayer ? ' player-row' : ''}${hasResearch ? ' is-expandable' : ''}${expanded ? ' expanded' : ''}"${researchAttrs}>
    ${isPlayer ? '' : `<span class="home-feed-row-sport">${escapeHtml(pick.sport)}</span>`}
    <div class="home-feed-row-body"><div class="home-feed-row-source">${escapeHtml(sourceName(pick))}</div><div class="home-feed-row-pick"${pickTextAttrs}>${escapeHtml(pick.pick)}</div><div class="home-feed-row-meta">${escapeHtml([formatOdds(pick), pick.odds != null && pick.price_verified !== true ? 'price unverified' : '', decision === 'PASS' ? '' : `${pick.units}u`, quarterKellyLabel(pick), formatStart(pick.start_time), activePickMode === 'player' ? '' : pick.decision].filter(Boolean).join(' | '))}</div>${researchDetailsHtml(pick, expanded)}</div>
    <div class="home-feed-row-pl ${pick.pl > 0 ? 'positive' : pick.pl < 0 ? 'negative' : 'neutral'}">${isUnsupportedPendingPick(pick) || (pick.result !== 'pending' && pick.price_verified !== true) ? 'P/L untracked' : pick.result === 'pending' ? decision === 'PASS' ? 'Pass' : `${pick.units}u risk` : signedUnits(pick.pl)}</div>
    <div class="home-feed-row-control">${pickResultBadge(pick)}</div>
  </div>`;
}

function bindHomePickTextExpansion(container: HTMLElement): void {
  container.querySelectorAll<HTMLElement>('[data-home-pick-text]').forEach(pickText => {
    const toggle = (): void => {
      const row = pickText.closest<HTMLElement>('.home-feed-row');
      if (!row) return;
      const expanded = row.classList.toggle('pick-text-expanded');
      pickText.setAttribute('aria-expanded', String(expanded));
    };
    pickText.addEventListener('click', event => {
      event.stopPropagation();
      toggle();
    });
    pickText.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      event.stopPropagation();
      toggle();
    });
  });
}

function bindResearchDetailCards(container: HTMLElement): void {
  container.querySelectorAll<HTMLElement>('[data-research-pick-card]').forEach(card => {
    const toggle = (): void => {
      const pickId = card.dataset.researchPickCard || '';
      if (!pickId) return;
      const expanded = !expandedResearchPickKeys.has(pickId);
      if (expanded) expandedResearchPickKeys.add(pickId);
      else expandedResearchPickKeys.delete(pickId);
      card.classList.toggle('expanded', expanded);
      card.setAttribute('aria-expanded', String(expanded));
      const label = card.querySelector<HTMLElement>('[data-player-expand-label]');
      if (label) label.textContent = expanded ? 'Hide research details' : 'Show research details';
    };
    card.addEventListener('click', event => {
      if ((event.target as HTMLElement).closest('a, button, input, select, textarea, label')) return;
      toggle();
    });
    card.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      toggle();
    });
  });
}

function bindPickCards(container: HTMLElement): void {
  bindHomePickTextExpansion(container);
  bindResearchDetailCards(container);
}

function updateOverallStats(): void {
  const stats = statsFor(activePickMode === 'player' ? rankingComparablePicks(getAllPicks()) : getAllPicks());
  const values: Record<string, string | number> = {
    'stat-picks': stats.total,
    'stat-wins': stats.wins,
    'stat-losses': stats.losses,
    'stat-pushes': stats.pushes,
    'stat-pending': stats.pending,
    'stat-acc': stats.winRate == null ? '—' : `${(stats.winRate * 100).toFixed(1)}%`,
    'stat-units': trackedUnits(stats),
    'stat-roi': stats.roi == null ? '—' : `${(stats.roi * 100).toFixed(1)}%`,
  };
  Object.entries(values).forEach(([id, value]) => {
    const element = document.getElementById(id);
    if (element) element.textContent = String(value);
  });
}

function renderRankings(): void {
  const allPicks = getAllPicks();
  const comparablePicks = rankingComparablePicks(allPicks);
  const rankingPicks = comparablePicks.filter(isSettledPick);
  const rankingTitle = document.getElementById('source-rankings-title');
  const rankingSubtitle = document.getElementById('source-rankings-subtitle');
  const dowSubtitle = document.getElementById('dow-subtitle');
  if (rankingTitle) rankingTitle.textContent = activePickMode === 'player' ? 'Model Rankings' : 'Source Rankings';
  if (rankingSubtitle) {
    rankingSubtitle.textContent = activePickMode === 'player'
      ? 'See how each active player-prop model performed on the latest decided slate.'
      : 'See how every source has performed across the picks and results collected here. Select a source for today, yesterday, last 7 days, and all-time records.';
  }
  if (dowSubtitle) {
    dowSubtitle.textContent = activePickMode === 'player'
      ? 'Player-prop model win rates by weekday. Green cells have at least three decided picks and a 55%+ win rate.'
      : 'Source win rates by weekday. Green cells have at least three decided picks and a 55%+ win rate.';
  }
  const bySource = new Map<string, Pick[]>();
  (activePickMode === 'player' ? comparablePicks : rankingPicks).forEach(pick => addPickToRankingBuckets(bySource, pick));
  const ranked = [...bySource.entries()].map(([source, picks]) => ({
    source,
    picks,
    stats: statsFor(picks.filter(isSettledPick)),
  }))
    .filter(item => activePickMode === 'player' || item.stats.wins + item.stats.losses > 0)
    .sort((a, b) => activePickMode === 'player'
      ? (
        (b.stats.wins + b.stats.losses) - (a.stats.wins + a.stats.losses)
        || (b.stats.roi ?? -999) - (a.stats.roi ?? -999)
        || b.stats.net - a.stats.net
        || a.source.localeCompare(b.source)
      )
      : (b.stats.roi ?? -999) - (a.stats.roi ?? -999) || b.stats.net - a.stats.net);
  const leaderboard = document.getElementById('leaderboard');
  if (leaderboard) {
    leaderboard.innerHTML = ranked.length ? ranked.map((item, index) => {
      const expanded = expandedSourceKeys.has(item.source);
      const records = sourceRecordLines(picksForRankingBucket(comparablePicks, item.source), centralDateKey());
      return `<article class="source-card ${index < 3 ? `rank-${index + 1}` : ''} ${expanded ? 'expanded' : ''}" data-source-card="${escapeHtml(item.source)}" role="button" tabindex="0" aria-expanded="${expanded}">
        <div class="card-rank">${index + 1}</div><div class="card-name">${escapeHtml(item.source)}</div>
        <div class="score-bar-wrap"><div class="score-label"><span>ACCURACY</span><span class="score-val">${item.stats.winRate == null ? '—' : `${(item.stats.winRate * 100).toFixed(1)}%`} (${item.stats.wins}-${item.stats.losses})</span></div><div class="bar-bg"><div class="bar-fill bar-acc" style="width:${(item.stats.winRate || 0) * 100}%"></div></div></div>
        <div class="score-bar-wrap"><div class="score-label"><span>ROI</span><span class="score-val">${item.stats.priced ? `${item.stats.roi == null ? '—' : `${(item.stats.roi * 100).toFixed(1)}%`} (${signedUnits(item.stats.net)})` : '— (no priced picks)'}</span></div><div class="bar-bg"><div class="bar-fill bar-roi" style="width:${Math.max(0, Math.min(100, 50 + (item.stats.roi || 0) * 100))}%"></div></div></div>
        <div class="algo-score"><div class="algo-score-val">${item.stats.total}</div><div class="algo-score-info">DECIDED PICKS<br>${escapeHtml(rankingBucketScopeLabel(item.source))}</div></div>
        <div class="source-expand-control"><span data-source-expand-label>${expanded ? 'Hide period records' : 'View period records'}</span><span class="source-expand-icon" aria-hidden="true">&#9662;</span></div>
        <div class="source-deep-dive">
          <div class="trend-deep-title">PERIOD RECORDS</div>
          <div class="source-record-list">${records.map(record => `<div class="source-record-item"><div class="source-record-label">${record.label}</div><div class="source-record-value">${record.text}</div></div>`).join('')}</div>
        </div>
      </article>`;
    }).join('') : '<div class="empty-state">Source records will appear here as games finish and scores come in.</div>';
    bindSourceCards(leaderboard);
  }

  const bySport = new Map<string, Pick[]>();
  rankingPicks.forEach(pick => bySport.set(pick.sport, [...(bySport.get(pick.sport) || []), pick]));
  const sportBoard = document.getElementById('sport-board');
  if (sportBoard) sportBoard.innerHTML = [...bySport.entries()].map(([sport, picks]) => {
    const stats = statsFor(picks);
    const profitLine = stats.priced
      ? `ROI ${stats.roi == null ? '—' : `${(stats.roi * 100).toFixed(1)}%`} • ${stats.priced} priced pick${stats.priced === 1 ? '' : 's'}`
      : 'P/L untracked — no verified-price picks yet';
    return `<div class="sport-card"><div class="sport-name">${escapeHtml(sport)}</div><div class="sport-meta">${stats.wins}-${stats.losses}${stats.pushes ? `-${stats.pushes}` : ''} record<br>${stats.total} decided picks</div><div class="sport-units ${trackedUnitsClass(stats)}">${trackedUnits(stats)}</div><div class="sport-meta">${profitLine}</div></div>`;
  }).join('');

  renderDayOfWeekTable();
  renderProfitQualificationBoard();
}

const QUALIFICATION_GATE_LABELS: Record<string, string> = {
  sourceSamples: 'Settled priced rows',
  distinctPriorDates: 'Distinct dates',
  positiveFlatRoi: 'Positive flat ROI',
  stableChronologicalHalves: 'Stable halves',
  probabilityPositiveEv: 'Pr(profit)',
};

function qualificationGateValue(value: number | boolean | null | undefined, key: string): string {
  if (value == null) return '—';
  if (typeof value === 'boolean') return value ? 'stable' : 'unstable';
  if (key === 'positiveFlatRoi') return `${(Number(value) * 100).toFixed(1)}%`;
  if (key === 'probabilityPositiveEv') return `${(Number(value) * 100).toFixed(1)}%`;
  return String(value);
}

function qualificationGateTarget(value: number | boolean | null | undefined, key: string): string {
  if (value == null) return '';
  if (typeof value === 'boolean') return 'required';
  if (key === 'positiveFlatRoi') return '> 0%';
  if (key === 'probabilityPositiveEv') return `≥ ${(Number(value) * 100).toFixed(0)}%`;
  return `≥ ${value}`;
}

function renderProfitQualificationBoard(): void {
  const container = document.getElementById('profit-qualification-board');
  if (!container) return;
  const payload = getProfitDeskPayload();
  const cards = Array.isArray(payload?.sources) ? payload!.sources!.filter(card => card && typeof card === 'object') : [];
  if (!cards.length) {
    container.innerHTML = '<div class="empty-state">Qualification progress appears after the Profit Desk publishes its next artifact.</div>';
    return;
  }
  const visible = cards.filter(card => (card.samples || 0) > 0 || (card.candidatesToday || 0) > 0).slice(0, 14);
  container.innerHTML = `<div class="qual-grid">${visible.map(card => {
    const qualified = card.evidenceQualified === true;
    const gates = card.gates && typeof card.gates === 'object' ? Object.entries(card.gates) : [];
    const record = `${card.wins ?? 0}-${card.losses ?? 0}`;
    const roi = card.flatRoi == null ? '—' : `${(Number(card.flatRoi) * 100).toFixed(1)}%`;
    return `<article class="qual-card ${qualified ? 'is-qualified' : ''}">
      <div class="qual-head">
        <div><div class="qual-source">${escapeHtml(card.source || card.sourceKey || 'Unknown source')}</div><div class="qual-meta">${escapeHtml(String(card.sport || '').toUpperCase())} • ${escapeHtml(String(card.mode || ''))} • ${record} at real prices • ROI ${escapeHtml(roi)}</div></div>
        <div class="qual-score ${qualified ? 'positive' : ''}"><strong>${card.gatesPassed ?? 0}/${card.gatesTotal ?? 5}</strong><span>${qualified ? 'ON THE CARD' : 'GATES CLEARED'}</span></div>
      </div>
      <ul class="qual-gates">${gates.map(([key, gate]) => `<li class="${gate?.passed ? 'pass' : 'fail'}"><span class="qual-gate-mark" aria-hidden="true">${gate?.passed ? '✓' : '✗'}</span><span class="qual-gate-label">${escapeHtml(QUALIFICATION_GATE_LABELS[key] || key)}</span><span class="qual-gate-value">${escapeHtml(qualificationGateValue(gate?.actual, key))} <small>${escapeHtml(qualificationGateTarget(gate?.required, key))}</small></span></li>`).join('')}</ul>
      ${(card.liveToday || 0) > 0 ? `<div class="qual-live-note">${card.liveToday} live pick${card.liveToday === 1 ? '' : 's'} on today's card</div>` : ''}
    </article>`;
  }).join('')}</div>
  <p class="qual-footnote">Evidence counts only settled picks with real executable prices and pregame timestamps, dated strictly before each slate. Sources qualify and lapse automatically as these numbers move.</p>`;
}

function bindSourceCards(leaderboard: HTMLElement): void {
  leaderboard.querySelectorAll<HTMLElement>('[data-source-card]').forEach(card => {
    const toggle = (): void => {
      const source = card.dataset.sourceCard || '';
      if (!source) return;
      const expanded = !expandedSourceKeys.has(source);
      if (expanded) expandedSourceKeys.add(source);
      else expandedSourceKeys.delete(source);
      card.classList.toggle('expanded', expanded);
      card.setAttribute('aria-expanded', String(expanded));
      const label = card.querySelector<HTMLElement>('[data-source-expand-label]');
      if (label) label.textContent = expanded ? 'Hide period records' : 'View period records';
    };
    card.addEventListener('click', toggle);
    card.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      toggle();
    });
  });
}

function renderDayOfWeekTable(): void {
  const container = document.getElementById('dow-model-breakdown');
  if (!container) return;
  const dayOrder = [1, 2, 3, 4, 5, 6, 0];
  const dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const bySource = new Map<string, Pick[][]>();

  rankingComparablePicks(getAllPicks()).filter(pick => pick.result === 'win' || pick.result === 'loss').forEach(pick => {
    const day = parseDateKey(pickDateKey(pick))?.getDay();
    if (day == null) return;
    rankingBucketNames(pick).forEach(source => {
      const buckets = bySource.get(source) || Array.from({ length: 7 }, () => []);
      buckets[day].push(pick);
      bySource.set(source, buckets);
    });
  });

  const sources = [...bySource.keys()].sort((a, b) => a.localeCompare(b));
  if (!sources.length) {
    container.innerHTML = '<div class="empty-state">No decided picks yet</div>';
    return;
  }

  container.innerHTML = `<table class="dow-table">
    <thead><tr><th>${activePickMode === 'player' ? 'Model' : 'Source'}</th>${dayLabels.map(day => `<th>${day}</th>`).join('')}</tr></thead>
    <tbody>${sources.map(source => `<tr><td class="dow-model-name">${escapeHtml(source)}</td>${dayOrder.map(day => {
      const stats = statsFor(bySource.get(source)?.[day] || []);
      const decided = stats.wins + stats.losses;
      const rate = stats.winRate == null ? null : stats.winRate * 100;
      const tone = decided < 3 || rate == null
        ? 'dow-cell-gray'
        : rate >= 55 ? 'dow-cell-green' : rate >= 50 ? 'dow-cell-yellow' : 'dow-cell-red';
      const text = rate == null ? '—' : `${rate.toFixed(0)}% (${stats.wins}-${stats.losses})`;
      return `<td class="${tone}" title="${decided} decided picks">${text}</td>`;
    }).join('')}</tr>`).join('')}</tbody>
  </table>`;
}

function renderSearch(): void {
  const input = document.getElementById('search-input') as HTMLInputElement | null;
  const results = document.getElementById('search-results');
  const meta = document.getElementById('search-meta');
  if (!input || !results || !meta) return;
  ensureSelection();
  const query = input.value.trim().toLowerCase();
  const pending = getAllPicks().filter(pick => isOpenPick(pick) && pickDateKey(pick) === selectedDate);
  const itemLabel = activePickMode === 'player' ? 'props' : 'picks';
  const subjects = activePickMode === 'player' ? 'player, prop, matchup, or source' : 'team, matchup, or source';
  const scope = `${dateLabel(selectedDate, true)} open ${itemLabel} (Central time)`;
  if (!query) {
    meta.textContent = `${pending.length} ${scope.toLowerCase()}`;
    results.innerHTML = `<div class="empty-state">Search for a ${subjects} in the selected date’s open ${itemLabel}</div>`;
    return;
  }
  const picks = pending.filter(pick => [
    pick.pick,
    pick.player,
    pick.team,
    pick.market,
    pick.reason,
    pick.key_factors,
    sourceName(pick),
    pick.sport,
    pick.date,
    gameName(pick),
  ].some(value => detailValues(value).some(detail => detail.toLowerCase().includes(query))))
    .sort(comparePickActionableStart);
  meta.textContent = `${picks.length} open ${picks.length === 1 ? itemLabel.slice(0, -1) : itemLabel} for "${input.value.trim()}" | ${scope}`;
  results.innerHTML = picks.length ? picks.map(pick => {
    const details = pickExpandableDetails(pick);
    const hasResearch = Boolean(details.reason || details.factors.length);
    const expanded = hasResearch && expandedResearchPickKeys.has(pick.id);
    return `<article class="search-card ${hasResearch ? 'is-expandable' : ''} ${expanded ? 'expanded' : ''}" ${hasResearch ? `data-research-pick-card="${escapeHtml(pick.id)}" role="button" tabindex="0" aria-expanded="${expanded}"` : ''}>
      <div class="search-card-top">${pickResultBadge(pick)}<span class="badge badge-source">${escapeHtml(sourceName(pick))}</span><div class="search-card-pick">${escapeHtml(pick.pick)}</div><div class="search-card-odds">${escapeHtml(formatOdds(pick))}</div></div>
      <div class="search-card-row"><div class="search-card-field"><span class="search-card-field-label">GAME</span><span class="search-card-field-val">${escapeHtml(gameName(pick))}</span></div><div class="search-card-field"><span class="search-card-field-label">DATE</span><span class="search-card-field-val">${escapeHtml(pick.date)}</span></div><div class="search-card-field"><span class="search-card-field-label">P/L</span><span class="search-card-field-val">${signedUnits(pick.pl)}</span></div></div>
      ${researchDetailsHtml(pick, expanded)}
    </article>`;
  }).join('') : `<div class="empty-state">No open ${itemLabel} match that search for the selected date</div>`;
  bindPickCards(results);
}

function canonicalTeamForPick(pick: Pick, label: string): string {
  const target = normalizeTeam(label);
  const matched = teamsForPick(pick)?.find(team => {
    const normalized = normalizeTeam(team);
    return normalized === target || normalized.includes(target) || target.includes(normalized);
  });
  return canonicalTeamToken(matched || label);
}

function trendMarketScope(pick: Pick, selection: string): string {
  const lower = selection.toLowerCase();
  const inning = lower.match(/\binning\s*(\d+)|\b(\d+)(?:st|nd|rd|th)?\s+inning\b/);
  if (inning) return `inning:${inning[1] || inning[2]}`;
  if (/\bf5\b|first five/.test(lower)) return 'first-five';
  if (lower.includes('team total')) {
    return `team-total:${canonicalTeamForPick(pick, String(pick.team || selection.split(/team total/i)[0]))}`;
  }
  return 'full-game';
}

function canonicalTrendLine(value: string): string {
  const normalized = value.replace(',', '.');
  const numeric = Number(normalized);
  return Number.isFinite(numeric) ? String(numeric) : normalized;
}

function canonicalTrendSignal(pick: Pick): { key: string; label: string; pass: boolean } {
  const selection = pick.pick
    .replace(/\s+\([^()]*(?:@|vs\.?)\s+[^()]*\)\s*$/i, '')
    .trim();
  const pass = String(pick.decision || '').trim().toUpperCase() === 'PASS';
  const scope = trendMarketScope(pick, selection);
  const total = selection.match(/^(over|under)\s+(\d+(?:\.\d+)?)(?:\s+(?:points?|runs?|goals?))?$/i);
  if (total) return { key: `${scope}:total:${total[1].toLowerCase()}:${total[2]}`, label: selection, pass };

  const noRun = selection.match(/\binning\s*(\d+).*?\bno runs?\b|\bno runs?\b.*?\binning\s*(\d+)/i);
  if (noRun) return { key: `inning:no-run:${noRun[1] || noRun[2]}`, label: selection, pass };

  const namedHandicap = selection.match(
    /^(.*?)\s+(?:(asian)\s+)?(?:hcp|handicap)\s*\(\s*([+-]?\d+(?:[.,]\d+)?)\s*\)$/i,
  );
  if (namedHandicap) {
    const soccerHandicap = namedHandicap[2]
      || String(pick.sport || '').trim().toUpperCase() === 'FIFA WC'
      || String(pick.market_type || '').trim() === 'soccer_asian_handicap';
    const family = soccerHandicap ? 'asian-handicap' : 'spread';
    return {
      key: `${scope}:${family}:${canonicalTeamForPick(pick, namedHandicap[1])}:${canonicalTrendLine(namedHandicap[3])}`,
      label: selection,
      pass,
    };
  }

  const asian = selection.match(/^(.*?)\s+asian\s+(?:hcp|handicap)\s*([+-]?\d+(?:[.,]\d+)?)$/i);
  if (asian) return { key: `${scope}:asian-handicap:${canonicalTeamForPick(pick, asian[1])}:${canonicalTrendLine(asian[2])}`, label: selection, pass };

  const spread = selection.match(/^(.*?)\s+([+-]\d+(?:[.,]\d+)?)$/);
  if (spread) return { key: `${scope}:spread:${canonicalTeamForPick(pick, spread[1])}:${canonicalTrendLine(spread[2])}`, label: selection, pass };

  const moneyline = selection.match(/^(.*?)\s+(?:ML|moneyline|to win|wins?)$/i);
  if (moneyline) return { key: `${scope}:side:${canonicalTeamForPick(pick, moneyline[1])}`, label: selection, pass };

  return { key: `pick:${normalizeTeam(selection)}`, label: selection, pass };
}

function trendSignalGroups(picks: Pick[]): TrendSignalGroup[] {
  const grouped = new Map<string, { labels: Set<string>; picks: Pick[]; pass: boolean }>();
  picks.forEach(pick => {
    const signal = canonicalTrendSignal(pick);
    const key = `${signal.pass ? 'pass' : 'bet'}:${signal.key}`;
    const current = grouped.get(key) || { labels: new Set<string>(), picks: [], pass: signal.pass };
    current.labels.add(signal.label);
    current.picks.push(pick);
    grouped.set(key, current);
  });
  return [...grouped.entries()].map(([key, group]) => ({
    key,
    label: [...group.labels].join(' / '),
    picks: group.picks,
    matching: !group.pass && new Set(group.picks.map(sourceName)).size >= 2,
    pass: group.pass,
  })).sort((a, b) => Number(b.matching) - Number(a.matching) || b.picks.length - a.picks.length);
}

function pickProbability(pick: Pick): number | null {
  if (pick.probability == null) return null;
  const raw = Number(pick.probability);
  if (!Number.isFinite(raw)) return null;
  const probability = raw > 1 ? raw / 100 : raw;
  return probability >= 0 && probability <= 1 ? probability : null;
}

function pickEdgePercent(pick: Pick): number | null {
  if (pick.market_edge == null && pick.edge == null) return null;
  const raw = Number(pick.market_edge ?? pick.edge);
  if (!Number.isFinite(raw)) return null;
  return Math.abs(raw) <= 1 ? raw * 100 : raw;
}

function dailyDecision(pick: Pick): string {
  return String(pick.decision || 'WATCH').trim().toUpperCase();
}

function isPublishedDailyPick(pick: Pick): boolean {
  const decision = dailyDecision(pick);
  return decision === 'BET' || decision === 'LEAN';
}

function dailyPickKey(pick: Pick): string {
  const selection = pick.pick.split('(', 1)[0].trim().toLowerCase()
    .replace(/\bfirst five\b/g, 'f5')
    .replace(/\b(?:moneyline|to win|wins?)\b/g, 'ml')
    .replace(/[^a-z0-9+.-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return `${gameKey(pick)}::${selection}`;
}

function uniqueDailyPicks(picks: Pick[]): Pick[] {
  const seen = new Set<string>();
  return picks.filter(pick => {
    const key = dailyPickKey(pick);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function dailySourceForms(date: string, todaysPicks: Pick[]): DailySourceForm[] {
  const comparable = rankingComparablePicks(getAllPicks());
  const historical = comparable.filter(pick => pickDateKey(pick) < date);
  const recentDates = [...new Set(historical.map(pickDateKey).filter(Boolean))].sort().slice(-3);
  const lastDate = recentDates.at(-1) || '';
  const sources = new Set(todaysPicks.map(sourceName));
  return [...sources].map(source => {
    const recent = historical.filter(pick => sourceName(pick) === source && recentDates.includes(pickDateKey(pick)) && pick.result !== 'pending');
    const last = recent.filter(pick => pickDateKey(pick) === lastDate);
    const recentStats = statsFor(recent);
    const lastStats = statsFor(last);
    const todayCalls = uniqueDailyPicks(todaysPicks
      .filter(pick => sourceName(pick) === source && isOpenPick(pick) && isPublishedDailyPick(pick))
      .sort(comparePickActionableStart));
    const score = (recentStats.winRate || 0) * 100 + Math.min(recentStats.wins + recentStats.losses, 20) * 0.35 + recentStats.net * 0.08;
    return { source, recentStats, lastStats, recentDates, todayCalls, score };
  }).filter(form => form.recentStats.wins + form.recentStats.losses >= 3)
    .sort((a, b) => b.score - a.score);
}

function weekdayLabel(date: string): string {
  const parsed = parseDateKey(date);
  return parsed ? parsed.toLocaleDateString('en-US', { weekday: 'long' }) : '';
}

function weekdaySourceForms(date: string, todaysPicks: Pick[]): WeekdaySourceForm[] {
  const day = parseDateKey(date)?.getDay();
  if (day == null) return [];
  const historical = rankingComparablePicks(getAllPicks()).filter(pick => (
    pickDateKey(pick) < date &&
    (pick.result === 'win' || pick.result === 'loss' || pick.result === 'push') &&
    parseDateKey(pickDateKey(pick))?.getDay() === day
  ));
  const sources = new Set(todaysPicks.map(sourceName));
  return [...sources].map(source => {
    const stats = statsFor(historical.filter(pick => sourceName(pick) === source));
    const decided = stats.wins + stats.losses;
    const todayCalls = uniqueDailyPicks(todaysPicks
      .filter(pick => sourceName(pick) === source && isOpenPick(pick) && isPublishedDailyPick(pick))
      .sort(comparePickActionableStart));
    const score = (stats.winRate || 0) * 100 + Math.min(decided, 20) * 0.35 + stats.net * 0.08;
    return { source, stats, decided, todayCalls, score };
  }).sort((a, b) => b.score - a.score);
}

function compareWeekdaySourceForms(left: WeekdaySourceForm, right: WeekdaySourceForm): number {
  if (dailySort === 'time' && left.todayCalls.length && right.todayCalls.length) {
    return comparePickActionableStart(left.todayCalls[0], right.todayCalls[0]) || right.score - left.score;
  }
  return right.score - left.score;
}

function dailyPickScore(pick: Pick, forms: Map<string, DailySourceForm>): number {
  const modelRank = activePickMode === 'player' ? playerModelRank(pick) : null;
  if (modelRank != null) return 10000 - modelRank;
  const probability = pickProbability(pick);
  const edge = pickEdgePercent(pick);
  const sourceRate = forms.get(sourceName(pick))?.recentStats.winRate;
  return (probability == null ? 45 : probability * 100)
    + Math.max(-10, Math.min(25, edge || 0)) * 0.65
    + (sourceRate == null ? 0 : (sourceRate - 0.5) * 30)
    + (dailyDecision(pick) === 'BET' ? 8 : dailyDecision(pick) === 'LEAN' ? 2 : 0);
}

function dailyGroupPercentage(group: DailyPickGroup): number {
  const probability = pickProbability(group.primary);
  if (probability != null) return probability;
  return group.score / 100;
}

function compareDailyGroupPercentage(left: DailyPickGroup, right: DailyPickGroup): number {
  return dailyGroupPercentage(right) - dailyGroupPercentage(left)
    || comparePickActionableStart(left.primary, right.primary)
    || right.picks.length - left.picks.length;
}

function compareDailyGroupTime(left: DailyPickGroup, right: DailyPickGroup): number {
  return comparePickActionableStart(left.primary, right.primary)
    || compareDailyGroupPercentage(left, right)
    || right.picks.length - left.picks.length;
}

function sortDailyGroups(groups: DailyPickGroup[]): DailyPickGroup[] {
  return [...groups].sort(dailySort === 'percentage' ? compareDailyGroupPercentage : compareDailyGroupTime);
}

function compareDailySourceForms(left: DailySourceForm, right: DailySourceForm): number {
  if (dailySort === 'time') {
    return comparePickActionableStart(left.todayCalls[0], right.todayCalls[0]) || right.score - left.score;
  }
  return right.score - left.score || comparePickActionableStart(left.todayCalls[0], right.todayCalls[0]);
}

function compareDailyConsensusSignal(
  left: { signal: TrendSignalGroup; game: Pick },
  right: { signal: TrendSignalGroup; game: Pick },
): number {
  if (dailySort === 'time') {
    return comparePickActionableStart(left.game, right.game) || right.signal.picks.length - left.signal.picks.length;
  }
  return (pickProbability(right.game) || 0) - (pickProbability(left.game) || 0)
    || right.signal.picks.length - left.signal.picks.length
    || comparePickActionableStart(left.game, right.game);
}

function dailyPickGroups(
  picks: Pick[],
  tagById: Map<string, Set<string>>,
  forms: Map<string, DailySourceForm>,
  allPicks: Pick[] = picks,
): DailyPickGroup[] {
  const includedKeys = new Set(picks.map(dailyPickKey));
  const groups = new Map<string, Pick[]>();
  allPicks.filter(pick => includedKeys.has(dailyPickKey(pick)))
    .forEach(pick => groups.set(dailyPickKey(pick), [...(groups.get(dailyPickKey(pick)) || []), pick]));
  return [...groups.entries()].map(([key, groupedPicks]) => {
    const bySource = new Map<string, Pick>();
    groupedPicks.forEach(pick => {
      const current = bySource.get(sourceName(pick));
      if (!current || dailyPickScore(pick, forms) > dailyPickScore(current, forms)) bySource.set(sourceName(pick), pick);
    });
    const ranked = [...bySource.values()].sort((a, b) => dailyPickScore(b, forms) - dailyPickScore(a, forms));
    const tags = [...new Set(groupedPicks.flatMap(pick => [...(tagById.get(pick.id) || [])]))];
    return { key, picks: ranked, primary: ranked[0], tags, score: dailyPickScore(ranked[0], forms) };
  }).sort((a, b) => comparePickActionableStart(a.primary, b.primary) || b.score - a.score || b.picks.length - a.picks.length);
}

function dailyPickGroupCard(group: DailyPickGroup): string {
  const pick = group.primary;
  const probability = pickProbability(pick);
  const edge = pickEdgePercent(pick);
  const decision = dailyDecision(pick);
  const pricey = group.picks.some(item => item.odds != null && item.odds <= -300);
  const metric = probability != null
    ? `${(probability * 100).toFixed(1)}%`
    : edge != null ? `${edge >= 0 ? '+' : ''}${edge.toFixed(1)}%` : formatOdds(pick) || 'TRACK';
  const metricLabel = probability != null ? 'MODEL WIN PROB' : edge != null ? 'MODEL EDGE' : 'MARKET PRICE';
  const sources = [...new Set(group.picks.map(sourceName))];
  const details = pickExpandableDetails(pick);
  const hasResearch = Boolean(details.reason || details.factors.length);
  const expanded = hasResearch && expandedResearchPickKeys.has(pick.id);
  return `<article class="daily-bet-card decision-${decision.toLowerCase()} ${pricey ? 'is-pricey' : ''} ${hasResearch ? 'is-expandable' : ''} ${expanded ? 'expanded' : ''}" ${hasResearch ? `data-research-pick-card="${escapeHtml(pick.id)}" role="button" tabindex="0" aria-expanded="${expanded}"` : ''}>
    <div class="daily-bet-top"><div><div class="daily-bet-source">${sources.length} ${sources.length === 1 ? 'SOURCE' : 'SOURCES'} | ${escapeHtml(pick.sport)}</div><div class="daily-bet-pick">${escapeHtml(pick.pick)}</div></div><div class="daily-bet-score"><strong>${escapeHtml(metric)}</strong><span>${metricLabel}</span></div></div>
    <div class="daily-bet-game">${escapeHtml(gameName(pick))} | ${escapeHtml(formatStart(pick.start_time))}</div>
    <div class="daily-bet-tags">${group.tags.map(tag => `<span class="${tag === 'PRICEY FAVORITE' ? 'pricey' : 'daily-qualifier-tag'}">${escapeHtml(tag)}</span>`).join('')}${pricey && !group.tags.includes('PRICEY FAVORITE') ? '<span class="pricey">PRICEY FAVORITE</span>' : ''}</div>
    <div class="daily-pick-source-list">${group.picks.map(sourcePick => {
      const sourceProbability = pickProbability(sourcePick);
      const sourceEdge = pickEdgePercent(sourcePick);
      const sourceMeta = [
        dailyDecision(sourcePick),
        formatOdds(sourcePick),
        sourceProbability == null ? '' : `${(sourceProbability * 100).toFixed(1)}%`,
        sourceEdge == null ? '' : `${sourceEdge >= 0 ? '+' : ''}${sourceEdge.toFixed(1)}% edge`,
      ].filter(Boolean).join(' | ');
      return `<div class="daily-pick-source-row"><strong>${escapeHtml(sourceName(sourcePick))}</strong><span>${escapeHtml(sourceMeta)}</span></div>`;
    }).join('')}</div>
    ${researchDetailsHtml(pick, expanded)}
  </article>`;
}

function dailySection(title: string, subtitle: string, body: string, meta = ''): string {
  return `<section class="daily-zone"><div class="daily-section-head"><div><div class="daily-section-title">${escapeHtml(title)}</div><div class="daily-section-sub">${escapeHtml(subtitle)}</div></div>${meta ? `<div class="daily-section-meta">${escapeHtml(meta)}</div>` : ''}</div>${body}</section>`;
}

function dailyPickGrid(groups: DailyPickGroup[]): string {
  if (!groups.length) return '<div class="daily-empty"><div class="daily-empty-title">Nothing qualifies yet</div><div class="daily-empty-sub">This view fills in when today’s picks meet its rules.</div></div>';
  return `<div class="daily-bet-grid">${groups.map(dailyPickGroupCard).join('')}</div>`;
}

function dailyHotModelCard(form: DailySourceForm): string {
  const recentDecided = form.recentStats.wins + form.recentStats.losses;
  const lastDecided = form.lastStats.wins + form.lastStats.losses;
  const todays = form.todayCalls.slice(0, 3);
  return `<article class="daily-model-card">
    <div class="daily-model-head"><div><div class="daily-model-kicker">HOT SOURCE</div><div class="daily-model-name">${escapeHtml(form.source)}</div></div><div class="daily-model-rate">${form.recentStats.winRate == null ? '—' : `${(form.recentStats.winRate * 100).toFixed(0)}%`}</div></div>
    <div class="daily-model-records"><span>Last ${form.recentDates.length} slates: ${form.recentStats.wins}-${form.recentStats.losses}${form.recentStats.pushes ? `-${form.recentStats.pushes}` : ''}</span><span>Last slate: ${lastDecided ? `${form.lastStats.wins}-${form.lastStats.losses}${form.lastStats.pushes ? `-${form.lastStats.pushes}` : ''}` : 'No decisions'}</span></div>
    <div class="daily-model-picks">${todays.length ? todays.map(pick => `<div><strong>${escapeHtml(pick.pick)}</strong><span>${escapeHtml([dailyDecision(pick), formatOdds(pick), pickProbability(pick) == null ? '' : `${(pickProbability(pick)! * 100).toFixed(1)}%`].filter(Boolean).join(' | '))}</span></div>`).join('') : '<div><strong>No published call today</strong><span>Recent form is hot, but the model is sitting out.</span></div>'}</div>
    <div class="daily-model-foot">${recentDecided} recent decisions${form.recentStats.priced ? ` | ${signedUnits(form.recentStats.net)}` : ''}</div>
  </article>`;
}

function weekdayRecordText(form: WeekdaySourceForm): string {
  return `${form.stats.wins}-${form.stats.losses}${form.stats.pushes ? `-${form.stats.pushes}` : ''}`;
}

function dailyDayFormCard(form: WeekdaySourceForm, dayName: string): string {
  const rate = form.stats.winRate == null ? null : form.stats.winRate * 100;
  const tone = rate == null ? 'unproven' : rate >= 55 ? 'hot' : rate >= 50 ? 'steady' : 'cold';
  const dayUpper = dayName.toUpperCase();
  const kicker = rate == null
    ? `NO ${dayUpper} DECISIONS YET`
    : tone === 'hot' ? `HOT ON ${dayUpper}S` : tone === 'steady' ? `STEADY ON ${dayUpper}S` : `COLD ON ${dayUpper}S`;
  const todays = form.todayCalls.slice(0, 3);
  const foot = tone === 'cold' && form.todayCalls.length
    ? `Careful: this source loses more than it wins on ${dayName}s.`
    : `${form.decided} decided ${dayName} pick${form.decided === 1 ? '' : 's'} before this slate`;
  return `<article class="daily-model-card daily-dayform-card is-${tone}">
    <div class="daily-model-head"><div><div class="daily-model-kicker">${escapeHtml(kicker)}</div><div class="daily-model-name">${escapeHtml(form.source)}</div></div><div class="daily-model-rate">${rate == null ? '—' : `${rate.toFixed(0)}%`}</div></div>
    <div class="daily-model-records"><span>${escapeHtml(dayName)}s: ${escapeHtml(weekdayRecordText(form))}</span><span>${form.stats.priced ? `${escapeHtml(dayName)} net: ${escapeHtml(signedUnits(form.stats.net))}` : 'No priced results'}</span></div>
    <div class="daily-model-picks">${todays.length ? todays.map(pick => `<div><strong>${escapeHtml(pick.pick)}</strong><span>${escapeHtml([dailyDecision(pick), formatOdds(pick), pickProbability(pick) == null ? '' : `${(pickProbability(pick)! * 100).toFixed(1)}%`].filter(Boolean).join(' | '))}</span></div>`).join('') : `<div><strong>No published call today</strong><span>${tone === 'cold' ? 'Sitting out may be for the best on this day.' : 'This source is sitting out this slate.'}</span></div>`}</div>
    <div class="daily-model-foot">${escapeHtml(foot)}</div>
  </article>`;
}

function dailyDayFormBody(date: string, dayForms: WeekdaySourceForm[], formsBySource: Map<string, DailySourceForm>): string {
  const dayName = weekdayLabel(date);
  const qualified = dayForms.filter(form => form.decided >= 3);
  if (!qualified.length) {
    return `<div class="daily-empty"><div class="daily-empty-title">Not enough ${escapeHtml(dayName)} history yet</div><div class="daily-empty-sub">A source needs at least three decided picks on past ${escapeHtml(dayName)}s before it can be ranked here.</div></div>`;
  }
  const actionable = qualified.filter(form => form.todayCalls.length).sort(compareWeekdaySourceForms);
  const benched = qualified.filter(form => !form.todayCalls.length);
  const unproven = dayForms.filter(form => form.decided < 3 && form.todayCalls.length);
  const spotlightForm = [...actionable].sort((a, b) => b.score - a.score).find(form => (form.stats.winRate || 0) >= 0.5) || null;
  let spotlightHtml: string;
  if (spotlightForm) {
    const pick = [...spotlightForm.todayCalls].sort((a, b) => dailyPickScore(b, formsBySource) - dailyPickScore(a, formsBySource))[0];
    const rate = (spotlightForm.stats.winRate || 0) * 100;
    const meta = [dailyDecision(pick), formatOdds(pick), pickProbability(pick) == null ? '' : `${(pickProbability(pick)! * 100).toFixed(1)}%`].filter(Boolean).join(' | ');
    spotlightHtml = `<article class="daily-dayform-spotlight">
      <div class="daily-dayform-spotlight-kicker">BEST BET FOR ${escapeHtml(dayName.toUpperCase())}</div>
      <div class="daily-dayform-spotlight-pick">${escapeHtml(pick.pick)}</div>
      <div class="daily-dayform-spotlight-game">${escapeHtml(gameName(pick))} | ${escapeHtml(formatStart(pick.start_time))}</div>
      <div class="daily-dayform-spotlight-meta"><span>${escapeHtml(spotlightForm.source)} is ${escapeHtml(weekdayRecordText(spotlightForm))} (${rate.toFixed(0)}%) on ${escapeHtml(dayName)}s</span><span>${escapeHtml(meta)}</span></div>
    </article>`;
  } else if (actionable.length) {
    spotlightHtml = `<div class="daily-dayform-warning"><strong>${escapeHtml(dayName)}s have been rough.</strong> No source with a winning ${escapeHtml(dayName)} record has a published call today, so there is no ${escapeHtml(dayName)} best bet. Treat this slate with extra caution.</div>`;
  } else {
    spotlightHtml = `<div class="daily-dayform-warning"><strong>No ranked source has a call today.</strong> Every source with enough ${escapeHtml(dayName)} history is sitting this slate out.</div>`;
  }
  const unprovenHtml = unproven.length
    ? `<div class="daily-dayform-unproven">Not enough ${escapeHtml(dayName)} history to rank: ${unproven.map(form => `${escapeHtml(form.source)} (${form.decided} decided)`).join(', ')}. Their calls still appear in Top Picks.</div>`
    : '';
  return `${spotlightHtml}<div class="daily-model-grid">${[...actionable, ...benched].map(form => dailyDayFormCard(form, dayName)).join('')}</div>${unprovenHtml}`;
}

function dailyConsensusCards(picks: Pick[]): string {
  const games = new Map<string, Pick[]>();
  picks.forEach(pick => games.set(gameKey(pick), [...(games.get(gameKey(pick)) || []), pick]));
  const matching = [...games.values()].flatMap(gamePicks => trendSignalGroups(gamePicks)
    .filter(signal => signal.matching)
    .map(signal => ({ signal, game: gamePicks[0] })))
    .sort(compareDailyConsensusSignal);
  if (!matching.length) return '<div class="daily-empty"><div class="daily-empty-title">No true consensus yet</div><div class="daily-empty-sub">Two independent sources must make the same market selection.</div></div>';
  return `<div class="daily-consensus-grid">${matching.map(({ signal, game }) => {
    const details = pickExpandableDetails(game);
    const hasResearch = Boolean(details.reason || details.factors.length);
    const expanded = hasResearch && expandedResearchPickKeys.has(game.id);
    return `<article class="daily-consensus-card ${hasResearch ? 'is-expandable' : ''} ${expanded ? 'expanded' : ''}" ${hasResearch ? `data-research-pick-card="${escapeHtml(game.id)}" role="button" tabindex="0" aria-expanded="${expanded}"` : ''}><div class="daily-consensus-count">${new Set(signal.picks.map(sourceName)).size} SOURCES</div><div class="daily-consensus-pick">${escapeHtml(signal.label)}</div><div class="daily-consensus-game">${escapeHtml(gameName(game))}</div><div class="trend-source-row">${[...new Set(signal.picks.map(sourceName))].map(source => `<span class="trend-source-pill">${escapeHtml(source)}</span>`).join('')}</div>${researchDetailsHtml(game, expanded)}</article>`;
  }).join('')}</div>`;
}

function formatAmericanOddsValue(odds: number | null | undefined): string {
  if (odds == null || !Number.isFinite(Number(odds))) return '--';
  return odds > 0 ? `+${Math.round(odds)}` : String(Math.round(odds));
}

function formatProbabilityValue(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(Number(value))) return '--';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatEvValue(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(Number(value))) return '--';
  const percent = Number(value) * 100;
  return `${percent >= 0 ? '+' : ''}${percent.toFixed(1)}%`;
}

function parlayRecordText(record: { wins?: number; losses?: number; pushes?: number } | null | undefined): string {
  if (!record || typeof record !== 'object') return '0-0';
  const wins = Number((record as { wins?: number }).wins || 0);
  const losses = Number((record as { losses?: number }).losses || 0);
  const pushes = Number((record as { pushes?: number }).pushes || 0);
  return `${wins}-${losses}${pushes ? `-${pushes}` : ''}`;
}

function parlayResultBadge(result: PickResult): string {
  return `<span class="daily-slate-result ${escapeHtml(result)}">${result === 'pending' ? 'OPEN' : result.toUpperCase()}</span>`;
}

function parlayCardPickMode(card: ParlayCard): PickMode | 'mixed' {
  if (card.pickMode === 'team' || card.pickMode === 'player') return card.pickMode;
  const hasPlayerLegs = card.legs.some(leg => leg.sourceType === 'player_prop');
  const hasTeamLegs = card.legs.some(leg => leg.sourceType !== 'player_prop');
  if (hasPlayerLegs && hasTeamLegs) return 'mixed';
  return hasPlayerLegs ? 'player' : 'team';
}

function parlayCardsForMode(payload: ParlayCardsPayload | null): ParlayCard[] {
  return (payload?.cards || []).filter(card => parlayCardPickMode(card) === activePickMode);
}

function parlayCardDedupeKey(card: ParlayCard): string {
  return [
    card.date || '',
    card.category || '',
    card.id || '',
    String(card.comboKey || ''),
  ].join('|');
}

function dedupeParlayCards(cards: ParlayCard[]): ParlayCard[] {
  const byKey = new Map<string, ParlayCard>();
  cards.forEach(card => byKey.set(parlayCardDedupeKey(card), card));
  return [...byKey.values()];
}

function parlayRankingCardsForDate(date: string, fallbackCards: ParlayCard[], engineVersion?: string): ParlayCard[] {
  const cutoff = date || centralDateKey();
  const historical = getParlayCardPayloads()
    .filter(payload => String(payload.date || '') <= cutoff)
    .filter(payload => !engineVersion || payload.engineVersion === engineVersion)
    .flatMap(payload => payload.cards || [])
    .filter(card => parlayCardPickMode(card) === activePickMode);
  return dedupeParlayCards(historical.length ? historical : fallbackCards);
}

function parlayResultMatches(card: ParlayCard): boolean {
  const result = card.result || 'pending';
  if (parlayResultMode === 'all') return true;
  if (parlayResultMode === 'pending') return result === 'pending';
  return result !== 'pending';
}

function parlayRecordForCards(cards: ParlayCard[]): { wins: number; losses: number; pushes: number; pending: number } {
  const uniqueCards = dedupeParlayCards(cards);
  return {
    wins: uniqueCards.filter(card => card.result === 'win').length,
    losses: uniqueCards.filter(card => card.result === 'loss').length,
    pushes: uniqueCards.filter(card => card.result === 'push').length,
    pending: uniqueCards.filter(card => card.result === 'pending').length,
  };
}

function parlayAverageOdds(cards: ParlayCard[]): number | null {
  const odds = cards.map(card => card.oddsAmerican).filter((value): value is number => value != null && Number.isFinite(value));
  return odds.length ? odds.reduce((sum, value) => sum + value, 0) / odds.length : null;
}

function parlayUniqueLegCount(cards: ParlayCard[]): number {
  return new Set(cards.flatMap(card => card.legs.map(leg => leg.legId))).size;
}

function parlayCategoryLabel(category: string): string {
  return category.split('_').map(part => part.charAt(0).toUpperCase() + part.slice(1)).join(' ');
}

function parlayCategoryOptions(cards: ParlayCard[], payload?: ParlayCardsPayload | null): ParlayViewOption[] {
  const byCategory = new Map<string, ParlayViewOption>();
  (payload?.categories || []).forEach(category => {
    const key = String(category.key || '').trim();
    if (!key) return;
    byCategory.set(key, {
      key,
      label: category.shortLabel || category.label || parlayCategoryLabel(key),
      count: 0,
      description: category.description || 'Qualified parlay cards',
    });
  });
  cards.forEach(card => {
    const key = String(card.category || '').trim();
    if (!key || byCategory.has(key)) return;
    byCategory.set(key, {
      key,
      label: card.categoryShortLabel || card.categoryLabel || parlayCategoryLabel(key),
      count: 0,
      description: card.whyQualified || 'Qualified parlay cards',
    });
  });
  return [...byCategory.values()];
}

function parlayFilterOptions(cards: ParlayCard[], payload?: ParlayCardsPayload | null): ParlayViewOption[] {
  const count = (category: ParlayView): number => category === 'all'
    ? cards.length
    : cards.filter(card => card.category === category).length;
  const options = parlayCategoryOptions(cards, payload)
    .map(option => ({ ...option, count: count(option.key) }))
    .filter(option => option.count > 0);
  return [
    { key: 'all', label: 'All', count: count('all'), description: 'Every qualified card' },
    ...options,
  ];
}

function parlayLegHtml(leg: ParlayCard['legs'][number]): string {
  const sourceMeta = [
    leg.source,
    leg.decision,
    formatAmericanOddsValue(leg.oddsAmerican),
    formatProbabilityValue(leg.estimatedProbability),
  ].filter(Boolean).join(' | ');
  const gameLine = [
    leg.game || leg.player || leg.market || 'Market',
    leg.startTime ? formatStart(leg.startTime) : '',
  ].filter(Boolean).join(' | ');
  return `<div class="parlay-leg result-${escapeHtml(leg.result)}">
    <div class="parlay-leg-main">${escapeHtml(leg.pick)}</div>
    <div class="parlay-leg-meta">${escapeHtml(sourceMeta)}</div>
    <div class="parlay-leg-sub">${escapeHtml(gameLine)}</div>
    ${parlayResultBadge(leg.result)}
  </div>`;
}

function parlayCardHtml(card: ParlayCard): string {
  const projectedPayout = card.oddsAmerican == null
    ? '--'
    : signedUnits(card.oddsAmerican > 0 ? card.oddsAmerican / 100 : 100 / Math.abs(card.oddsAmerican));
  const result = card.result || 'pending';
  const resultClass = result === 'win' ? 'win' : result === 'loss' ? 'loss' : result === 'push' ? 'push' : 'pending';
  return `<article class="parlay-card result-${resultClass}">
    <div class="parlay-header">
      <div>
        <div class="parlay-title">${escapeHtml(card.categoryShortLabel || card.categoryLabel)}</div>
        <div class="parlay-subtitle">${escapeHtml(card.sportMix)} | ${card.legCount} legs${card.fallback ? ' | fallback' : ''}</div>
      </div>
      <div class="parlay-meta"><strong>${escapeHtml(formatAmericanOddsValue(card.oddsAmerican))}</strong><span>${escapeHtml(projectedPayout)} on 1u</span></div>
    </div>
    <div class="parlay-metrics">
      <div><span>Hit Prob</span><strong>${escapeHtml(formatProbabilityValue(card.estimatedProbability))}</strong></div>
      <div><span>EV</span><strong class="${Number(card.parlayEv || 0) >= 0 ? 'positive' : 'negative'}">${escapeHtml(formatEvValue(card.parlayEv))}</strong></div>
      <div><span>Fair</span><strong>${escapeHtml(formatAmericanOddsValue(card.fairOdds))}</strong></div>
      <div><span>Result</span><strong class="${resultClass}">${result === 'pending' ? 'OPEN' : result.toUpperCase()}</strong></div>
    </div>
    <div class="parlay-leg-list">${card.legs.map(parlayLegHtml).join('')}</div>
    <div class="parlay-warn">${escapeHtml(card.whyQualified || 'Qualified by parlay engine rules.')}</div>
  </article>`;
}

function parlayGrid(cards: ParlayCard[]): string {
  if (!cards.length) {
    return '<div class="daily-empty"><div class="daily-empty-title">No qualified slips</div><div class="daily-empty-sub">The engine leaves weak slates blank instead of forcing action.</div></div>';
  }
  return `<div class="parlay-grid">${cards.map(parlayCardHtml).join('')}</div>`;
}

function parlaySections(cards: ParlayCard[], view: ParlayView, payload?: ParlayCardsPayload | null): string {
  if (view !== 'all') {
    const option = parlayFilterOptions(cards, payload).find(item => item.key === view);
    const filtered = cards.filter(card => card.category === view);
    return dailySection(option?.label || 'Parlays', option?.description || 'Qualified parlay cards.', parlayGrid(filtered), `${filtered.length} slips`);
  }
  return parlayFilterOptions(cards, payload)
    .filter(option => option.key !== 'all')
    .map(option => {
      const filtered = cards.filter(card => card.category === option.key);
      if (!filtered.length) return '';
      return dailySection(option.label, option.description, parlayGrid(filtered), `${filtered.length} slips`);
    })
    .join('') || parlayGrid([]);
}

function parlayRankingsForCards(cards: ParlayCard[], payload?: ParlayCardsPayload | null): ParlayRanking[] {
  const uniqueCards = dedupeParlayCards(cards);
  const rows = parlayFilterOptions(uniqueCards, payload)
    .filter(option => option.key !== 'all')
    .map(option => {
      const categoryCards = uniqueCards.filter(card => card.category === option.key);
      const record = parlayRecordForCards(categoryCards);
      const settled = record.wins + record.losses;
      const netUnits = Number(categoryCards.reduce((sum, card) => sum + Number(card.profitUnits || 0), 0).toFixed(2));
      const recentForm = categoryCards
        .filter(card => card.result === 'win' || card.result === 'loss' || card.result === 'push')
        .sort((left, right) => String(left.date || '').localeCompare(String(right.date || '')))
        .slice(-5)
        .map(card => card.result === 'win' ? 'W' : card.result === 'loss' ? 'L' : 'P')
        .join('');
      return {
        category: option.key,
        label: option.label,
        description: option.description,
        wins: record.wins,
        losses: record.losses,
        pushes: record.pushes,
        pending: record.pending,
        settled,
        hitRate: settled ? record.wins / settled : null,
        roi: settled ? netUnits / settled : null,
        netUnits,
        averageOdds: parlayAverageOdds(categoryCards),
        recentForm,
      };
    })
    .filter(row => row.settled || row.pushes || row.pending);
  return rows.sort((left, right) => (
    (right.settled ? 1 : 0) - (left.settled ? 1 : 0)
    || (right.hitRate || 0) - (left.hitRate || 0)
    || right.pending - left.pending
  ));
}

function parlayRankingsPanel(rankings: ParlayRanking[] = []): string {
  if (!rankings.length) {
    return dailySection('Parlay Rankings', 'Whole-slip category records appear after generated cards settle.', '<div class="daily-empty"><div class="daily-empty-title">No settled slips yet</div><div class="daily-empty-sub">Rankings update as parlay cards are graded.</div></div>');
  }
  const rows = rankings.map((row, index) => {
    const hitRate = row.hitRate == null ? '--' : `${(row.hitRate * 100).toFixed(1)}%`;
    const roi = row.roi == null ? '--' : `${(row.roi * 100).toFixed(1)}%`;
    const averageOdds = formatAmericanOddsValue(row.averageOdds);
    const slips = row.settled + row.pushes + row.pending;
    const netClass = row.netUnits > 0 ? 'positive' : row.netUnits < 0 ? 'negative' : 'neutral';
    return `<div class="daily-ranking-row rank-${index + 1}">
      <div class="daily-ranking-rank">${index + 1}</div>
      <div class="daily-ranking-source">${escapeHtml(row.label)}</div>
      <div class="daily-ranking-metric">${slips}</div>
      <div class="daily-ranking-metric">${row.wins}-${row.losses}${row.pushes ? `-${row.pushes}` : ''}</div>
      <div class="daily-ranking-metric daily-ranking-hide-sm">${escapeHtml(hitRate)}</div>
      <div class="daily-ranking-metric ${netClass}">${escapeHtml(signedUnits(row.netUnits))}</div>
      <div class="daily-ranking-metric daily-ranking-hide-sm">${escapeHtml(roi)}</div>
      <div class="daily-ranking-metric daily-ranking-hide-sm">${escapeHtml(averageOdds)}</div>
      <div class="daily-ranking-metric daily-ranking-hide-sm">${escapeHtml(row.recentForm || '--')}</div>
    </div>`;
  }).join('');
  return dailySection(
    'Parlay Rankings',
    'Whole-card records, units, ROI, average odds, and recent form from generated 1u slip tracking.',
    `<div class="daily-ranking-list"><div class="daily-ranking-row header"><div>#</div><div>Category</div><div>Slips</div><div>Record</div><div class="daily-ranking-hide-sm">Hit Rate</div><div>Units</div><div class="daily-ranking-hide-sm">ROI</div><div class="daily-ranking-hide-sm">Avg Odds</div><div class="daily-ranking-hide-sm">Form</div></div>${rows}</div>`,
  );
}

function numericSummary(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function profitDeskCandidateKey(candidate: ProfitDeskCandidate): string {
  return candidate.id || [candidate.mode, candidate.source, candidate.sport, candidate.game, candidate.player, candidate.market, candidate.pick]
    .map(value => String(value || '').trim().toLowerCase())
    .join('::');
}

function profitDeskCandidates(payload: ProfitDeskPayload | null): ProfitDeskCandidate[] {
  if (!payload) return [];
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  const portfolio = payload.portfolio && typeof payload.portfolio === 'object'
    ? Object.values(payload.portfolio).flatMap(value => Array.isArray(value) ? value : [])
    : [];
  const byKey = new Map<string, ProfitDeskCandidate>();
  [...candidates, ...portfolio].forEach(candidate => {
    if (!candidate || typeof candidate !== 'object') return;
    const key = profitDeskCandidateKey(candidate);
    const existing = byKey.get(key);
    byKey.set(key, existing ? { ...existing, ...candidate, portfolioSelected: existing.portfolioSelected || candidate.portfolioSelected || portfolio.includes(candidate) } : candidate);
  });
  return [...byKey.values()];
}

function profitDeskPortfolio(payload: ProfitDeskPayload | null): ProfitDeskCandidate[] {
  if (!payload?.portfolio || typeof payload.portfolio !== 'object') return [];
  const allPortfolio = payload.portfolio.all;
  const teamPortfolio = payload.portfolio.team;
  const playerPortfolio = payload.portfolio.player;
  return [
    ...(Array.isArray(allPortfolio) ? allPortfolio : []),
    ...(Array.isArray(teamPortfolio) ? teamPortfolio : []),
    ...(Array.isArray(playerPortfolio) ? playerPortfolio : []),
  ].filter((candidate): candidate is ProfitDeskCandidate => Boolean(candidate) && typeof candidate === 'object');
}

function profitDeskModeSummary(payload: ProfitDeskPayload | null): ProfitDeskModeSummary | null {
  return payload?.summary || null;
}

function compareProfitDeskCandidate(left: ProfitDeskCandidate, right: ProfitDeskCandidate): number {
  const leftRank = Number(left.rank);
  const rightRank = Number(right.rank);
  const safeLeftRank = Number.isFinite(leftRank) ? leftRank : Number.MAX_SAFE_INTEGER;
  const safeRightRank = Number.isFinite(rightRank) ? rightRank : Number.MAX_SAFE_INTEGER;
  return safeLeftRank - safeRightRank
    || String(left.startTime || '').localeCompare(String(right.startTime || ''))
    || String(left.pick || '').localeCompare(String(right.pick || ''));
}

function formatProfitDeskGeneratedAt(value: string | undefined): string {
  if (!value) return 'time unavailable';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString('en-US', {
    timeZone: DISPLAY_TIME_ZONE,
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short',
  });
}

function profitSummaryMetric(label: string, value: string | number, tone = '', detail = ''): string {
  return `<div class="profit-summary-metric ${escapeHtml(tone)}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${detail ? `<small>${escapeHtml(detail)}</small>` : ''}</div>`;
}

function profitDeskPriceQualityLabel(value: unknown): string {
  const labels: Record<string, string> = {
    verified_two_sided: 'Verified two-sided',
    verified_no_vig: 'Verified no-vig',
    one_sided: 'One-sided only',
    assumed: 'Assumed price',
    missing: 'Price missing',
    stale: 'Stale price',
  };
  const key = String(value || 'missing');
  return labels[key] || key.replace(/_/g, ' ');
}

function profitDeskBlockerText(blocker: ProfitDeskBlocker | string): { label: string; detail: string } {
  if (typeof blocker === 'string') return { label: blocker, detail: '' };
  return {
    label: String(blocker.label || blocker.code || 'Blocked'),
    detail: String(blocker.detail || ''),
  };
}

function profitDeskBlockersHtml(blockers: Array<ProfitDeskBlocker | string> | undefined): string {
  if (!Array.isArray(blockers) || !blockers.length) {
    return '<div class="profit-candidate-clear"><span aria-hidden="true">✓</span> No active blockers in the artifact</div>';
  }
  return `<div class="profit-blockers"><div class="profit-blockers-label">WHY IT IS BLOCKED</div><ul>${blockers.map(blocker => {
    const reason = profitDeskBlockerText(blocker);
    return `<li><strong>${escapeHtml(reason.label)}</strong>${reason.detail ? `<span>${escapeHtml(reason.detail)}</span>` : ''}</li>`;
  }).join('')}</ul></div>`;
}

function profitDeskCandidateCard(candidate: ProfitDeskCandidate): string {
  const estimate = candidate.estimate || {};
  const laneEstimate = candidate.lane === 'value' && estimate.value ? estimate.value : estimate;
  const estimateAliases = laneEstimate as typeof laneEstimate & {
    conservativeProbability?: number | null;
    probabilityPositiveEV?: number | null;
    prPositiveEv?: number | null;
  };
  const evidence = candidate.evidence || {};
  const price = candidate.price || {};
  const conservativeProbability = laneEstimate.lowerProbability ?? estimateAliases.conservativeProbability;
  const positiveEvProbability = laneEstimate.probabilityPositiveEv ?? estimateAliases.probabilityPositiveEV ?? estimateAliases.prPositiveEv;
  const marketProbability = price.noVigProbability ?? estimate.marketProbability;
  const sourceSamples = numericSummary(evidence.sourceSamples, 0);
  const segmentSamples = numericSummary(evidence.segmentSamples, 0);
  const distinctDates = numericSummary(evidence.sourceDistinctDates ?? evidence.distinctDates, 0);
  const wins = numericSummary(evidence.sourceWins ?? evidence.wins, 0);
  const losses = numericSummary(evidence.sourceLosses ?? evidence.losses, 0);
  const flatNet = evidence.sourceFlatNetUnits ?? evidence.flatNetUnits;
  const flatRoi = evidence.sourceFlatRoi ?? evidence.flatRoi;
  const age = price.ageHours == null || !Number.isFinite(Number(price.ageHours)) ? '--' : `${Number(price.ageHours).toFixed(Number(price.ageHours) < 10 ? 1 : 0)}h`;
  const freshLabel = price.fresh ? 'Fresh' : price.quality === 'missing' ? 'Missing' : 'Not fresh';
  const consensus = Array.isArray(candidate.consensusSources) ? candidate.consensusSources.filter(Boolean) : [];
  const meta = [candidate.game, candidate.player, candidate.market, candidate.startTime ? formatStart(candidate.startTime) : ''].filter(Boolean).join(' • ');
  const evidenceTone = evidence.priorOnly ? 'PRIOR-DATED EVIDENCE' : 'DATED FLAT-STAKE EVIDENCE';
  const stake = Number(candidate.stakeUnits || 0);
  const isLive = candidate.liveQualified === true && stake > 0;
  const stakeTier = String(candidate.lane || candidate.tier || 'watch').toUpperCase();
  return `<article class="profit-candidate tier-${escapeHtml(candidate.tier || 'watch')}" aria-label="${escapeHtml(candidate.pick || 'Profit Desk candidate')}">
    <div class="profit-candidate-head">
      <div><div class="profit-candidate-kicker">${candidate.rank == null ? '' : `#${escapeHtml(candidate.rank)} • `}${escapeHtml(candidate.sport || 'SPORT')} • ${escapeHtml(candidate.source || 'UNKNOWN SOURCE')}</div><h3>${escapeHtml(candidate.pick || 'Unnamed candidate')}</h3><p>${escapeHtml(meta || 'Game details unavailable')}</p></div>
      <div class="profit-shadow-stake ${isLive ? 'is-live' : ''}"><span>${escapeHtml(stakeTier)}</span><strong>${escapeHtml(`${stake}u`)}</strong><small>${isLive ? 'LIVE STAKE' : 'NOT LIVE'}</small></div>
    </div>
    <div class="profit-price-row"><span>${escapeHtml(formatAmericanOddsValue(candidate.oddsAmerican))} price</span><span class="quality-${escapeHtml(price.quality || 'missing')}">${escapeHtml(profitDeskPriceQualityLabel(price.quality))}</span><span>${escapeHtml(freshLabel)} • ${escapeHtml(age)}</span></div>
    <div class="profit-metric-grid">
      <div><span>Break-even</span><strong>${escapeHtml(formatProbabilityValue(price.breakEvenProbability))}</strong></div>
      <div><span>Market no-vig</span><strong>${escapeHtml(formatProbabilityValue(marketProbability))}</strong></div>
      <div class="primary"><span>Conservative probability</span><strong>${escapeHtml(formatProbabilityValue(conservativeProbability))}</strong></div>
      <div class="primary"><span>Expected value</span><strong>${escapeHtml(formatEvValue(laneEstimate.expectedValue))}</strong></div>
      <div><span>Pr(EV &gt; 0)</span><strong>${escapeHtml(formatProbabilityValue(positiveEvProbability))}</strong></div>
      <div><span>Market-relative alpha</span><strong>${escapeHtml(formatEvValue(laneEstimate.alpha))}</strong></div>
    </div>
    <div class="profit-evidence-row">
      <div><span>${escapeHtml(evidenceTone)}</span><strong>${sourceSamples} source / ${segmentSamples} segment samples</strong></div>
      <div><span>Coverage</span><strong>${distinctDates} distinct dates • ${wins}-${losses}</strong></div>
      <div><span>Flat-stake evidence</span><strong>${escapeHtml(flatNet == null ? '--' : signedUnits(Number(flatNet)))} • ${escapeHtml(formatEvValue(flatRoi))} ROI</strong></div>
    </div>
    <div class="profit-price-source"><strong>PRICE TRACE</strong><span>${escapeHtml(price.source || 'No observed price source')} • updated ${escapeHtml(formatProfitDeskGeneratedAt(price.updatedAt || undefined))} • ${price.twoSided ? 'two-sided market' : 'not two-sided'}</span></div>
    ${candidate.closing && candidate.closing.oddsAmerican != null ? `<div class="profit-price-source profit-closing ${Number(candidate.closing.clv || 0) >= 0 ? 'is-positive' : 'is-negative'}"><strong>CLOSING LINE</strong><span>closed ${escapeHtml(formatAmericanOddsValue(candidate.closing.oddsAmerican))}${candidate.closing.clv == null ? '' : ` • CLV ${escapeHtml(formatEvValue(candidate.closing.clv))}`}${candidate.closing.noVigProbability == null ? '' : ` • closing no-vig ${escapeHtml(formatProbabilityValue(candidate.closing.noVigProbability))}`}</span></div>` : ''}
    ${consensus.length ? `<div class="profit-consensus"><span>CONSENSUS CONTEXT</span>${consensus.map(source => `<strong>${escapeHtml(source)}</strong>`).join('')}</div>` : ''}
    ${profitDeskBlockersHtml(candidate.blockers)}
    <div class="profit-shadow-warning">${isLive
      ? `${escapeHtml(stakeTier)} lane: ${candidate.lane === 'edge' ? 'strict segment-level market-alpha evidence' : 'source-level flat-ROI evidence at posted prices'}. Recorded stake: ${escapeHtml(`${stake}u`)} flat.`
      : 'Research context is not live proof. Recorded stake: 0u.'}</div>
  </article>`;
}

function profitDeskCandidateSection(
  title: string,
  subtitle: string,
  candidates: ProfitDeskCandidate[],
  emptyTitle: string,
  emptyDetail: string,
): string {
  return `<section class="profit-section"><div class="profit-section-head"><div><div class="profit-section-kicker">PRECOMPUTED DECISION OUTPUT</div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(subtitle)}</p></div><span>${candidates.length} CANDIDATE${candidates.length === 1 ? '' : 'S'}</span></div>${candidates.length
    ? `<div class="profit-candidate-grid">${candidates.map(profitDeskCandidateCard).join('')}</div>`
    : `<div class="profit-empty"><strong>${escapeHtml(emptyTitle)}</strong><span>${escapeHtml(emptyDetail)}</span></div>`}</section>`;
}

function humanizeProfitGate(value: string): string {
  return value.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ').replace(/^./, first => first.toUpperCase());
}

function formatProfitGateValue(value: unknown): string {
  if (value == null || value === '') return 'Not set';
  if (typeof value === 'boolean') return value ? 'Required' : 'Not required';
  if (Array.isArray(value)) return value.map(item => String(item)).join(', ');
  if (typeof value === 'object') return Object.entries(value as Record<string, unknown>)
    .map(([key, nested]) => `${humanizeProfitGate(key)}: ${formatProfitGateValue(nested)}`)
    .join(' • ');
  return String(value);
}

function profitDeskMethodHtml(payload: ProfitDeskPayload | null): string {
  const gates = payload?.policy?.gates && typeof payload.policy.gates === 'object' ? Object.entries(payload.policy.gates) : [];
  const notes = Array.isArray(payload?.policy?.notes) ? payload.policy.notes : [];
  const policyStatus = payload?.policy?.status || payload?.policy?.mode || payload?.phase || 'artifact unavailable';
  return `<section class="profit-section profit-method">
    <div class="profit-section-head"><div><div class="profit-section-kicker">MARKET-ANCHORED SELECTION POLICY</div><h2>How a pick earns promotion</h2><p>Every stake starts from a real observed price with a no-vig or break-even baseline, adds only a source&rsquo;s historically proven excess over the market, shrinks thin samples toward zero, and stakes by lane (EDGE 1.0u, VALUE 0.5u) or abstains. The browser displays a dated decision artifact; it never rescores raw picks.</p></div><span>${escapeHtml(String(policyStatus).toUpperCase())}</span></div>
    <div class="profit-policy-grid">
      <div><div class="profit-policy-title">Promotion requirements</div>${gates.length ? `<dl>${gates.map(([key, value]) => `<div><dt>${escapeHtml(humanizeProfitGate(key))}</dt><dd>${escapeHtml(formatProfitGateValue(value))}</dd></div>`).join('')}</dl>` : '<p>No policy gates were published for this date. That is itself a blocker; nothing can be promoted.</p>'}</div>
      <div><div class="profit-policy-title">Where to see the evidence</div><p>Full qualification rules and each source&rsquo;s progress live under Rankings &rarr; Profit Desk Qualification. Live recordkeeping begins only after promotion${payload?.policy?.firstLiveDate ? ` on ${escapeHtml(dateLabel(payload.policy.firstLiveDate, true))}` : ''}.</p>${notes.length ? `<ul>${notes.map(note => `<li>${escapeHtml(note)}</li>`).join('')}</ul>` : ''}</div>
    </div>
  </section>`;
}

function bindProfitDeskControls(container: HTMLElement): void {
  container.querySelectorAll<HTMLButtonElement>('[data-profit-sport]').forEach(button => {
    button.addEventListener('click', () => setProfitDeskSport(button.dataset.profitSport || 'ALL'));
  });
}

function setDailyView(view: string): void {
  if (activePickMode === 'player' && view === 'consensus') view = 'picks';
  if (view === 'picks' || view === 'consensus' || view === 'sources' || view === 'research' || view === 'dayform') {
    dailyView = view;
    renderDaily();
  }
}

function setProfitView(view: string): void {
  if (view === 'card' || view === 'watchlist' || view === 'method') {
    profitView = view;
    renderProfit();
  }
}

function setProfitDeskSport(sport: string): void {
  profitDeskSport = sport || 'ALL';
  renderProfit();
}

function setParlayView(view: string): void {
  if (!view || !/^[a-z0-9_-]+$/i.test(view)) return;
  parlayView = view;
  renderParlays();
}

function setParlayResultMode(mode: string): void {
  if (mode === 'pending' || mode === 'all' || mode === 'settled') {
    parlayResultMode = mode;
    renderParlays();
  }
}

function setDailySort(sort: string): void {
  if (sort === 'time' || sort === 'percentage') {
    dailySort = sort;
    renderDaily();
  }
}

function renderDaily(): void {
  const container = document.getElementById('daily-container');
  if (!container) return;
  ensureSelection();
  const key = selectedDate || latestAvailableDateKey();
  const picks = getAllPicks().filter(pick => pickDateKey(pick) === key);
  const stats = statsFor(picks);
  const pending = picks.filter(isOpenPick);
  if (activePickMode === 'player' && dailyView === 'consensus') dailyView = 'picks';
  const forms = dailySourceForms(key, picks);
  const formsBySource = new Map(forms.map(form => [form.source, form]));
  const ranked = (candidates: Pick[]) => [...candidates].sort((a, b) => dailyPickScore(b, formsBySource) - dailyPickScore(a, formsBySource));
  const modelCalls = uniqueDailyPicks(ranked(pending.filter(isPublishedDailyPick))).slice(0, 8);
  const probabilityLeaders = uniqueDailyPicks([...pending].filter(pick => pickProbability(pick) != null)
    .sort((a, b) => (pickProbability(b) || 0) - (pickProbability(a) || 0))).slice(0, 8);
  const valueZone = uniqueDailyPicks(ranked(pending.filter(pick => isPublishedDailyPick(pick) && ((pick.odds || 0) > 0 || (pickEdgePercent(pick) || 0) >= 10)))).slice(0, 6);
  const researchQueue = uniqueDailyPicks([...pending].filter(pick => (
    (pickProbability(pick) || 0) >= 0.6 && !isPublishedDailyPick(pick)
  ) || (pick.odds != null && pick.odds <= -300)).sort((a, b) => (pickProbability(b) || 0) - (pickProbability(a) || 0))).slice(0, 6);
  const priceyCount = uniqueDailyPicks(pending.filter(pick => pick.odds != null && pick.odds <= -300)).length;
  const tagsById = new Map<string, Set<string>>();
  const addTag = (tagPicks: Pick[], tag: string): void => tagPicks.forEach(pick => {
    const tags = tagsById.get(pick.id) || new Set<string>();
    tags.add(tag);
    tagsById.set(pick.id, tags);
  });
  addTag(modelCalls, 'MODEL GREENLIGHT');
  addTag(valueZone, 'VALUE');
  addTag(probabilityLeaders, 'PROBABILITY LEADER');
  addTag(researchQueue, 'RESEARCH');
  addTag(pending.filter(pick => pick.odds != null && pick.odds <= -300), 'PRICEY FAVORITE');

  const topCandidates = [...new Map(
    [...modelCalls, ...valueZone, ...probabilityLeaders.filter(isPublishedDailyPick)]
      .map(pick => [pick.id, pick]),
  ).values()];
  const topGroups = sortDailyGroups(dailyPickGroups(topCandidates, tagsById, formsBySource, pending));
  const topKeys = new Set(topGroups.map(group => group.key));
  const playerResearchPool = activePickMode === 'player'
    ? uniqueDailyPicks(ranked(pending.filter(pick => !topKeys.has(dailyPickKey(pick)) && (
      !isPublishedDailyPick(pick) ||
      pickProbability(pick) != null ||
      pickEdgePercent(pick) != null ||
      Boolean(pick.reason || pick.rationale || pick.key_factors)
    )))).slice(0, 8)
    : [];
  addTag(playerResearchPool, 'RESEARCH');
  const researchCandidates = [...new Map(
    [...researchQueue, ...playerResearchPool, ...probabilityLeaders.filter(pick => !isPublishedDailyPick(pick))]
      .filter(pick => !topKeys.has(dailyPickKey(pick)))
      .map(pick => [pick.id, pick]),
  ).values()];
  const researchGroups = sortDailyGroups(dailyPickGroups(researchCandidates, tagsById, formsBySource, pending));
  const hotForms = forms.filter(form => form.todayCalls.length)
    .sort(compareDailySourceForms)
    .slice(0, 8);
  const dayForms = weekdaySourceForms(key, picks);
  const dayName = weekdayLabel(key) || 'Day';
  const dayFormCount = dayForms.filter(form => form.decided >= 3 && form.todayCalls.length).length;
  const games = new Map<string, Pick[]>();
  pending.forEach(pick => games.set(gameKey(pick), [...(games.get(gameKey(pick)) || []), pick]));
  const consensusCount = [...games.values()].reduce((total, gamePicks) => total + trendSignalGroups(gamePicks).filter(signal => signal.matching).length, 0);
  const viewOptionsBase: DailyViewOption[] = [
    { key: 'picks', label: 'Top Picks', count: topGroups.length, description: 'Unique actionable markets' },
    { key: 'consensus', label: 'Consensus', count: consensusCount, description: 'All matching market signals' },
    { key: 'sources', label: 'Active Sources', count: hotForms.length, description: 'Sources issuing BET/LEAN calls today' },
    { key: 'dayform', label: 'Day Form', count: dayFormCount, description: `How sources do on ${dayName}s` },
    { key: 'research', label: 'Research', count: researchGroups.length, description: activePickMode === 'player' ? 'Next-best prop candidates' : 'High probability and pricey spots' },
  ];
  const viewOptions = viewOptionsBase.filter(option => activePickMode !== 'player' || option.key !== 'consensus');
  const activeView = viewOptions.find(option => option.key === dailyView) || viewOptions[0];
  const sortOptions: Array<{ key: DailySort; label: string; description: string }> = [
    { key: 'time', label: 'By Time', description: 'Upcoming first' },
    { key: 'percentage', label: 'By Percentage', description: 'Highest model % first' },
  ];
  const activeSort = sortOptions.find(option => option.key === dailySort) || sortOptions[0];
  const dailyFocus = activePickMode === 'player' ? 'top picks, sources, day form, or research' : 'picks, consensus, sources, day form, or research';
  const researchSubtitle = activePickMode === 'player'
    ? 'Next-best player prop candidates and pass research, excluding anything already in Top Picks.'
    : 'High-probability non-published calls and expensive favorites, excluding anything already in Top Picks.';
  const activeBody = dailyView === 'picks'
    ? dailySection('Top Picks', 'Greenlights, value, and high-probability BET/LEAN calls merged into one card per market.', dailyPickGrid(topGroups), `${topGroups.length} unique markets`)
    : dailyView === 'consensus'
      ? dailySection('Consensus Signals', 'Same market selection from at least two independent sources.', dailyConsensusCards(pending), `${consensusCount} matching signals`)
      : dailyView === 'sources'
        ? dailySection('Hot Sources', 'Recent three-slate form plus each source’s unique BET/LEAN calls today.', hotForms.length ? `<div class="daily-model-grid">${hotForms.map(dailyHotModelCard).join('')}</div>` : '<div class="daily-empty"><div class="daily-empty-title">No hot source has a published call today</div><div class="daily-empty-sub">This view appears when a source has enough recent decisions and a current greenlight.</div></div>', `${hotForms.length} active sources`)
        : dailyView === 'dayform'
          ? dailySection(`For ${dayName}s`, `Every source publishing today, ranked by its record on past ${dayName}s only. Some days of the week are simply weaker — this shows who has actually delivered on this one.`, dailyDayFormBody(key, dayForms, formsBySource), `${dayFormCount} ranked source${dayFormCount === 1 ? '' : 's'} with calls today`)
          : dailySection('Research Queue', researchSubtitle, dailyPickGrid(researchGroups), `${researchGroups.length} unique markets`);

  container.innerHTML = `<div class="daily-hero"><div class="daily-hero-row"><div><div class="daily-eyebrow">TODAY'S QUICK READ</div><div class="daily-title">The Shortlist</div><div class="daily-sub">${escapeHtml(dateLabel(key, true))} | Each unique market appears once. Choose a view to focus on ${dailyFocus}.</div></div><div class="daily-clock-wrap"><div class="daily-clock-label">PICKS FOR</div><div class="daily-clock">${escapeHtml(key)}</div></div></div></div>
    <div class="daily-view-shell">
      <div class="daily-view-copy"><div class="daily-view-eyebrow">CHOOSE A VIEW</div><div class="daily-view-title">${escapeHtml(activeView.label)}</div><div class="daily-view-description">Sorted ${escapeHtml(activeSort.label.toLowerCase())}; ${stats.pending} picks remain open and ${priceyCount} are pricey favorites.</div></div>
      <div class="daily-view-nav" role="tablist" aria-label="Daily shortlist categories">${viewOptions.map(option => `<button class="daily-view-tab ${dailyView === option.key ? 'active' : ''}" type="button" role="tab" aria-selected="${dailyView === option.key}" onclick="setDailyView('${option.key}')"><span class="daily-view-tab-count">${option.count}</span><span class="daily-view-tab-label">${option.label}</span><span class="daily-view-tab-desc">${option.description}</span></button>`).join('')}</div>
      <div class="daily-controls-row">
        ${inlineDatePickerHtml('daily', dailyCalendarOpen, 'Best Bets Date')}
        <label class="daily-view-select-wrap"><span>Daily category</span><select class="daily-view-select" onchange="setDailyView(this.value)">${viewOptions.map(option => `<option value="${option.key}" ${dailyView === option.key ? 'selected' : ''}>${option.label} (${option.count})</option>`).join('')}</select></label>
        <div class="daily-sort-control" role="group" aria-label="Sort best bets">${sortOptions.map(option => `<button type="button" class="${dailySort === option.key ? 'active' : ''}" onclick="setDailySort('${option.key}')"><span>${escapeHtml(option.label)}</span><small>${escapeHtml(option.description)}</small></button>`).join('')}</div>
      </div>
    </div>
    <div class="daily-active-content">${activeBody}</div>
    <div class="daily-disclaimer"><strong>Quick read, not a blind card.</strong> Model probability estimates the chance of winning. Edge compares that chance with the market price. Recent records and consensus add context, but none guarantees the next result. Duplicate market cards are merged, with every contributing source shown inside the card.</div>`;
  bindInlineDatePicker('daily');
  bindPickCards(container);
}

function renderProfit(): void {
  const container = document.getElementById('profit-container');
  if (!container) return;
  ensureSelection();
  const key = selectedDate || latestAvailableDateKey();
  const payload = getProfitDeskPayload(key);
  const candidatePool = profitDeskCandidates(payload);
  // The Profit Desk is the whole slate's decision screen: it always shows
  // every mode so a live pick can never hide behind the team/player toggle.
  const modeCandidates = candidatePool;
  const sports = [...new Set(modeCandidates.map(candidate => String(candidate.sport || '').trim()).filter(Boolean))].sort();
  if (profitDeskSport !== 'ALL' && !sports.includes(profitDeskSport)) profitDeskSport = 'ALL';
  const filteredCandidates = modeCandidates.filter(candidate => (
    (profitDeskSport === 'ALL' || candidate.sport === profitDeskSport) &&
    (activeFilters.size === 0 || activeFilters.has(String(candidate.sport || '')) || activeFilters.has(String(candidate.source || '')))
  ));
  const portfolioKeys = new Set(profitDeskPortfolio(payload).map(profitDeskCandidateKey));
  const cardCandidates = filteredCandidates
    .filter(candidate => candidate.portfolioSelected || portfolioKeys.has(profitDeskCandidateKey(candidate)))
    .sort(compareProfitDeskCandidate);
  const watchCandidates = filteredCandidates
    .filter(candidate => (candidate.blockers?.length || 0) > 0 || candidate.tier === 'watch' || candidate.tier === 'avoid')
    .sort(compareProfitDeskCandidate);
  const modeSummary = profitDeskModeSummary(payload);
  const researchQualified = numericSummary(modeSummary?.researchQualified ?? modeSummary?.shadowQualified, modeCandidates.filter(candidate => candidate.tier === 'edge' || candidate.tier === 'value').length);
  const liveQualified = numericSummary(modeSummary?.liveQualified, 0);
  const viewOptions: Array<{ key: ProfitView; label: string; count: number; description: string }> = [
    { key: 'card', label: 'Card', count: cardCandidates.length, description: 'Qualified picks with real stakes' },
    { key: 'watchlist', label: 'Watchlist', count: watchCandidates.length, description: 'Blocked, with exact reasons' },
    { key: 'method', label: 'Method', count: 4, description: 'How a pick earns a stake' },
  ];
  const activeView = viewOptions.find(option => option.key === profitView) || viewOptions[0];
  const activeBody = profitView === 'card'
    ? profitDeskCandidateSection(
      'Live Card',
      'Evidence-qualified picks only. EDGE clears strict segment-level market-alpha gates at 1.0u; VALUE clears source-level flat-ROI gates at 0.5u.',
      cardCandidates,
      'No candidate qualified for a stake',
      payload ? 'The engine abstained after price, uncertainty, evidence, and overlap checks. Sitting out is a valid result.' : 'No dated Profit Desk artifact was published, so no substitute recommendation is shown.',
    )
    : profitView === 'watchlist'
      ? profitDeskCandidateSection(
        'Watchlist & Rejections',
        'Candidates stay here until every blocker is cleared. Reasons come directly from the precomputed policy artifact.',
        watchCandidates,
        'No blocked candidates in this scope',
        payload ? 'Try All Sports or clear a Home sport/source filter.' : 'The pipeline must publish this date before a watchlist can be evaluated.',
      )
      : profitDeskMethodHtml(payload);
  const summary = payload?.summary;
  const record = summary?.liveRecordToDate || summary?.liveRecord;
  const decisionTitle = liveQualified > 0 ? `${liveQualified} live pick${liveQualified === 1 ? '' : 's'}` : 'Sit out';
  const decisionDetail = liveQualified > 0
    ? 'Evidence-qualified picks with real stakes. EDGE stakes 1.0u; VALUE stakes 0.5u. Everything else stays 0u.'
    : researchQualified > 0
      ? `${researchQualified} research-qualified candidate${researchQualified === 1 ? '' : 's'} on a pre-live slate, all at 0u.`
      : 'No position, no forced action, and no fallback to unpriced picks.';
  const pipelineState = payload
    ? `${String(payload.policy?.status || payload.policy?.mode || payload.phase || 'live').toUpperCase()} • ${escapeHtml(payload.engineVersion || 'profit desk')} • generated ${escapeHtml(formatProfitDeskGeneratedAt(payload.generatedAt))}`
    : 'MISSING ARTIFACT • PIPELINE HAS NOT PUBLISHED THIS DATE';
  const evaluated = numericSummary(modeSummary?.candidatesEvaluated ?? modeSummary?.candidateCount, modeCandidates.length);
  const observed = numericSummary(modeSummary?.observedPriceCandidates, modeCandidates.filter(candidate => Boolean(candidate.price?.source)).length);
  const selected = numericSummary(modeSummary?.portfolioCandidates ?? modeSummary?.selected, cardCandidates.length);
  const evidenceRows = numericSummary(modeSummary?.evidenceRows, 0);

  container.innerHTML = `<div class="profit-desk-shell">
    <section class="profit-decision ${payload ? 'has-data' : 'is-missing'}" aria-labelledby="profit-desk-decision">
      <div class="profit-decision-copy">
        <div class="profit-kicker">${payload ? 'LIVE DECISION' : 'PIPELINE STATUS'} • ALL MARKETS</div>
        <h1 class="profit-decision-title" id="profit-desk-decision">${payload ? escapeHtml(decisionTitle) : 'No recommendation'}</h1>
        <p class="profit-decision-detail">${payload ? escapeHtml(decisionDetail) : `No precomputed Profit Desk artifact exists for ${escapeHtml(key)}. This screen will not improvise a recommendation from raw pick feeds or client-side scoring.`}</p>
        <div class="profit-decision-badges"><span class="profit-status-badge">${payload && liveQualified > 0 ? `${liveQualified} LIVE • STAKED` : payload && researchQualified > 0 ? 'RESEARCH ONLY • 0U' : payload ? 'SIT OUT' : 'ABSTAIN'}</span><span>${pipelineState}</span></div>
      </div>
      <div class="profit-decision-date">${inlineDatePickerHtml('profit', profitCalendarOpen, 'Profit Desk Date')}</div>
    </section>
    <section class="profit-summary" aria-label="Profit Desk summary">
      ${profitSummaryMetric('Evaluated', evaluated)}
      ${profitSummaryMetric('Observed prices', observed)}
      ${profitSummaryMetric('Qualified', researchQualified, 'accent')}
      ${profitSummaryMetric('Portfolio', selected)}
      ${profitSummaryMetric('Live picks', liveQualified, liveQualified ? 'positive' : 'neutral')}
      ${profitSummaryMetric('Evidence rows', evidenceRows)}
      ${profitSummaryMetric('Live record', record ? `${Number(record.wins || 0)}-${Number(record.losses || 0)}${record.pushes ? `-${record.pushes}` : ''}` : '0-0', '', record ? `${signedUnits(Number(record.netUnits || 0))} • ${formatEvValue(record.roi)}${record.avgClv == null ? '' : ` • CLV ${formatEvValue(record.avgClv)}`}` : 'No live proof yet')}
    </section>
    <section class="profit-controls" aria-label="Profit Desk controls">
      <div><div class="profit-control-kicker">DECISION VIEW</div><div class="profit-control-title">${escapeHtml(activeView.label)}</div><p>${escapeHtml(activeView.description)} • Board filters: ${escapeHtml(activeFilterSummary())}</p></div>
      <div class="profit-view-tabs" role="tablist" aria-label="Profit Desk views">${viewOptions.map(option => `<button type="button" role="tab" aria-selected="${profitView === option.key}" class="${profitView === option.key ? 'active' : ''}" onclick="setProfitView('${option.key}')"><strong>${option.count}</strong><span>${escapeHtml(option.label)}</span></button>`).join('')}</div>
      <div class="profit-sport-filter" role="group" aria-label="Filter Profit Desk candidates by sport"><span>SPORT</span>${['ALL', ...sports].map(sport => `<button type="button" data-profit-sport="${escapeHtml(sport)}" aria-pressed="${profitDeskSport === sport}" class="${profitDeskSport === sport ? 'active' : ''}">${escapeHtml(sport === 'ALL' ? 'All Sports' : sport)}</button>`).join('')}</div>
    </section>
    <div class="profit-active-view">${activeBody}</div>
    <aside class="profit-proof-note"><strong>What counts as evidence here:</strong> observed market prices, no-vig or break-even baselines, conservative uncertainty-adjusted estimates, and dated flat-stake tracking. A stake exists only while its lane's gates stay clear; when they fail, the correct stake is 0u.</aside>
  </div>`;
  bindInlineDatePicker('profit');
  bindProfitDeskControls(container);
}

function renderParlays(): void {
  const container = document.getElementById('parlays-container');
  if (!container) return;
  ensureSelection();
  const today = centralDateKey();
  const requestedDate = selectedDate || today;
  const payload = getParlayCardsPayload(requestedDate) || (selectedDate ? null : getParlayCardsPayload());
  const key = payload?.date || requestedDate;
  const modeCards = parlayCardsForMode(payload);
  const allCards = payload?.cards || [];
  const teamCardCount = allCards.filter(card => parlayCardPickMode(card) === 'team').length;
  const playerCardCount = allCards.filter(card => parlayCardPickMode(card) === 'player').length;
  const otherMode = activePickMode === 'team' ? 'player' : 'team';
  const otherModeCount = activePickMode === 'team' ? playerCardCount : teamCardCount;
  const visibleCards = modeCards.filter(parlayResultMatches);
  const viewOptions = parlayFilterOptions(visibleCards, payload);
  const activeView = viewOptions.find(option => option.key === parlayView) || viewOptions[0];
  if (!viewOptions.some(option => option.key === parlayView)) parlayView = 'all';
  const record = parlayRecordForCards(modeCards);
  const twoLegCards = visibleCards.filter(card => card.legCount === 2).length;
  const averageOdds = formatAmericanOddsValue(parlayAverageOdds(visibleCards));
  const boardLabel = activePickMode === 'player' ? 'Player Prop' : 'Team';
  const boardDescription = activePickMode === 'player'
    ? 'Disciplined 2-leg slips from consensus-qualified, market-priced player props.'
    : 'Disciplined 2-leg slips from sources with proven trailing edge over market prices.';
  const generatedAt = payload?.generatedAt ? formatStart(payload.generatedAt) : 'TBD';
  const resultOptions: Array<{ key: ResultMode; label: string; description: string }> = [
    { key: 'pending', label: 'Open', description: 'Unsettled slips' },
    { key: 'all', label: 'All', description: 'Every slip' },
    { key: 'settled', label: 'Results', description: 'Settled slips' },
  ];
  const emptyModeBody = !modeCards.length && otherModeCount
    ? dailySection(
      `No ${boardLabel} Slips`,
      `${otherMode === 'team' ? 'Team' : 'Player'} mode has ${otherModeCount} slip${otherModeCount === 1 ? '' : 's'} for this date.`,
      `<div class="daily-empty"><div class="daily-empty-title">Switch to ${otherMode === 'team' ? 'Team' : 'Player'} mode for this slate</div><div class="daily-empty-sub">The parlay engine keeps Team and Player boards separate so one side cannot crowd out the other.</div></div>`,
    )
    : '';
  const noDatePayloadBody = !payload
    ? dailySection(
      'No Parlay Payload',
      'No generated parlay-card file exists for the selected date.',
      '<div class="daily-empty"><div class="daily-empty-title">No slips generated for this date</div><div class="daily-empty-sub">Pick another date or wait for the next parlay-card refresh.</div></div>',
    )
    : '';
  const activeBody = noDatePayloadBody || emptyModeBody || parlaySections(visibleCards, parlayView, payload);
  const rankingsPanel = payload ? parlayRankingsPanel(parlayRankingsForCards(parlayRankingCardsForDate(key, modeCards, payload.engineVersion), payload)) : '';

  container.innerHTML = `<div class="daily-hero"><div class="daily-hero-row"><div><div class="daily-eyebrow">PARLAY BOARD</div><div class="daily-title">${escapeHtml(boardLabel)} Parlays</div><div class="daily-sub">${escapeHtml(dateLabel(key, true))} | ${escapeHtml(boardDescription)}</div></div><div class="daily-clock-wrap"><div class="daily-clock-label">SLATE</div><div class="daily-clock">${escapeHtml(key)}</div><div class="daily-countdown">Updated ${escapeHtml(generatedAt)}</div></div></div></div>
    <div class="daily-stats-strip">
      <div class="daily-stat"><div class="daily-stat-val accent">${twoLegCards}</div><div class="daily-stat-label">Shown 2-Leg Slips</div></div>
      <div class="daily-stat"><div class="daily-stat-val">${visibleCards.length}</div><div class="daily-stat-label">Shown Slips</div></div>
      <div class="daily-stat"><div class="daily-stat-val accent3">${escapeHtml(averageOdds)}</div><div class="daily-stat-label">Shown Avg Odds</div></div>
      <div class="daily-stat"><div class="daily-stat-val neutral">${escapeHtml(parlayRecordText(record))}</div><div class="daily-stat-label">Card Record</div></div>
      <div class="daily-stat"><div class="daily-stat-val">${teamCardCount}/${playerCardCount}</div><div class="daily-stat-label">Team / Player</div></div>
    </div>
    <div class="daily-view-shell">
      <div class="daily-view-copy"><div class="daily-view-eyebrow">FILTER PARLAYS</div><div class="daily-view-title">${escapeHtml(activeView.label)}</div><div class="daily-view-description">Records count each whole parlay slip once; leg results only decide whether the card wins, loses, pushes, or stays open.</div></div>
      <div class="daily-view-nav" role="tablist" aria-label="Parlay board filters">${viewOptions.map(option => `<button class="daily-view-tab ${parlayView === option.key ? 'active' : ''}" type="button" role="tab" aria-selected="${parlayView === option.key}" onclick="setParlayView(${escapeHtml(JSON.stringify(option.key))})"><span class="daily-view-tab-count">${option.count}</span><span class="daily-view-tab-label">${option.label}</span><span class="daily-view-tab-desc">${option.description}</span></button>`).join('')}</div>
      <div class="daily-controls-row">
        ${inlineDatePickerHtml('parlay', parlayCalendarOpen, 'Parlay Date')}
        <label class="daily-view-select-wrap"><span>Parlay filter</span><select class="daily-view-select" onchange="setParlayView(this.value)">${viewOptions.map(option => `<option value="${escapeHtml(option.key)}" ${parlayView === option.key ? 'selected' : ''}>${escapeHtml(option.label)} (${option.count})</option>`).join('')}</select></label>
        <div class="daily-sort-control" role="group" aria-label="Parlay result filter">${resultOptions.map(option => `<button type="button" class="${parlayResultMode === option.key ? 'active' : ''}" onclick="setParlayResultMode('${option.key}')"><span>${escapeHtml(option.label)}</span><small>${escapeHtml(option.description)}</small></button>`).join('')}</div>
      </div>
    </div>
    <div class="daily-active-content">${activeBody}${rankingsPanel}</div>
    <div class="daily-disclaimer"><strong>Parlay tracking, not a forced ticket.</strong> Odds are multiplied from individual American prices, hit probability is anchored to market-implied leg probabilities plus each source's proven trailing edge, and every generated slip is tracked at 1u as one card for rankings.</div>`;
  bindInlineDatePicker('parlay');
}

function render(): void {
  renderHome();
  updateOverallStats();
  renderRankings();
  const active = document.querySelector('.tab-content.active')?.id;
  if (active === 'tab-search') renderSearch();
  if (active === 'tab-daily') renderDaily();
  if (active === 'tab-profit') {
    renderProfit();
    renderParlays();
  }
}

function switchTab(name: string): void {
  // The standalone Parlays tab merged into Profit Desk; keep old deep
  // links working.
  if (name === 'parlays') name = 'profit';
  document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  document.querySelector<HTMLElement>(`.tab[onclick*="'${name}'"]`)?.classList.add('active');
  document.getElementById(`tab-${name}`)?.classList.add('active');
  if (name === 'home') renderHome();
  if (name === 'search') renderSearch();
  if (name === 'daily') renderDaily();
  if (name === 'profit') {
    renderProfit();
    renderParlays();
  }
}

function setHomeResultMode(mode: string): void {
  if (mode === 'pending' || mode === 'all' || mode === 'settled') {
    homeMode = mode;
    renderHome();
  }
}

function toggleHomeDatePicker(event?: Event): void {
  event?.stopPropagation();
  calendarOpen = !calendarOpen;
  dailyCalendarOpen = false;
  profitCalendarOpen = false;
  parlayCalendarOpen = false;
  renderHome();
}

function toggleDailyDatePicker(event?: Event): void {
  event?.stopPropagation();
  dailyCalendarOpen = !dailyCalendarOpen;
  calendarOpen = false;
  profitCalendarOpen = false;
  parlayCalendarOpen = false;
  renderDaily();
}

function toggleProfitDatePicker(event?: Event): void {
  event?.stopPropagation();
  profitCalendarOpen = !profitCalendarOpen;
  calendarOpen = false;
  dailyCalendarOpen = false;
  parlayCalendarOpen = false;
  renderProfit();
}

function toggleParlayDatePicker(event?: Event): void {
  event?.stopPropagation();
  parlayCalendarOpen = !parlayCalendarOpen;
  calendarOpen = false;
  dailyCalendarOpen = false;
  profitCalendarOpen = false;
  renderParlays();
}

function normalizeTeam(value: unknown): string {
  return String(value || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\b(the|baseball|basketball|club)\b/g, ' ').replace(/\s+/g, ' ').trim();
}

function canonicalTeamToken(value: unknown): string {
  const normalized = normalizeTeam(value);
  const multiWordNames = [
    'red sox',
    'white sox',
    'blue jays',
    'trail blazers',
    'golden knights',
    'maple leafs',
    'red wings',
  ];
  const suffix = multiWordNames.find(name => normalized === name || normalized.endsWith(` ${name}`));
  return suffix || normalized.split(' ').at(-1) || normalized;
}

function teamMatches(label: string, team: Record<string, unknown>): boolean {
  const target = normalizeTeam(label);
  const names = [team.displayName, team.shortDisplayName, team.name, team.abbreviation].map(normalizeTeam);
  return names.some(name => name && (name === target || name.includes(target) || target.includes(name)));
}

function teamsForPick(pick: Pick): [string, string] | null {
  if (pick.away_team && pick.home_team) return [String(pick.away_team), String(pick.home_team)];
  const matchup = gameName(pick).split(/\s+(?:vs|@)\s+/i).map(value => value.trim()).filter(Boolean);
  return matchup.length === 2 ? [matchup[0], matchup[1]] : null;
}

function findEspnEventForPick(pick: Pick, events: unknown[]): { event: Record<string, unknown>; game: Record<string, unknown> } | null {
  const teams = teamsForPick(pick);
  if (!teams) return null;
  for (const event of events) {
    if (!event || typeof event !== 'object') continue;
    const eventObject = event as Record<string, unknown>;
    const competition = eventObject.competitions;
    const game = Array.isArray(competition) ? competition[0] as Record<string, unknown> : null;
    const competitors = Array.isArray(game?.competitors) ? game.competitors as Record<string, unknown>[] : [];
    if (competitors.length !== 2) continue;
    const teamObjects = competitors.map(competitor => competitor.team as Record<string, unknown>);
    if ((teamMatches(teams[0], teamObjects[0]) && teamMatches(teams[1], teamObjects[1])) ||
        (teamMatches(teams[0], teamObjects[1]) && teamMatches(teams[1], teamObjects[0]))) return { event: eventObject, game };
  }
  return null;
}

function findEspnGame(pick: Pick, events: unknown[]): Record<string, unknown> | null {
  return findEspnEventForPick(pick, events)?.game || null;
}

function espnStatus(event: Record<string, unknown>, game: Record<string, unknown>): Record<string, unknown> {
  const status = game.status || event.status;
  if (!status || typeof status !== 'object') return {};
  const type = (status as Record<string, unknown>).type;
  return type && typeof type === 'object' ? type as Record<string, unknown> : {};
}

function homeScoreInfo(sport: string, event: Record<string, unknown>, game: Record<string, unknown>): HomeScoreInfo {
  const competitors = Array.isArray(game.competitors) ? game.competitors as Record<string, unknown>[] : [];
  const away = competitors.find(competitor => competitor.homeAway === 'away') || competitors[0] || {};
  const home = competitors.find(competitor => competitor.homeAway === 'home') || competitors[1] || {};
  const teamCode = (competitor: Record<string, unknown>): string => {
    const team = competitor.team && typeof competitor.team === 'object' ? competitor.team as Record<string, unknown> : {};
    return String(team.abbreviation || team.shortDisplayName || team.name || '').trim();
  };
  const type = espnStatus(event, game);
  const state = String(type.state || '').toLowerCase();
  const name = String(type.name || '').toUpperCase();
  const detail = String(type.shortDetail || type.detail || type.description || '').trim();
  const delayed = ['STATUS_POSTPONED', 'STATUS_SUSPENDED', 'STATUS_CANCELED', 'STATUS_CANCELLED'].includes(name);
  const final = Boolean(type.completed) || state === 'post' || ['STATUS_FINAL', 'STATUS_FULL_TIME'].includes(name);
  const live = state === 'in';
  const awayScore = Number(away.score);
  const homeScore = Number(home.score);
  const hasScore = Number.isFinite(awayScore) && Number.isFinite(homeScore);
  const score = hasScore ? `${teamCode(away)} ${awayScore} - ${teamCode(home)} ${homeScore}` : '';
  const startTime = String(game.date || event.date || '');
  const tone: HomeScoreInfo['tone'] = delayed ? 'delayed' : final ? 'final' : live ? 'live' : 'pregame';
  const text = delayed
    ? detail || 'Delayed'
    : final ? score ? `Final | ${score}` : 'Final'
      : live ? score ? `${score} | ${detail || 'Live'}` : detail || 'Live'
        : startTime ? `Starts ${formatStart(startTime)}` : detail || 'Scheduled';
  return { eventId: String(event.id || ''), sport, tone, text, startTime };
}

function homeScoreChipHtml(info: HomeScoreInfo | undefined, fallbackStart: unknown, gameLabel: string): string {
  if (!info) {
    return fallbackStart ? `<span class="home-score-chip pregame">${escapeHtml(`Starts ${formatStart(fallbackStart)}`)}</span>` : '';
  }
  const sportSlug = info.sport.toLowerCase();
  const url = info.eventId ? `https://www.espn.com/${sportSlug}/game/_/gameId/${encodeURIComponent(info.eventId)}` : '';
  const tag = url ? 'a' : 'span';
  const attrs = url ? ` href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" title="Open ESPN box score for ${escapeHtml(gameLabel)}"` : '';
  return `<${tag} class="home-score-chip ${info.tone}"${attrs}>${escapeHtml(info.text)}</${tag}>`;
}

function homeTabActive(): boolean {
  return Boolean(document.getElementById('tab-home')?.classList.contains('active'));
}

async function refreshHomeScores(date: string, picks: Pick[]): Promise<void> {
  if (!date || !picks.length || !homeTabActive()) return;
  const supported = picks.filter(pick => ESPN_ENDPOINTS[pick.sport]);
  if (!supported.length) return;
  const key = `${date}::${[...new Set(supported.map(gameKey))].sort().join('|')}`;
  const now = Date.now();
  if (homeScoreRefreshKey === key || now - (homeScoreFetches.get(key) || 0) < HOME_SCORE_TTL_MS) return;
  homeScoreRefreshKey = key;
  homeScoreFetches.set(key, now);
  let changed = false;
  const bySport = new Map<string, Pick[]>();
  supported.forEach(pick => bySport.set(pick.sport, [...(bySport.get(pick.sport) || []), pick]));

  try {
    for (const [sport, sportPicks] of bySport) {
      const endpoint = ESPN_ENDPOINTS[sport];
      try {
        const response = await fetch(`https://site.api.espn.com/apis/site/v2/sports/${endpoint[0]}/${endpoint[1]}/scoreboard?dates=${date.replace(/-/g, '')}`, { cache: 'no-store' });
        if (!response.ok) continue;
        const payload = await response.json() as { events?: unknown[] };
        sportPicks.forEach(pick => {
          const matched = findEspnEventForPick(pick, payload.events || []);
          if (!matched) return;
          const next = homeScoreInfo(sport, matched.event, matched.game);
          const previous = homeScores.get(gameKey(pick));
          if (JSON.stringify(previous) !== JSON.stringify(next)) {
            homeScores.set(gameKey(pick), next);
            changed = true;
          }
          if (next.startTime) setLocalGameTime(pick.id, next.startTime);
        });
      } catch {
        // A missing scoreboard should not prevent the rest of the Home slate from rendering.
      }
    }
  } finally {
    homeScoreRefreshKey = '';
  }

  if (changed && homeTabActive()) renderHome();
}

function scoreForTeam(game: Record<string, unknown>, label: string): [number, number] | null {
  const competitors = Array.isArray(game.competitors) ? game.competitors as Record<string, unknown>[] : [];
  const selected = competitors.find(competitor => teamMatches(label, competitor.team as Record<string, unknown>));
  const opponent = competitors.find(competitor => competitor !== selected);
  if (!selected || !opponent) return null;
  return [Number(selected.score), Number(opponent.score)];
}

function lineScoreRuns(competitor: Record<string, unknown>, inning: number): number | null {
  const linescores = Array.isArray(competitor.linescores)
    ? competitor.linescores
    : Array.isArray(competitor.lineScores) ? competitor.lineScores : [];
  const entry = linescores[inning - 1];
  if (!entry || typeof entry !== 'object') return null;
  const raw = (entry as Record<string, unknown>).value
    ?? (entry as Record<string, unknown>).score
    ?? (entry as Record<string, unknown>).runs;
  const runs = Number(raw);
  return Number.isFinite(runs) ? runs : null;
}

function firstFiveRuns(competitor: Record<string, unknown>): number | null {
  const runs = Array.from({ length: 5 }, (_, index) => lineScoreRuns(competitor, index + 1));
  return runs.some(value => value == null) ? null : runs.reduce<number>((sum, value) => sum + Number(value), 0);
}

function firstFiveScoreForTeam(game: Record<string, unknown>, label: string): [number, number] | null {
  const competitors = Array.isArray(game.competitors) ? game.competitors as Record<string, unknown>[] : [];
  const selected = competitors.find(competitor => teamMatches(label, competitor.team as Record<string, unknown>));
  const opponent = competitors.find(competitor => competitor !== selected);
  if (!selected || !opponent) return null;
  const selectedRuns = firstFiveRuns(selected);
  const opponentRuns = firstFiveRuns(opponent);
  return selectedRuns == null || opponentRuns == null ? null : [selectedRuns, opponentRuns];
}

function gradePick(pick: Pick, game: Record<string, unknown>): PickResult {
  const text = pick.pick.split('(', 1)[0].trim();
  const lower = text.toLowerCase();
  const competitors = Array.isArray(game.competitors) ? game.competitors as Record<string, unknown>[] : [];
  const scores = competitors.map(competitor => Number(competitor.score));
  if (scores.some(score => !Number.isFinite(score))) return 'pending';

  const noRunInning = lower.match(/\binning\s+([1-8])\s*[-:–—]?\s*no\s+runs?\s+scored\b/);
  if (pick.sport === 'MLB' && noRunInning) {
    const runs = competitors.map(competitor => lineScoreRuns(competitor, Number(noRunInning[1])));
    if (runs.some(value => value == null)) return 'pending';
    return runs.reduce<number>((sum, value) => sum + Number(value), 0) === 0 ? 'win' : 'loss';
  }

  const firstFive = pick.sport === 'MLB' && (
    ['f5_side', 'f5_total', 'first_five', 'first-five'].includes(String(pick.market || '').toLowerCase())
    || /\bf5\b|first\s*five/.test(lower)
  );
  if (firstFive) {
    const totalMatch = lower.match(/\b(over|under)\s+(\d+(?:\.\d+)?)\s*(?:f5|first\s*five)\b/);
    if (totalMatch) {
      const firstFiveScores = competitors.map(firstFiveRuns);
      if (firstFiveScores.some(value => value == null)) return 'pending';
      const total = firstFiveScores.reduce<number>((sum, value) => sum + Number(value), 0);
      const line = Number(totalMatch[2]);
      if (total === line) return 'push';
      return totalMatch[1] === 'over' ? (total > line ? 'win' : 'loss') : (total < line ? 'win' : 'loss');
    }
    const team = String(pick.team || text.replace(/\s+(?:f5|first\s*five)\s+ml\b/i, '')).trim();
    const score = firstFiveScoreForTeam(game, team);
    if (!score) return 'pending';
    return score[0] === score[1] ? 'push' : score[0] > score[1] ? 'win' : 'loss';
  }

  const total = scores[0] + scores[1];
  const totalMatch = lower.match(/\b(over|under)\s+(\d+(?:\.\d+)?)/);
  const teamGoals = lower.match(/^(.*?)\s+(over|under)\s+(\d+(?:\.\d+)?)\s*tg\b/);
  const teamTotal = lower.match(/^(.*?)\s+team total\s+(over|under)\s+(\d+(?:\.\d+)?)/);
  const teamMarket = teamTotal || teamGoals;
  if (teamMarket) {
    const score = scoreForTeam(game, teamMarket[1]);
    if (!score) return 'pending';
    const line = Number(teamMarket[3]);
    if (score[0] === line) return 'push';
    return teamMarket[2] === 'over' ? (score[0] > line ? 'win' : 'loss') : (score[0] < line ? 'win' : 'loss');
  }
  if (totalMatch) {
    const line = Number(totalMatch[2]);
    if (total === line) return 'push';
    return totalMatch[1] === 'over' ? (total > line ? 'win' : 'loss') : (total < line ? 'win' : 'loss');
  }
  const spread = text.match(/^(.*?)\s+([+-]\d+(?:\.\d+)?)/);
  if (spread) {
    const score = scoreForTeam(game, spread[1]);
    if (!score) return 'pending';
    const adjusted = score[0] + Number(spread[2]);
    return adjusted === score[1] ? 'push' : adjusted > score[1] ? 'win' : 'loss';
  }
  const moneyline = text.match(/^(.*?)\s+(?:ML|moneyline|to win|wins?)\b/i);
  const team = moneyline?.[1] || String(pick.team || '').trim();
  if (team) {
    const score = scoreForTeam(game, team);
    if (!score) return 'pending';
    return score[0] === score[1] ? 'push' : score[0] > score[1] ? 'win' : 'loss';
  }
  return 'pending';
}

type PlayerPropDescriptor = {
  playerName: string;
  playerIds: string[];
  statKey: string;
  selection: 'OVER' | 'UNDER';
  line: number;
};

function playerPropDescriptor(pick: Pick): PlayerPropDescriptor | null {
  const playerName = String(pick.player_name || pick.player || '').trim();
  const playerIds = [pick.player_id, pick.market_athlete_id]
    .map(value => String(value || '').trim())
    .filter(Boolean);
  const aliases: Record<string, string> = {
    totalrebounds: 'rebounds',
    threepointersmade: 'three_pointers_made',
    pointsreboundsassists: 'points_rebounds_assists',
    pointsrebounds: 'points_rebounds',
    pointsassists: 'points_assists',
    stealsblocks: 'steals_blocks',
    hitsrunsrbis: 'hits_runs_rbis',
    batterwalks: 'batter_walks',
    batterstrikeouts: 'batter_strikeouts',
    totalbases: 'total_bases',
    homeruns: 'home_runs',
    stolenbases: 'stolen_bases',
    pitcherwalksallowed: 'pitcher_walks_allowed',
    pitcheroutsrecorded: 'pitcher_outs_recorded',
    pitcherhitsallowed: 'pitcher_hits_allowed',
    pitcherearnedrunsallowed: 'pitcher_earned_runs_allowed',
  };
  const rawStatKey = String(pick.stat_key || pick.market || '').trim().toLowerCase();
  const compactStatKey = rawStatKey.replace(/[^a-z0-9]/g, '');
  const statKey = aliases[compactStatKey] || rawStatKey.replace(/\s+/g, '_');
  const selection = String(pick.selection || pick.direction || '').trim().toUpperCase();
  const line = Number(pick.line ?? pick.market_line);
  if (!playerName || !statKey || (selection !== 'OVER' && selection !== 'UNDER') || !Number.isFinite(line)) return null;
  return { playerName, playerIds, statKey, selection, line };
}

function normalizePersonName(value: unknown): string[] {
  return String(value || '').toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
    .replace(/\b(jr|sr|ii|iii|iv)\b/g, '').replace(/[^a-z0-9 ]/g, ' ')
    .replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
}

function personNamesMatch(left: unknown, right: unknown): boolean {
  const a = normalizePersonName(left);
  const b = normalizePersonName(right);
  if (!a.length || !b.length || a.at(-1) !== b.at(-1)) return false;
  return a.join(' ') === b.join(' ') || a[0] === b[0] || a[0]?.[0] === b[0]?.[0];
}

function numericStat(value: unknown): number | null {
  if (value == null || value === '') return null;
  const text = String(value).split('/', 1)[0].trim();
  const number = Number(text);
  return Number.isFinite(number) ? number : null;
}

function inningsToOuts(value: unknown): number | null {
  const text = String(value ?? '').trim();
  if (!text) return null;
  const [innings, partial = '0'] = text.split('.', 2);
  const whole = Number(innings);
  const remainder = Number(partial.slice(0, 1));
  return Number.isFinite(whole) && Number.isFinite(remainder) ? (whole * 3) + Math.min(2, Math.max(0, remainder)) : null;
}

function mlbLivePlayerRecord(feed: Record<string, unknown>, descriptor: PlayerPropDescriptor): Record<string, unknown> | null {
  const liveData = feed.liveData && typeof feed.liveData === 'object' ? feed.liveData as Record<string, unknown> : {};
  const boxscore = liveData.boxscore && typeof liveData.boxscore === 'object' ? liveData.boxscore as Record<string, unknown> : {};
  const teams = boxscore.teams && typeof boxscore.teams === 'object' ? boxscore.teams as Record<string, unknown> : {};
  let nameMatch: Record<string, unknown> | null = null;
  for (const side of ['away', 'home']) {
    const team = teams[side] && typeof teams[side] === 'object' ? teams[side] as Record<string, unknown> : {};
    const roster = team.players && typeof team.players === 'object' ? team.players as Record<string, unknown> : {};
    for (const candidate of Object.values(roster)) {
      if (!candidate || typeof candidate !== 'object') continue;
      const record = candidate as Record<string, unknown>;
      const person = record.person && typeof record.person === 'object' ? record.person as Record<string, unknown> : {};
      const personId = String(person.id || '').trim();
      if (personId && descriptor.playerIds.includes(personId)) return record;
      if (!nameMatch && personNamesMatch(descriptor.playerName, person.fullName)) nameMatch = record;
    }
  }
  return nameMatch;
}

function mlbLivePlayerHasActivity(feed: Record<string, unknown>, descriptor: PlayerPropDescriptor): boolean {
  const player = mlbLivePlayerRecord(feed, descriptor);
  if (!player) return false;
  const stats = player.stats && typeof player.stats === 'object' ? player.stats as Record<string, unknown> : {};
  const batting = stats.batting && typeof stats.batting === 'object' ? stats.batting as Record<string, unknown> : {};
  const pitching = stats.pitching && typeof stats.pitching === 'object' ? stats.pitching as Record<string, unknown> : {};
  return Object.keys(batting).length > 0 || Object.keys(pitching).length > 0;
}

function espnPlayerStat(summary: Record<string, unknown>, descriptor: PlayerPropDescriptor): number | null {
  const boxscore = summary.boxscore && typeof summary.boxscore === 'object' ? summary.boxscore as Record<string, unknown> : {};
  const players = Array.isArray(boxscore.players) ? boxscore.players as Record<string, unknown>[] : [];
  const values: Record<string, number> = {};
  for (const teamBlock of players) {
    const sections = Array.isArray(teamBlock.statistics) ? teamBlock.statistics as Record<string, unknown>[] : [];
    for (const section of sections) {
      const labels = Array.isArray(section.labels) ? section.labels.map(label => String(label).trim().toUpperCase()) : [];
      const athletes = Array.isArray(section.athletes) ? section.athletes as Record<string, unknown>[] : [];
      for (const athlete of athletes) {
        const person = athlete.athlete && typeof athlete.athlete === 'object' ? athlete.athlete as Record<string, unknown> : {};
        if (!personNamesMatch(descriptor.playerName, person.displayName)) continue;
        const stats = Array.isArray(athlete.stats) ? athlete.stats : [];
        labels.forEach((label, index) => {
          const value = label === 'IP' ? inningsToOuts(stats[index]) : numericStat(stats[index]);
          if (value != null) values[label] = value;
        });
      }
    }
  }
  const components: Record<string, string[]> = {
    points_rebounds_assists: ['points', 'rebounds', 'assists'],
    points_rebounds: ['points', 'rebounds'],
    points_assists: ['points', 'assists'],
    steals_blocks: ['steals', 'blocks'],
    hits_runs_rbis: ['hits', 'runs', 'rbis'],
  };
  if (components[descriptor.statKey]) {
    const numbers = components[descriptor.statKey].map(statKey => espnPlayerStat(summary, { ...descriptor, statKey }));
    return numbers.every(value => value != null) ? numbers.reduce<number>((sum, value) => sum + Number(value), 0) : null;
  }
  const labels: Record<string, string[]> = {
    points: ['PTS'], rebounds: ['REB', 'TREB', 'TOTREB', 'TOTAL REBOUNDS'], assists: ['AST'],
    three_pointers_made: ['3PM', '3PT', '3FGM', 'FG3M'], steals: ['STL'], blocks: ['BLK'],
    hits: ['H', 'HITS'], runs: ['R', 'RUNS'], rbis: ['RBI', 'RBIS'], batter_walks: ['BB', 'WALKS'],
    batter_strikeouts: ['K', 'SO'], doubles: ['2B'], triples: ['3B'], home_runs: ['HR'],
    stolen_bases: ['SB'], strikeouts: ['K', 'SO'], pitcher_walks_allowed: ['BB', 'WALKS'],
    pitcher_outs_recorded: ['IP'], pitcher_hits_allowed: ['H', 'HITS'], pitcher_earned_runs_allowed: ['ER'],
  };
  for (const label of labels[descriptor.statKey] || []) if (values[label] != null) return values[label];
  const hits = values.H;
  const doubles = values['2B'];
  const triples = values['3B'];
  const homers = values.HR;
  if ([hits, doubles, triples, homers].every(value => value != null)) {
    const singles = Math.max(0, hits - doubles - triples - homers);
    if (descriptor.statKey === 'singles') return singles;
    if (descriptor.statKey === 'total_bases') return singles + (2 * doubles) + (3 * triples) + (4 * homers);
  }
  return null;
}

function mlbLivePlayerStat(feed: Record<string, unknown>, descriptor: PlayerPropDescriptor): number | null {
  const player = mlbLivePlayerRecord(feed, descriptor);
  if (!player) return null;
  const stats = player.stats && typeof player.stats === 'object' ? player.stats as Record<string, unknown> : {};
  const batting = stats.batting && typeof stats.batting === 'object' ? stats.batting as Record<string, unknown> : {};
  const pitching = stats.pitching && typeof stats.pitching === 'object' ? stats.pitching as Record<string, unknown> : {};
  const get = (record: Record<string, unknown>, key: string): number | null => numericStat(record[key]);
  const batterKeys: Record<string, string> = {
    hits: 'hits', runs: 'runs', rbis: 'rbi', batter_walks: 'baseOnBalls', batter_strikeouts: 'strikeOuts',
    doubles: 'doubles', triples: 'triples', home_runs: 'homeRuns', stolen_bases: 'stolenBases',
  };
  const pitcherKeys: Record<string, string> = {
    strikeouts: 'strikeOuts', pitcher_walks_allowed: 'baseOnBalls', pitcher_outs_recorded: 'outs',
    pitcher_hits_allowed: 'hits', pitcher_earned_runs_allowed: 'earnedRuns',
  };
  if (batterKeys[descriptor.statKey]) return get(batting, batterKeys[descriptor.statKey]);
  if (pitcherKeys[descriptor.statKey]) return get(pitching, pitcherKeys[descriptor.statKey]);
  if (descriptor.statKey === 'hits_runs_rbis') {
    const values = ['hits', 'runs', 'rbi'].map(key => get(batting, key));
    return values.every(value => value != null) ? values.reduce<number>((sum, value) => sum + Number(value), 0) : null;
  }
  const totalBases = get(batting, 'totalBases');
  if (descriptor.statKey === 'total_bases' && totalBases != null) return totalBases;
  const values = ['hits', 'doubles', 'triples', 'homeRuns'].map(key => get(batting, key));
  if (values.every(value => value != null)) {
    const [hits, doubles, triples, homers] = values.map(Number);
    const singles = Math.max(0, hits - doubles - triples - homers);
    if (descriptor.statKey === 'singles') return singles;
    if (descriptor.statKey === 'total_bases') return singles + (2 * doubles) + (3 * triples) + (4 * homers);
  }
  return null;
}

function gradePlayerValue(descriptor: PlayerPropDescriptor, actual: number | null): PickResult {
  if (actual == null) return 'pending';
  if (actual === descriptor.line) return 'push';
  return descriptor.selection === 'OVER'
    ? actual > descriptor.line ? 'win' : 'loss'
    : actual < descriptor.line ? 'win' : 'loss';
}

function gradeMlbPlayerValue(descriptor: PlayerPropDescriptor, feed: Record<string, unknown>): PickResult {
  const result = gradePlayerValue(descriptor, mlbLivePlayerStat(feed, descriptor));
  if (result !== 'pending') return result;
  if (mlbGameIsFinal(feed) && mlbLivePlayerRecord(feed, descriptor) && !mlbLivePlayerHasActivity(feed, descriptor)) return 'push';
  return 'pending';
}

async function fetchRemoteJson(url: string): Promise<Record<string, unknown> | null> {
  try {
    const response = await fetch(url, { cache: 'no-store' });
    return response.ok ? await response.json() as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function findMlbGamePk(schedule: Record<string, unknown>, pick: Pick): string {
  const teams = teamsForPick(pick);
  if (!teams) return '';
  const expected = new Set(teams.map(canonicalTeamToken));
  const dates = Array.isArray(schedule.dates) ? schedule.dates as Record<string, unknown>[] : [];
  for (const date of dates) {
    const games = Array.isArray(date.games) ? date.games as Record<string, unknown>[] : [];
    for (const game of games) {
      const gameTeams = game.teams && typeof game.teams === 'object' ? game.teams as Record<string, unknown> : {};
      const actual = ['away', 'home'].map(side => {
        const entry = gameTeams[side] && typeof gameTeams[side] === 'object' ? gameTeams[side] as Record<string, unknown> : {};
        const team = entry.team && typeof entry.team === 'object' ? entry.team as Record<string, unknown> : {};
        return canonicalTeamToken(team.name);
      });
      if (actual.every(team => expected.has(team))) return String(game.gamePk || '');
    }
  }
  return '';
}

function mlbGameIsFinal(feed: Record<string, unknown>): boolean {
  const gameData = feed.gameData && typeof feed.gameData === 'object' ? feed.gameData as Record<string, unknown> : {};
  const status = gameData.status && typeof gameData.status === 'object' ? gameData.status as Record<string, unknown> : {};
  const abstract = String(status.abstractGameState || '').trim().toLowerCase();
  const coded = String(status.codedGameState || '').trim().toUpperCase();
  return abstract === 'final' || coded === 'F';
}

async function gradeDate(date: string, picks: Pick[]): Promise<number> {
  let graded = 0;
  const dateParam = date.replace(/-/g, '');
  const summaryCache = new Map<string, Record<string, unknown> | null>();
  const mlbFeedCache = new Map<string, Record<string, unknown> | null>();
  let mlbSchedule: Record<string, unknown> | null | undefined;
  for (const [sport, endpoint] of Object.entries(ESPN_ENDPOINTS)) {
    const sportPicks = picks.filter(pick => pick.sport === sport && isOpenPick(pick));
    if (!sportPicks.length) continue;
    try {
      const response = await fetch(`https://site.api.espn.com/apis/site/v2/sports/${endpoint[0]}/${endpoint[1]}/scoreboard?dates=${dateParam}`, { cache: 'no-store' });
      if (!response.ok) continue;
      const payload = await response.json() as { events?: unknown[] };
      for (const pick of sportPicks) {
        const matched = findEspnEventForPick(pick, payload.events || []);
        if (!matched) continue;
        const { event, game } = matched;
        const startTime = String(game.date || '');
        if (startTime) setLocalGameTime(pick.id, startTime);
        const status = game.status as { type?: { completed?: boolean; name?: string } } | undefined;
        const statusName = String(status?.type?.name || '').toUpperCase();
        if (['STATUS_POSTPONED', 'STATUS_CANCELED', 'STATUS_CANCELLED'].includes(statusName)) {
          setLocalResult(pick.id, 'push');
          graded += 1;
        } else if (status?.type?.completed) {
          let result: PickResult = 'pending';
          const descriptor = playerPropDescriptor(pick);
          if (!descriptor) {
            result = gradePick(pick, game);
          } else if (sport === 'MLB') {
            if (mlbSchedule === undefined) {
              mlbSchedule = await fetchRemoteJson(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${encodeURIComponent(date)}&hydrate=team`);
            }
            const gamePk = mlbSchedule ? findMlbGamePk(mlbSchedule, pick) : '';
            if (gamePk && !mlbFeedCache.has(gamePk)) {
              mlbFeedCache.set(gamePk, await fetchRemoteJson(`https://statsapi.mlb.com/api/v1.1/game/${encodeURIComponent(gamePk)}/feed/live`));
            }
            result = gamePk ? gradeMlbPlayerValue(descriptor, mlbFeedCache.get(gamePk) || {}) : 'pending';
          } else {
            const eventId = String(event.id || '');
            if (eventId && !summaryCache.has(eventId)) {
              summaryCache.set(eventId, await fetchRemoteJson(`https://site.api.espn.com/apis/site/v2/sports/${endpoint[0]}/${endpoint[1]}/summary?event=${encodeURIComponent(eventId)}`));
            }
            result = gradePlayerValue(descriptor, eventId ? espnPlayerStat(summaryCache.get(eventId) || {}, descriptor) : null);
          }
          if (result !== 'pending') {
            setLocalResult(pick.id, result);
            graded += 1;
          }
        }
      }
    } catch {
      // One unavailable scoreboard should not block other sports or dates.
    }
  }

  const mlbPlayerPending = picks.filter(pick => (
    pick.sport === 'MLB'
    && isOpenPick(pick)
    && playerPropDescriptor(pick)
  ));
  if (mlbPlayerPending.length) {
    if (mlbSchedule === undefined) {
      mlbSchedule = await fetchRemoteJson(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${encodeURIComponent(date)}&hydrate=team`);
    }
    for (const pick of mlbPlayerPending) {
      const descriptor = playerPropDescriptor(pick);
      if (!descriptor) continue;
      const gamePk = mlbSchedule ? findMlbGamePk(mlbSchedule, pick) : '';
      if (!gamePk) continue;
      if (!mlbFeedCache.has(gamePk)) {
        mlbFeedCache.set(gamePk, await fetchRemoteJson(`https://statsapi.mlb.com/api/v1.1/game/${encodeURIComponent(gamePk)}/feed/live`));
      }
      const feed = mlbFeedCache.get(gamePk) || {};
      if (!mlbGameIsFinal(feed)) continue;
      const result = gradeMlbPlayerValue(descriptor, feed);
      if (result !== 'pending') {
        setLocalResult(pick.id, result);
        graded += 1;
      }
    }
  }

  return graded;
}

async function refreshAutoGrades(): Promise<void> {
  if (refreshInFlight) return;
  refreshInFlight = true;
  setRefreshStatus('Checking for fresh picks and final scores...');
  const button = document.getElementById('refresh-btn') as HTMLButtonElement | null;
  if (button) button.disabled = true;
  try {
    await loadAllData();
    updateSyncStatus();
    const pending = getAllPicks().filter(isOpenPick);
    const byDate = new Map<string, Pick[]>();
    pending.forEach(pick => byDate.set(pickDateKey(pick), [...(byDate.get(pickDateKey(pick)) || []), pick]));
    let graded = 0;
    for (const [date, picks] of byDate) graded += await gradeDate(date, picks);
    render();
    setRefreshStatus(graded
      ? `Updated ${graded} finished pick${graded === 1 ? '' : 's'}`
      : activePickMode === 'player' ? 'Player props refreshed' : 'You’re up to date — no new final scores', 'ok');
  } catch {
    setRefreshStatus('Couldn’t check for updates right now', 'error');
  } finally {
    refreshInFlight = false;
    if (button) button.disabled = false;
  }
}

async function refreshForCentralClock(): Promise<void> {
  const today = centralDateKey();
  if (today !== lastCentralDate && followCentralToday) {
    selectedDate = today;
    calendarMonth = today.slice(0, 7);
  }
  lastCentralDate = today;
  await refreshAutoGrades();
}

function updatedAgoLabel(value: string): string {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return '';
  const minutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60_000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hr${hours === 1 ? '' : 's'} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? '' : 's'} ago`;
}

function updateSyncStatus(): void {
  const status = getCacheStatus();
  latestPicksUpdatedAt = status.updatedAt;
  const syncStatus = document.getElementById('sync-status');
  if (syncStatus) syncStatus.textContent = status.updatedAt
    ? `Picks updated ${updatedAgoLabel(status.updatedAt)}${status.runTime ? ` • ${status.runTime}` : ''}`
    : 'Latest pick update time unavailable';
}

function switchPickMode(mode: PickMode): void {
  activePickMode = mode;
  setDataPickMode(mode);
  activeFilters.clear();
  homeMode = 'pending';
  dailyView = 'picks';
  profitView = 'card';
  profitDeskSport = 'ALL';
  parlayView = 'all';
  parlayResultMode = 'pending';
  selectedDate = '';
  followCentralToday = true;
  calendarMonth = '';
  calendarOpen = false;
  dailyCalendarOpen = false;
  profitCalendarOpen = false;
  parlayCalendarOpen = false;
  filterMoreOpen = false;
  expandedSourceKeys.clear();
  expandedResearchPickKeys.clear();
  homeScores.clear();
  const search = document.getElementById('search-input') as HTMLInputElement | null;
  if (search) {
    search.value = '';
    search.placeholder = mode === 'player'
      ? 'Find a player, prop, matchup, or source in the selected date’s open props...'
      : 'Find a team, matchup, or source in the selected date’s open picks...';
  }
  updateSyncStatus();
  render();
}

function goHome(event?: Event): void {
  event?.preventDefault();
  switchTab('home');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

Object.assign(window, {
  switchTab,
  goHome,
  setHomeResultMode,
  setDailyView,
  setProfitView,
  setProfitDeskSport,
  setParlayView,
  setParlayResultMode,
  setDailySort,
  toggleHomeDatePicker,
  toggleDailyDatePicker,
  toggleProfitDatePicker,
  toggleParlayDatePicker,
  refreshAutoGrades,
  renderSearch,
});

document.addEventListener('click', event => {
  const wrap = document.getElementById('home-date-wrap');
  if (calendarOpen && wrap && !wrap.contains(event.target as Node)) {
    calendarOpen = false;
    renderHome();
  }
  const dailyWrap = document.getElementById('daily-date-wrap');
  if (dailyCalendarOpen && dailyWrap && !dailyWrap.contains(event.target as Node)) {
    dailyCalendarOpen = false;
    renderDaily();
  }
  const profitWrap = document.getElementById('profit-date-wrap');
  if (profitCalendarOpen && profitWrap && !profitWrap.contains(event.target as Node)) {
    profitCalendarOpen = false;
    renderProfit();
  }
  const parlayWrap = document.getElementById('parlay-date-wrap');
  if (parlayCalendarOpen && parlayWrap && !parlayWrap.contains(event.target as Node)) {
    parlayCalendarOpen = false;
    renderParlays();
  }
  const filterWrap = document.getElementById('filter-more-wrap');
  if (filterMoreOpen && filterWrap && !filterWrap.contains(event.target as Node)) {
    filterMoreOpen = false;
    renderFilters();
  }
});

document.addEventListener('pickledger:modechange', event => {
  const mode = (event as CustomEvent<{ mode?: PickMode }>).detail?.mode;
  if (mode === 'team' || mode === 'player') switchPickMode(mode);
});

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  initMobileMode();
  activePickMode = initPickMode();
  setDataPickMode(activePickMode);
  initSettingsUI();
  await loadAllData();
  lastCentralDate = centralDateKey();
  updateSyncStatus();
  render();
  window.setInterval(() => {
    if (latestPicksUpdatedAt) updateSyncStatus();
  }, 60_000);
  window.setInterval(() => void refreshForCentralClock(), AUTO_REFRESH_MS);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') void refreshForCentralClock();
  });
});
