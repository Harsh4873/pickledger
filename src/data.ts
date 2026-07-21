export type PickResult = 'pending' | 'win' | 'loss' | 'push';
export type PickMode = 'team' | 'player';

export interface Pick {
  id: string;
  source: string;
  pick: string;
  sport: string;
  date: string;
  units: number;
  odds: number | null;
  price_verified?: boolean;
  price_provenance?: 'verified' | 'unverified' | 'assumed' | 'missing';
  result: PickResult;
  pl: number;
  probability?: number | null;
  confidence?: number | string | null;
  start_time?: string | null;
  game_start_time?: string | null;
  away_team?: string;
  home_team?: string;
  team?: string;
  matchup?: string;
  game?: string;
  decision?: string;
  edge?: number | null;
  market_edge?: number | null;
  line?: number | null;
  market_line?: number | null;
  kelly?: number | string | null;
  kelly_units?: number | string | null;
  full_kelly?: number | string | null;
  quarter_kelly?: number | string | null;
  recommended_units?: number | string | null;
  reason?: string | null;
  rationale?: string | null;
  key_factors?: unknown;
  player?: string;
  player_name?: string;
  market?: string;
  scope?: string;
  external_player_feed?: boolean;
  ml_rank?: number | string | null;
  model_rank?: number | string | null;
  rank?: number | string | null;
  ml_rank_epoch?: string | null;
  ranking_epoch?: string | null;
  ranking_updated_at?: string | null;
  model_epoch?: string | null;
  consensus_applicable_models?: unknown;
  consensus_record_models?: unknown;
  [key: string]: unknown;
}

interface ModelBucket {
  ok?: boolean;
  picks?: unknown[];
  games?: unknown[];
  [key: string]: unknown;
}

interface ModelCachePayload {
  date?: string;
  generatedAt?: string;
  updatedAt?: string;
  models?: Record<string, ModelBucket>;
  [key: string]: unknown;
}

interface CacheManifest {
  files?: string[];
}

interface PlayerPropsPayload {
  date?: string;
  slate_date?: string;
  generatedAt?: string;
  updatedAt?: string;
  picks?: unknown[];
  props?: unknown[];
  player_props?: unknown[];
  recommendations?: unknown[];
  models?: Record<string, unknown>;
  sports?: Record<string, unknown>;
  leagues?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ParlayLeg {
  legId: string;
  pickId?: string;
  source: string;
  sourceType?: string;
  sport: string;
  pick: string;
  decision?: string;
  oddsAmerican: number | null;
  decimalOdds?: number | null;
  estimatedProbability?: number | null;
  probabilitySource?: string;
  game?: string;
  market?: string;
  player?: string;
  result: PickResult;
  startTime?: string;
  consensusSources?: string[];
}

export interface ParlayCard {
  id: string;
  comboKey?: string;
  date?: string;
  category: string;
  categoryLabel: string;
  categoryShortLabel?: string;
  title?: string;
  fallback?: boolean;
  legCount: number;
  activeLegCount?: number;
  sportMix: string;
  sportPattern?: string;
  sports?: string[];
  hasPlayerProp?: boolean;
  pickMode?: PickMode | 'mixed';
  oddsAmerican: number | null;
  decimalOdds?: number | null;
  estimatedProbability?: number | null;
  geomeanProbability?: number | null;
  fairOdds?: number | null;
  parlayEv?: number | null;
  payoutQuality?: number | null;
  averageSourceForm?: number | null;
  consensusLegs?: number;
  result: PickResult;
  profitUnits?: number | null;
  stakeUnits?: number | null;
  whyQualified?: string;
  legs: ParlayLeg[];
}

export interface ParlayCategorySummary {
  key: string;
  label: string;
  shortLabel?: string;
  description?: string;
  count: number;
  threeLegCount?: number;
  fallbackCount?: number;
  weight?: number;
  record?: {
    wins?: number;
    losses?: number;
    pushes?: number;
    pending?: number;
    hitRate?: number | null;
    roi?: number | null;
    netUnits?: number | null;
    averageOdds?: number | null;
    recentForm?: string;
  };
}

export interface ParlayRanking {
  category: string;
  label: string;
  description?: string;
  wins: number;
  losses: number;
  pushes: number;
  pending: number;
  settled: number;
  hitRate: number | null;
  roi: number | null;
  netUnits: number;
  averageOdds: number | null;
  recentForm?: string;
}

export interface ParlayCardsPayload {
  date: string;
  generatedAt?: string;
  engineVersion?: string;
  summary?: {
    eligibleLegs?: number;
    generatedThreeLegCandidates?: number;
    displayedCards?: number;
    threeLegCards?: number;
    twoLegFallbackCards?: number;
    averageOdds?: number | null;
    record?: ParlayCategorySummary['record'];
  };
  categories?: ParlayCategorySummary[];
  rankings?: ParlayRanking[];
  cards?: ParlayCard[];
  notices?: string[];
  [key: string]: unknown;
}

export type ProfitDeskTier = 'edge' | 'value' | 'shadow' | 'watch' | 'avoid';
export type ProfitDeskLane = 'edge' | 'value';
export type ProfitDeskPriceQuality =
  | 'verified_two_sided'
  | 'verified_no_vig'
  | 'one_sided'
  | 'assumed'
  | 'missing'
  | 'stale';

export interface ProfitDeskPrice {
  quality?: ProfitDeskPriceQuality | string;
  source?: string;
  updatedAt?: string | null;
  ageHours?: number | null;
  fresh?: boolean;
  twoSided?: boolean;
  noVigProbability?: number | null;
  breakEvenProbability?: number | null;
}

export interface ProfitDeskEstimate {
  marketProbability?: number | null;
  alpha?: number | null;
  alphaStdError?: number | null;
  probability?: number | null;
  lowerProbability?: number | null;
  expectedValue?: number | null;
  conservativeExpectedValue?: number | null;
  probabilityPositiveEv?: number | null;
  value?: ProfitDeskEstimate;
}

export interface ProfitDeskEvidence {
  sourceSamples?: number;
  segmentSamples?: number;
  distinctDates?: number;
  sourceDistinctDates?: number;
  wins?: number;
  losses?: number;
  sourceWins?: number;
  sourceLosses?: number;
  flatNetUnits?: number | null;
  flatRoi?: number | null;
  sourceFlatNetUnits?: number | null;
  sourceFlatRoi?: number | null;
  priorOnly?: boolean;
}

export interface ProfitDeskBlocker {
  code?: string;
  label?: string;
  detail?: string;
}

export interface ProfitDeskCandidate {
  id?: string;
  mode?: PickMode;
  tier?: ProfitDeskTier | string;
  portfolioSelected?: boolean;
  rank?: number | null;
  source?: string;
  sport?: string;
  pick?: string;
  game?: string;
  player?: string;
  market?: string;
  startTime?: string | null;
  result?: PickResult | string;
  decision?: string;
  oddsAmerican?: number | null;
  stakeUnits?: number | null;
  lane?: ProfitDeskLane | string | null;
  liveQualified?: boolean;
  closing?: ProfitDeskClosing;
  edgeQualified?: boolean;
  valueQualified?: boolean;
  price?: ProfitDeskPrice;
  estimate?: ProfitDeskEstimate;
  evidence?: ProfitDeskEvidence;
  blockers?: Array<ProfitDeskBlocker | string>;
  laneBlockers?: Partial<Record<'structural' | 'edge' | 'value', string[]>>;
  consensusSources?: string[];
}

export interface ProfitDeskModeSummary {
  candidateCount?: number;
  candidatesEvaluated?: number;
  observedPriceCandidates?: number;
  shadowQualified?: number;
  researchQualified?: number;
  edgeQualified?: number;
  valueQualified?: number;
  watchlist?: number;
  selected?: number;
  portfolioCandidates?: number;
  liveQualified?: number;
  evidenceRows?: number;
}

export interface ProfitDeskLiveRecord {
  wins?: number;
  losses?: number;
  pushes?: number;
  pending?: number;
  settled?: number;
  netUnits?: number | null;
  roi?: number | null;
  clvCount?: number;
  avgClv?: number | null;
}

export interface ProfitDeskClosing {
  oddsAmerican?: number | null;
  decimalOdds?: number | null;
  noVigProbability?: number | null;
  capturedAt?: string | null;
  provider?: string | null;
  clv?: number | null;
}

export interface ProfitDeskSummary extends ProfitDeskModeSummary {
  liveRecord?: ProfitDeskLiveRecord;
  liveRecordToDate?: ProfitDeskLiveRecord;
  modes?: Partial<Record<PickMode, ProfitDeskModeSummary>>;
}

export interface ProfitDeskPolicy {
  status?: string;
  mode?: string;
  firstLiveDate?: string | null;
  gates?: Record<string, unknown>;
  notes?: string[];
}

export interface ProfitDeskGateProgress {
  required?: number | boolean | null;
  actual?: number | boolean | null;
  passed?: boolean;
}

export interface ProfitDeskSourceCard {
  mode?: PickMode | string;
  sourceKey?: string;
  sport?: string;
  source?: string;
  samples?: number;
  distinctDates?: number;
  wins?: number;
  losses?: number;
  flatNetUnits?: number | null;
  flatRoi?: number | null;
  alpha?: number | null;
  probabilityPositiveEv?: number | null;
  gates?: Record<string, ProfitDeskGateProgress>;
  gatesPassed?: number;
  gatesTotal?: number;
  evidenceQualified?: boolean;
  candidatesToday?: number;
  liveToday?: number;
}

export interface ProfitDeskPayload {
  schemaVersion?: string | number;
  date: string;
  generatedAt?: string;
  engineVersion?: string;
  phase?: string;
  policy?: ProfitDeskPolicy;
  summary?: ProfitDeskSummary;
  candidates?: ProfitDeskCandidate[];
  portfolio?: Partial<Record<PickMode | 'all', ProfitDeskCandidate[]>>;
  sources?: ProfitDeskSourceCard[];
  [key: string]: unknown;
}

const RESULT_STORAGE_KEY = 'pickledger_static_results_v2';
const GAME_TIME_STORAGE_KEY = 'pickledger_static_game_times_v2';
// NBA Summer League and the FIFA World Cup archived 2026-07-19: both
// seasons ended (summer league finale + World Cup final same day).
const ARCHIVED_SPORTS = new Set(['NBA', 'NBA SUMMER', 'FIFA WC']);
// Shadow-mode sports: their picks are graded and ledger-tracked
// server-side but render nowhere on the site until an explicit go-live.
const SHADOW_SPORTS = new Set(['NFL']);
const PLAYER_PROPS_ML_SOURCE = 'player_props_ml_v1';
// First snapshot produced by the ML slate-engine launch in commit b6f9dbe.
const PLAYER_PROPS_ML_FIRST_SNAPSHOT_AT = Date.parse('2026-06-16T19:04:34.909830Z');
const PLAYER_PROPS_PUBLIC_START_DATE = '2026-06-23';
const SOURCE_LABELS: Record<string, string> = {
  mlb_new: 'MLB Model',
  mlb_inning: 'MLB Inning',
  mlb_first_five: 'MLB First Five',
  mlb_team_total: 'MLB Team Total',
  mls: 'MLS Model',
  nfl: 'NFL Model',
  wnba: 'WNBA Model',
  nba: 'NBA New',
  nba_playoffs: 'NBA Playoffs',
  nba_summer: 'NBA Summer League',
  fifa_world_cup: 'FIFA Model',
  sportytrader: 'SportyTrader',
  sportytrader_nba: 'SportyTraderNBA',
  sportytrader_nba_summer: 'SportyTraderNBASummer',
  sportytrader_mlb: 'SportyTraderMLB',
  sportytrader_wnba: 'SportyTraderWNBA',
  sportytrader_fifa_world_cup: 'SportyTraderFIFAWorldCup',
  sportsgambler: 'SportsGambler',
  sportsgambler_nba: 'SportsGamblerNBA',
  sportsgambler_nba_summer: 'SportsGamblerNBASummer',
  sportsgambler_mlb: 'SportsGamblerMLB',
  sportsgambler_wnba: 'SportsGamblerWNBA',
  sportsgambler_fifa_world_cup: 'SportsGamblerFIFAWorldCup',
  scores24_nba_summer: 'Scores24NBASummer',
  scores24_wnba: 'Scores24WNBA',
  scores24_mlb: 'Scores24MLB',
  scores24_fifa_world_cup: 'Scores24FIFAWorldCup',
  forebet_mls: 'ForebetMLS',
  forebet_mlb: 'ForebetMLB',
  forebet_wnba: 'ForebetWNBA',
  // Covers buckets carry per-row source labels ("Covers · <Author>",
  // "Covers Computer MLB", …); these are fallbacks for rows missing one.
  covers_experts_mlb: 'Covers Expert',
  covers_experts_wnba: 'Covers Expert',
  covers_computer_mlb: 'Covers Computer MLB',
  covers_consensus_mlb: 'Covers Consensus MLB',
  covers_consensus_wnba: 'Covers Consensus WNBA',
  covers_props_mlb: 'Covers Props (BAT X)',
};

const SOURCE_ALIASES: Record<string, string> = {
  'MLB NEW': 'MLB Model',
  'MLB New': 'MLB Model',
  'FIFA WC In-House': 'FIFA Model',
};

// The in-house MLB publishers each cover multiple markets under one bucket.
// The board tracks each market as its own source so a moneyline record can
// never hide a bad totals record (or vice versa). Applied at load time, so
// the split is retroactive across every committed cache day — the legacy
// "MLB Model" history decomposes into its ML and Total components with the
// underlying algorithms untouched.
const MARKET_SOURCE_LABELS: Record<string, Record<string, string>> = {
  mlb_new: { h2h: 'MLB ML', moneyline: 'MLB ML', totals: 'MLB Total', total: 'MLB Total' },
  mlb_first_five: { f5_side: 'MLB F5', f5_total: 'MLB F5 Total' },
  // Early-June wnba rows predate market_type stamping and were all
  // moneylines, so the empty-market fallback belongs to WNBA ML.
  wnba: { h2h: 'WNBA ML', moneyline: 'WNBA ML', '': 'WNBA ML', spread: 'WNBA Spread', totals: 'WNBA Total', total: 'WNBA Total' },
  // Summer league only ever bet moneylines; relabeled for naming
  // consistency with the other per-market sources. No spread/total
  // variants were built — the league's season ends 2026-07-19.
  nba_summer: { h2h: 'NBA Summer ML', moneyline: 'NBA Summer ML', '': 'NBA Summer ML' },
  mls: { moneyline: 'MLS ML', total: 'MLS Total', totals: 'MLS Total', spread: 'MLS Spread' },
  nfl: { h2h: 'NFL ML', moneyline: 'NFL ML', totals: 'NFL Total', total: 'NFL Total', spread: 'NFL Spread' },
};

function teamSourceLabel(modelKey: string, raw: Record<string, unknown>): string {
  const base = SOURCE_LABELS[modelKey] || modelKey;
  const byMarket = MARKET_SOURCE_LABELS[modelKey];
  if (!byMarket) return base;
  const market = String(raw.market || raw.market_type || '').trim().toLowerCase();
  return byMarket[market] ?? base;
}

const PLAYER_PROP_SOURCE_LABELS: Record<string, string> = {
  nba_player_props: 'NBAPlayerProps',
  mlb_player_props: 'MLBPlayerProps',
  wnba_player_props: 'WNBAPlayerProps',
  wnba_3pm: 'WNBA3PM',
  mlb_player_props_season: 'MLB Season Props',
  mlb_player_props_all_time: 'MLB All Time Props',
  mlb_player_props_hot_l10: 'MLB Hot L10 Props',
  mlb_player_props_matchup_h2h: 'MLB Matchup H2H Props',
  wnba_player_props_season: 'WNBA Season Props',
  wnba_player_props_all_time: 'WNBA All Time Props',
  wnba_player_props_hot_l10: 'WNBA Hot L10 Props',
  wnba_player_props_matchup_h2h: 'WNBA Matchup H2H Props',
};

let activePickMode: PickMode = 'team';
let teamPicks: Pick[] = [];
let playerPicks: Pick[] = [];
let resultOverrides: Record<string, PickResult> = {};
let gameTimes: Record<string, string> = {};
let latestTeamCache: ModelCachePayload | null = null;
let latestPlayerCache: PlayerPropsPayload | null = null;
let parlayPayloads: ParlayCardsPayload[] = [];
let latestParlayPayload: ParlayCardsPayload | null = null;
let profitDeskPayloads: ProfitDeskPayload[] = [];
let latestProfitDeskPayload: ProfitDeskPayload | null = null;

function readStorage<T>(key: string, fallback: T): T {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || '');
    return parsed && typeof parsed === 'object' ? parsed as T : fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // The viewer remains usable when storage is blocked.
  }
}

function normalizeResult(value: unknown): PickResult {
  const result = String(value || '').trim().toLowerCase();
  if (result === 'win' || result === 'w') return 'win';
  if (result === 'loss' || result === 'l') return 'loss';
  if (result === 'push' || result === 'p') return 'push';
  return 'pending';
}

function numberOrNull(value: unknown): number | null {
  if (value === '' || value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function stableHash(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36);
}

function stablePickId(raw: Record<string, unknown>, date: string, source: string): string {
  const existing = String(raw.id || '').trim();
  if (existing) return existing;
  return `pick-${stableHash(JSON.stringify([
    source,
    raw.sport,
    date,
    raw.pick,
    raw.selection || raw.prop || raw.bet,
    raw.player || raw.player_name,
    raw.market || raw.market_type,
    raw.ml_rank_epoch || raw.ranking_epoch || raw.model_epoch,
    raw.matchup || raw.game,
    raw.away_team,
    raw.home_team,
  ]))}`;
}

export function calculateProfit(pick: Pick, result: PickResult = pick.result): number {
  if (result === 'pending' || result === 'push') return 0;
  if (pick.price_verified !== true) return 0;
  const odds = numberOrNull(pick.odds);
  if (odds == null || odds === 0) return 0;
  if (result === 'loss') return -pick.units;
  return Number((odds > 0 ? pick.units * odds / 100 : pick.units * 100 / Math.abs(odds)).toFixed(2));
}

function normalizedPriceProvenance(raw: Record<string, unknown>, odds: number | null): {
  verified: boolean;
  provenance: Pick['price_provenance'];
} {
  if (odds == null || odds === 0) return { verified: false, provenance: 'missing' };
  const quality = String(raw.price_quality || raw.odds_quality || raw.market_price_quality || '').trim().toLowerCase();
  const verifiedQualities = new Set(['verified', 'verified_two_sided', 'verified_no_vig', 'observed_sportsbook']);
  const explicitlyVerified = raw.price_verified === true || raw.odds_verified === true || verifiedQualities.has(quality);
  if (explicitlyVerified) return { verified: true, provenance: 'verified' };
  const provenanceText = [
    raw.pricing_type,
    raw.price_source,
    raw.odds_source,
    raw.line_source,
    raw.market_source,
  ].map(value => String(value || '').trim().toLowerCase()).filter(Boolean).join(' ');
  const nonExecutable = /assumed|synthetic|proxy|fallback|default|estimated|model[_ ]price/.test(provenanceText);
  const usesAssumedPrice = raw.odds == null && raw.american_odds == null && raw.price == null && raw.assumed_odds != null;
  if (usesAssumedPrice || quality === 'assumed' || nonExecutable) return { verified: false, provenance: 'assumed' };
  const observedMarker = /posted|sportsbook|bookmaker|observed|executable/.test(provenanceText);
  if (raw.market_priced === true && observedMarker) return { verified: true, provenance: 'verified' };
  return { verified: false, provenance: 'unverified' };
}

function normalizePick(
  input: unknown,
  fallbackDate: string,
  fallbackSource: string,
  gameByMatchup: Map<string, Record<string, unknown>> = new Map(),
  playerProp = false,
): Pick | null {
  if (!input || typeof input !== 'object') return null;
  const raw = input as Record<string, unknown>;
  const pickText = String(raw.pick || raw.selection || raw.prop || raw.bet || '').trim();
  if (!pickText) return null;

  const rawSource = String((playerProp && fallbackSource) ? fallbackSource : (raw.source || fallbackSource || 'Unknown')).trim();
  const source = SOURCE_ALIASES[rawSource] || rawSource;
  const date = String(raw.date || raw.game_date || raw.slate_date || raw.Date || fallbackDate || '').trim();
  const matchup = String(raw.matchup || raw.game || raw.event || '').trim();
  const game = gameByMatchup.get(matchup);
  const id = stablePickId(raw, date, rawSource);
  const embeddedResult = normalizeResult(raw.result);
  const localResult = normalizeResult(resultOverrides[id]);
  const result = embeddedResult === 'pending' ? localResult : embeddedResult;
  const decision = String(raw.decision || '').trim().toUpperCase();
  const units = numberOrNull(raw.units ?? raw.stake_units ?? raw.recommended_units ?? raw.quarter_kelly)
    ?? (playerProp && decision === 'PASS' ? 0 : 1);
  const startTime = String(
    raw.start_time || raw.game_start_time ||
    game?.start_time || game?.game_start_time ||
    gameTimes[id] || '',
  ).trim() || null;
  const odds = numberOrNull(raw.odds ?? raw.assumed_odds ?? raw.american_odds ?? raw.price);
  const priceProvenance = normalizedPriceProvenance(raw, odds);

  const pick: Pick = {
    ...raw,
    id,
    source,
    pick: pickText,
    sport: String(raw.sport || raw.league || 'OTHER').trim().toUpperCase(),
    matchup: matchup || undefined,
    player: String(raw.player || raw.player_name || '').trim() || undefined,
    reason: String(raw.reason || raw.rationale || raw.notes || '').trim() || undefined,
    key_factors: raw.key_factors ?? raw.factors ?? raw.guardrail_reasons,
    date,
    units,
    odds,
    price_verified: priceProvenance.verified,
    price_provenance: priceProvenance.provenance,
    probability: numberOrNull(raw.probability ?? raw.model_probability ?? raw.predicted_probability),
    result,
    pl: 0,
    start_time: startTime,
    game_start_time: startTime,
  };
  pick.pl = calculateProfit(pick, result);
  return pick;
}

function isTrackedPick(pick: Pick): boolean {
  const decision = String(pick.decision || '').trim().toUpperCase();
  return decision === 'BET' || decision === 'LEAN';
}

function isTrackedPlayerProp(pick: Pick): boolean {
  const decision = String(pick.decision || '').trim().toUpperCase();
  return decision === 'BET' || decision === 'LEAN' || decision === 'PASS';
}

function isPlayerScopedPick(pick: Pick): boolean {
  return String(pick.scope || '').trim().toLowerCase() === 'player';
}

function isMlEraPlayerProp(pick: Pick): boolean {
  if (String(pick.probability_source || '').trim() !== PLAYER_PROPS_ML_SOURCE) return false;
  if (String(pick.date || '') < PLAYER_PROPS_PUBLIC_START_DATE) return false;
  const timestamp = Date.parse(String(
    pick.ranking_updated_at || pick.generated_at || pick.created_at || '',
  ));
  return Number.isFinite(timestamp) && timestamp >= PLAYER_PROPS_ML_FIRST_SNAPSHOT_AT;
}

function picksFromCache(payload: ModelCachePayload): Pick[] {
  const date = String(payload.date || '').trim();
  const models = payload.models && typeof payload.models === 'object' ? payload.models : {};
  const picks: Pick[] = [];

  for (const [modelKey, bucket] of Object.entries(models)) {
    if (!bucket || typeof bucket !== 'object' || bucket.ok === false) continue;
    const gameByMatchup = new Map<string, Record<string, unknown>>();
    if (Array.isArray(bucket.games)) {
      for (const item of bucket.games) {
        if (!item || typeof item !== 'object') continue;
        const game = item as Record<string, unknown>;
        const matchup = String(game.matchup || game.game || '').trim();
        if (matchup) gameByMatchup.set(matchup, game);
      }
    }
    for (const raw of Array.isArray(bucket.picks) ? bucket.picks : []) {
      if (!raw || typeof raw !== 'object') continue;
      const rawRecord = raw as Record<string, unknown>;
      const source = teamSourceLabel(modelKey, rawRecord);
      // Committed rows carry their own legacy source label ("MLB Model"),
      // which normalizePick would prefer — override it so the per-market
      // split actually lands.
      const input = MARKET_SOURCE_LABELS[modelKey] ? { ...rawRecord, source } : rawRecord;
      const pick = normalizePick(input, date, source, gameByMatchup);
      if (pick && isTrackedPick(pick)) picks.push(pick);
    }
  }
  return picks;
}

function playerPropRecords(payload: PlayerPropsPayload): Array<{ raw: unknown; source: string }> {
  const records: Array<{ raw: unknown; source: string }> = [];
  const addBucket = (bucket: unknown, source: string): void => {
    if (Array.isArray(bucket)) {
      bucket.forEach(raw => records.push({ raw, source }));
      return;
    }
    if (!bucket || typeof bucket !== 'object') return;
    const value = bucket as Record<string, unknown>;
    if (value.ok === false) return;
    for (const key of ['picks', 'props', 'player_props', 'recommendations']) {
      if (Array.isArray(value[key])) addBucket(value[key], source);
    }
  };

  addBucket(payload, 'Player Props');
  for (const containerKey of ['models', 'sports', 'leagues']) {
    const container = payload[containerKey];
    if (!container || typeof container !== 'object' || Array.isArray(container)) continue;
    for (const [source, bucket] of Object.entries(container as Record<string, unknown>)) {
      addBucket(bucket, PLAYER_PROP_SOURCE_LABELS[source] || source);
    }
  }
  return records;
}

function picksFromPlayerProps(payload: PlayerPropsPayload): Pick[] {
  const date = String(payload.date || payload.slate_date || '').trim();
  return playerPropRecords(payload)
    .map(({ raw, source }) => normalizePick(raw, date, source, new Map(), true))
    .filter((pick): pick is Pick => Boolean(pick) && isTrackedPlayerProp(pick));
}

async function fetchJson<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(`${path}?v=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) return null;
    return await response.json() as T;
  } catch {
    return null;
  }
}

async function loadCacheFiles(): Promise<ModelCachePayload[]> {
  const manifest = await fetchJson<CacheManifest>('./data/model_cache/index.json');
  const files = Array.isArray(manifest?.files)
    ? manifest.files.filter(file => /^\d{4}-\d{2}-\d{2}\.json$/.test(file))
    : [];
  if (!files.length) {
    const fallback = await fetchJson<ModelCachePayload>('./data/model_cache/latest.json');
    latestTeamCache = fallback;
    return fallback ? [fallback] : [];
  }

  const payloads = (await Promise.all(
    files.map(file => fetchJson<ModelCachePayload>(`./data/model_cache/${file}`)),
  )).filter((payload): payload is ModelCachePayload => Boolean(payload));
  payloads.sort((a, b) => String(a.date || '').localeCompare(String(b.date || '')));
  latestTeamCache = payloads[payloads.length - 1] || null;
  return payloads;
}

async function loadPlayerCacheFiles(): Promise<PlayerPropsPayload[]> {
  const manifest = await fetchJson<CacheManifest>('./data/player_props_cache/index.json');
  const files = Array.isArray(manifest?.files)
    ? manifest.files.filter(file => /^\d{4}-\d{2}-\d{2}\.json$/.test(file))
    : [];
  if (!files.length) {
    const fallback = await fetchJson<PlayerPropsPayload>('./data/player_props_cache/latest.json');
    latestPlayerCache = fallback;
    return fallback ? [fallback] : [];
  }

  const payloads = (await Promise.all(
    files.map(file => fetchJson<PlayerPropsPayload>(`./data/player_props_cache/${file}`)),
  )).filter((payload): payload is PlayerPropsPayload => Boolean(payload));
  payloads.sort((a, b) => String(a.date || a.slate_date || '').localeCompare(String(b.date || b.slate_date || '')));
  latestPlayerCache = payloads[payloads.length - 1] || null;
  return payloads;
}

async function loadParlayCardFiles(): Promise<ParlayCardsPayload[]> {
  const manifest = await fetchJson<CacheManifest>('./data/parlay_cards/index.json');
  const files = Array.isArray(manifest?.files)
    ? manifest.files.filter(file => /^\d{4}-\d{2}-\d{2}\.json$/.test(file))
    : [];
  if (!files.length) {
    const fallback = await fetchJson<ParlayCardsPayload>('./data/parlay_cards/latest.json');
    latestParlayPayload = fallback;
    parlayPayloads = fallback ? [fallback] : [];
    return parlayPayloads;
  }

  const payloads = (await Promise.all(
    files.map(file => fetchJson<ParlayCardsPayload>(`./data/parlay_cards/${file}`)),
  )).filter((payload): payload is ParlayCardsPayload => Boolean(payload));
  payloads.sort((a, b) => String(a.date || '').localeCompare(String(b.date || '')));
  latestParlayPayload = payloads[payloads.length - 1] || null;
  parlayPayloads = payloads;
  return payloads;
}

async function loadProfitDeskFiles(): Promise<ProfitDeskPayload[]> {
  const manifest = await fetchJson<CacheManifest>('./data/profit_desk/index.json');
  const files = Array.isArray(manifest?.files)
    ? manifest.files.filter(file => /^\d{4}-\d{2}-\d{2}\.json$/.test(file))
    : [];
  if (!files.length) {
    const fallback = await fetchJson<ProfitDeskPayload>('./data/profit_desk/latest.json');
    profitDeskPayloads = fallback?.date ? [fallback] : [];
    latestProfitDeskPayload = profitDeskPayloads[0] || null;
    return profitDeskPayloads;
  }

  const payloads = (await Promise.all(
    files.map(file => fetchJson<ProfitDeskPayload>(`./data/profit_desk/${file}`)),
  )).filter((payload): payload is ProfitDeskPayload => Boolean(payload?.date));
  payloads.sort((a, b) => String(a.date || '').localeCompare(String(b.date || '')));
  profitDeskPayloads = payloads;
  latestProfitDeskPayload = payloads[payloads.length - 1] || null;
  return payloads;
}

function sortPicks(picks: Pick[]): Pick[] {
  return picks.sort((a, b) => (
    a.date.localeCompare(b.date) ||
    a.sport.localeCompare(b.sport) ||
    a.source.localeCompare(b.source) ||
    a.pick.localeCompare(b.pick)
  ));
}

export function setPickMode(mode: PickMode): void {
  activePickMode = mode;
}

export function getPickMode(): PickMode {
  return activePickMode;
}

export async function loadAllData(): Promise<Pick[]> {
  resultOverrides = readStorage<Record<string, PickResult>>(RESULT_STORAGE_KEY, {});
  gameTimes = readStorage<Record<string, string>>(GAME_TIME_STORAGE_KEY, {});
  const [cachePayloads, playerPayloads] = await Promise.all([
    loadCacheFiles(),
    loadPlayerCacheFiles(),
    loadParlayCardFiles(),
    loadProfitDeskFiles(),
  ]);
  const teamById = new Map<string, Pick>();
  const playerById = new Map<string, Pick>();
  cachePayloads.flatMap(picksFromCache).forEach(pick => {
    if (isPlayerScopedPick(pick)) playerById.set(pick.id, pick);
    else teamById.set(pick.id, pick);
  });
  playerPayloads.flatMap(picksFromPlayerProps).forEach(pick => playerById.set(pick.id, pick));
  teamPicks = sortPicks([...teamById.values()].filter(pick => !ARCHIVED_SPORTS.has(pick.sport) && !SHADOW_SPORTS.has(pick.sport)));
  // External player-prop feeds (scope=player rows in the team cache, e.g.
  // Covers) render in Player mode alongside the in-house ML-era props; the
  // scope routing above already keeps them out of Team mode and rankings.
  playerPicks = sortPicks([...playerById.values()].filter(
    pick => !ARCHIVED_SPORTS.has(pick.sport) && (isMlEraPlayerProp(pick) || pick.external_player_feed === true),
  ));
  return getAllPicks();
}

export function getAllPicks(): Pick[] {
  return activePickMode === 'player' ? playerPicks : teamPicks;
}

export function getParlayCardsPayload(date?: string): ParlayCardsPayload | null {
  if (date) {
    return parlayPayloads.find(payload => payload.date === date) || null;
  }
  return latestParlayPayload;
}

export function getParlayCardPayloads(): ParlayCardsPayload[] {
  return parlayPayloads;
}

export function getProfitDeskPayload(date?: string): ProfitDeskPayload | null {
  if (date) {
    return profitDeskPayloads.find(payload => payload.date === date) || null;
  }
  return latestProfitDeskPayload;
}

export function getProfitDeskPayloads(): ProfitDeskPayload[] {
  return profitDeskPayloads;
}

export function getResults(): Record<string, PickResult> {
  return resultOverrides;
}

export function setLocalResult(id: string, result: PickResult): void {
  resultOverrides[id] = result;
  writeStorage(RESULT_STORAGE_KEY, resultOverrides);
  const pick = getAllPicks().find(item => item.id === id);
  if (pick) {
    pick.result = result;
    pick.pl = calculateProfit(pick, result);
  }
}

export function setLocalGameTime(id: string, startTime: string): void {
  gameTimes[id] = startTime;
  writeStorage(GAME_TIME_STORAGE_KEY, gameTimes);
  const pick = getAllPicks().find(item => item.id === id);
  if (pick) {
    pick.start_time = startTime;
    pick.game_start_time = startTime;
  }
}

function latestPayloadTimestamp(value: unknown): number {
  if (!value || typeof value !== 'object') return 0;
  let latest = 0;
  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    if ((key === 'generatedAt' || key === 'updatedAt') && typeof nested === 'string') {
      const timestamp = new Date(nested).getTime();
      if (Number.isFinite(timestamp)) latest = Math.max(latest, timestamp);
    } else if (nested && typeof nested === 'object') {
      latest = Math.max(latest, latestPayloadTimestamp(nested));
    }
  }
  return latest;
}

export function getCacheStatus(): { date: string; runTime: string; updatedAt: string; pickCount: number } {
  const latestCache = activePickMode === 'player' ? latestPlayerCache : latestTeamCache;
  const latestTimestamp = latestPayloadTimestamp(latestCache);
  const parsed = new Date(latestTimestamp);
  return {
    date: String(latestCache?.date || latestCache?.slate_date || ''),
    runTime: !latestTimestamp || Number.isNaN(parsed.getTime())
      ? ''
      : parsed.toLocaleTimeString('en-US', {
        timeZone: 'America/Chicago',
        hour: 'numeric',
        minute: '2-digit',
        timeZoneName: 'short',
      }),
    updatedAt: latestTimestamp ? parsed.toISOString() : '',
    pickCount: getAllPicks().length,
  };
}
