import type { Pick } from './data';

export type RankingDecisionFilter = 'STAKED' | 'BET' | 'LEAN' | 'PASS';

export type RankingScope = {
  sports: ReadonlySet<string>;
  sources: ReadonlySet<string>;
  decision: RankingDecisionFilter;
};

export const PLAYER_PROP_RANKING_START_DATE = '2026-06-23';
export const MLB_TEAM_CONSENSUS_EPOCH_PREFIX = 'MLB:mlb_team_consensus_v1';
export const MLB_TEAM_CONSENSUS_SOURCES = new Set([
  'MLB Model', 'MLB ML', 'MLB Total',
  'MLB First Five', 'MLB F5', 'MLB F5 Total',
  'MLB Inning', 'MLB Team Total',
]);
// The 2026-07-19 board rebuild: rankings restart from this date so stale
// records don't carry into the redesigned source split. The proven MLB
// moneyline/total split (formerly "MLB Model") keeps its full
// consensus-era history — that record is the one worth preserving.
export const TEAM_RANKING_START_DATE = '2026-07-19';
// "MLB Model" stays listed as a safety net: any mlb_new row whose market
// tag fails the ML/Total split must never drop out of the record.
export const LEGACY_RECORD_SOURCES = new Set(['MLB ML', 'MLB Total', 'MLB Model']);
// WNBA redesign (2026-07-19): the proven moneyline record carries over as
// WNBA ML; the rebuilt spread/total variants (and any stray legacy label)
// restart from the redesign date.
export const WNBA_RESET_SOURCES = new Set(['WNBA Model', 'WNBA Spread', 'WNBA Total']);
// WNBA totals v2 (2026-07-26): the totals projection was retrained after
// the v1 run finished 24-44 (93% Unders — the model carried a systematic
// low-total bias). The WNBA Total record and rankings restart at the v2
// cutover; ML and Spread keep their 2026-07-19 reset above.
export const WNBA_TOTAL_RANKING_START_DATE = '2026-07-26';
// MLB Inning v2 (2026-08-25): inning-shaped pitching prior + shrunk
// 30-game team history replaced the flat ERA conversion that published
// noisy early innings and starved the bullpen path. The MLB Inning
// record restarts at the cutover; other MLB team variants keep 7/19.
export const MLB_INNING_RANKING_START_DATE = '2026-08-25';
// MLS v2 (2026-07-25): the trained Dixon-Coles engine replaced the
// FIFA-derived heuristic wholesale, so the tracked record restarts at the
// cutover. The legacy 'MLS Model' label is included so a row that misses
// the per-market split can never carry the old engine's record forward.
export const MLS_RESET_SOURCES = new Set(['MLS Model', 'MLS ML', 'MLS Spread', 'MLS Total']);
export const MLS_RANKING_START_DATE = '2026-07-25';

const DISPLAY_TIME_ZONE = 'America/Chicago';

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

export function rankingPickDateKey(pick: Pick): string {
  const raw = String(pick.date || '').trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? '' : centralDateKey(parsed);
}

export function rankingSourceName(pick: Pick): string {
  return String(pick.source || 'Unknown').trim();
}

export function rankingSportName(pick: Pick): string {
  return String(pick.sport || '').trim() || 'Other';
}

export function rankingDecisionOf(pick: Pick): string {
  return String(pick.decision || 'WATCH').trim().toUpperCase();
}

export function rankingDecisionMatches(pick: Pick, filter: RankingDecisionFilter): boolean {
  const decision = rankingDecisionOf(pick);
  if (filter === 'STAKED') return decision === 'BET' || decision === 'LEAN';
  return decision === filter;
}

/** Best Bets Research: high-probability sit-outs plus juice favorites. */
export function isDailyResearchCandidate(pick: Pick, probability: number | null): boolean {
  const decision = rankingDecisionOf(pick);
  const unpublished = decision !== 'BET' && decision !== 'LEAN';
  const highProbPass = unpublished && probability != null && probability >= 0.6;
  const priceyFavorite = pick.odds != null && pick.odds <= -300;
  return highProbPass || priceyFavorite;
}

/** Replay Research from the same posted BET/LEAN/PASS universe the live board uses. */
export function dailyResearchPool(
  picks: Pick[],
  probabilityOf: (pick: Pick) => number | null,
): Pick[] {
  return picks.filter(pick => {
    const decision = rankingDecisionOf(pick);
    if (decision !== 'BET' && decision !== 'LEAN' && decision !== 'PASS') return false;
    return isDailyResearchCandidate(pick, probabilityOf(pick));
  });
}

export function defaultRankingBucketNames(pick: Pick): string[] {
  return [rankingSourceName(pick)];
}

/** Keep every football market discoverable, even when it has only PASS forecasts. */
export function footballModelRecords(picks: Pick[]): { source: string; staked: Pick[]; passes: Pick[] }[] {
  return ['NFL', 'CFB'].flatMap(sport => ['ML', 'Spread', 'Total'].map(market => {
    const source = `${sport} ${market}`;
    const rows = picks.filter(pick => pick.sport === sport && rankingSourceName(pick) === source && pick.scraped !== true);
    return {
      source,
      staked: rows.filter(pick => rankingDecisionMatches(pick, 'STAKED')),
      passes: rows.filter(pick => rankingDecisionMatches(pick, 'PASS')),
    };
  }));
}

export function isTeamRankingWindowPick(pick: Pick): boolean {
  const source = rankingSourceName(pick);
  const isConsensusSource = String(pick.sport || '').toUpperCase() === 'MLB'
    && MLB_TEAM_CONSENSUS_SOURCES.has(source);
  // Record resets apply ONLY to in-house model variants (MLB/WNBA at
  // the 2026-07-19 redesign, MLS at the 2026-07-25 v2 cutover, WNBA
  // totals again at the 2026-07-26 v2 retrain). External feeds and
  // every other source keep their full history — including CFB/NFL,
  // which were never part of those cutovers.
  if (source === 'WNBA Total') {
    return rankingPickDateKey(pick) >= WNBA_TOTAL_RANKING_START_DATE;
  }
  if (source === 'MLB Inning') {
    return rankingPickDateKey(pick) >= MLB_INNING_RANKING_START_DATE;
  }
  if (WNBA_RESET_SOURCES.has(source)) {
    return rankingPickDateKey(pick) >= TEAM_RANKING_START_DATE;
  }
  if (MLS_RESET_SOURCES.has(source)) {
    return rankingPickDateKey(pick) >= MLS_RANKING_START_DATE;
  }
  if (!isConsensusSource) return true;
  // Legacy sources (MLB ML, MLB Total, MLB Model) keep their full
  // consensus-era history but only count picks that carry the v1 epoch
  // stamp — this filters out any pre-consensus rows from their record.
  if (LEGACY_RECORD_SOURCES.has(source)) {
    const epoch = String(pick.ml_rank_epoch || pick.ranking_epoch || pick.model_epoch || '').trim();
    return epoch.startsWith(MLB_TEAM_CONSENSUS_EPOCH_PREFIX);
  }
  // Non-legacy consensus sources (Team Total, F5, etc.) are date-gated
  // from the 2026-07-19 redesign. MLB Inning is gated earlier at v2.
  return rankingPickDateKey(pick) >= TEAM_RANKING_START_DATE;
}

export function rankingComparableTeamPicks(picks: Pick[]): Pick[] {
  return picks.filter(isTeamRankingWindowPick);
}

export function rankingScopedPicks(
  picks: Pick[],
  scope: RankingScope,
  bucketNames: (pick: Pick) => readonly string[] = defaultRankingBucketNames,
): Pick[] {
  return picks.filter(pick => {
    if (scope.sports.size && !scope.sports.has(rankingSportName(pick))) return false;
    if (scope.sources.size && !bucketNames(pick).some(name => scope.sources.has(name))) return false;
    return rankingDecisionMatches(pick, scope.decision);
  });
}

/** Overall Stats on Rankings always match the currently scoped ranking pool. */
export function rankingOverallPool(
  picks: Pick[],
  scope: RankingScope,
  bucketNames?: (pick: Pick) => readonly string[],
): Pick[] {
  return rankingScopedPicks(rankingComparableTeamPicks(picks), scope, bucketNames);
}
