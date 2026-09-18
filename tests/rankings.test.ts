import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { Pick } from '../src/data.ts';
import {
  TEAM_RANKING_START_DATE,
  dailyResearchPool,
  footballModelRecords,
  isDailyResearchCandidate,
  isTeamRankingWindowPick,
  rankingComparableTeamPicks,
  rankingOverallPool,
  rankingScopedPicks,
  type RankingScope,
} from '../src/rankings.ts';

function pick(overrides: Partial<Pick> & { id: string }): Pick {
  return {
    source: 'CFB ML',
    pick: 'Home State ML',
    sport: 'CFB',
    date: '2026-09-12',
    units: 0.5,
    odds: -110,
    result: 'pending',
    pl: 0,
    decision: 'BET',
    price_verified: true,
    ...overrides,
  };
}

function record(picks: Pick[]): { wins: number; losses: number; pending: number; net: number } {
  return {
    wins: picks.filter(item => item.result === 'win').length,
    losses: picks.filter(item => item.result === 'loss').length,
    pending: picks.filter(item => item.result === 'pending').length,
    net: Number(picks.reduce((sum, item) => sum + item.pl, 0).toFixed(2)),
  };
}

function scope(partial: Partial<RankingScope> = {}): RankingScope {
  return {
    sports: new Set<string>(),
    sources: new Set<string>(),
    decision: 'STAKED',
    ...partial,
  };
}

test('football summary retains PASS-only and empty models without mixing their bet records', () => {
  const win = pick({ id: 'cfb-win', result: 'win' });
  const pass = pick({ id: 'cfb-pass', source: 'CFB Spread', decision: 'PASS', units: 0, result: 'win' });
  const lean = pick({ id: 'nfl-lean', sport: 'NFL', source: 'NFL Total', decision: 'LEAN', result: 'loss' });
  const rows = footballModelRecords([
    win, pass, lean,
    pick({ id: 'external', source: 'SportyTraderCFB', result: 'win' }),
    pick({ id: 'scraped', scraped: true, result: 'win' }),
  ]);
  assert.deepEqual(rows.map(row => row.source), ['NFL ML', 'NFL Spread', 'NFL Total', 'CFB ML', 'CFB Spread', 'CFB Total']);
  assert.deepEqual(rows.find(row => row.source === 'CFB ML')?.staked, [win]);
  assert.deepEqual(rows.find(row => row.source === 'NFL Total')?.staked, [lean]);
  assert.deepEqual(rows.find(row => row.source === 'CFB Spread'), { source: 'CFB Spread', staked: [], passes: [pass] });
  assert.deepEqual(rows.find(row => row.source === 'CFB Total'), { source: 'CFB Total', staked: [], passes: [] });
});

const mlbWin = pick({
  id: 'mlb-win',
  sport: 'MLB',
  source: 'MLB ML',
  date: '2026-08-01',
  decision: 'BET',
  result: 'win',
  pl: 0.91,
  ml_rank_epoch: 'MLB:mlb_team_consensus_v1:keep',
});
const cfbBetWin = pick({
  id: 'cfb-bet-win',
  sport: 'CFB',
  source: 'CFB ML',
  decision: 'BET',
  result: 'win',
  units: 0.5,
  pl: 0.45,
});
const cfbLeanLoss = pick({
  id: 'cfb-lean-loss',
  sport: 'CFB',
  source: 'CFB Spread',
  pick: 'Home State -3.5',
  decision: 'LEAN',
  result: 'loss',
  units: 0.25,
  pl: -0.25,
});
const cfbPassWin = pick({
  id: 'cfb-pass-win',
  sport: 'CFB',
  source: 'CFB Total',
  pick: 'Under 54.5',
  decision: 'PASS',
  result: 'win',
  units: 0,
  pl: 0,
});
const nflLeanWin = pick({
  id: 'nfl-lean-win',
  sport: 'NFL',
  source: 'NFL ML',
  pick: 'Seahawks ML',
  decision: 'LEAN',
  result: 'win',
  units: 0.25,
  pl: 0.23,
});
const nflPassPending = pick({
  id: 'nfl-pass-pending',
  sport: 'NFL',
  source: 'NFL Spread',
  pick: 'Seahawks -3',
  decision: 'PASS',
  result: 'pending',
  units: 0,
  pl: 0,
});

const slate = [mlbWin, cfbBetWin, cfbLeanLoss, cfbPassWin, nflLeanWin, nflPassPending];

test('CFB and NFL settled rows stay in the team ranking window', () => {
  const earlyCfb = pick({
    id: 'cfb-early',
    date: '2026-07-01',
    result: 'win',
  });
  const earlyNfl = pick({
    id: 'nfl-early',
    sport: 'NFL',
    source: 'NFL ML',
    date: '2026-07-01',
    result: 'loss',
  });
  assert.ok(earlyCfb.date < TEAM_RANKING_START_DATE);
  assert.equal(isTeamRankingWindowPick(earlyCfb), true);
  assert.equal(isTeamRankingWindowPick(earlyNfl), true);
  assert.equal(isTeamRankingWindowPick(cfbPassWin), true);
  assert.deepEqual(
    rankingComparableTeamPicks(slate).map(item => item.id).sort(),
    slate.map(item => item.id).sort(),
  );
});

test('ranking sport and decision filters recompute the top-section pool', () => {
  const allStaked = rankingOverallPool(slate, scope());
  assert.deepEqual(allStaked.map(item => item.id).sort(), [
    'cfb-bet-win',
    'cfb-lean-loss',
    'mlb-win',
    'nfl-lean-win',
  ]);
  assert.deepEqual(record(allStaked), { wins: 3, losses: 1, pending: 0, net: 1.34 });

  const cfbStaked = rankingOverallPool(slate, scope({ sports: new Set(['CFB']) }));
  assert.deepEqual(cfbStaked.map(item => item.id).sort(), ['cfb-bet-win', 'cfb-lean-loss']);
  assert.deepEqual(record(cfbStaked), { wins: 1, losses: 1, pending: 0, net: 0.20 });

  const cfbPass = rankingOverallPool(slate, scope({
    sports: new Set(['CFB']),
    decision: 'PASS',
  }));
  assert.deepEqual(cfbPass.map(item => item.id), ['cfb-pass-win']);
  assert.deepEqual(record(cfbPass), { wins: 1, losses: 0, pending: 0, net: 0 });

  const nflMl = rankingOverallPool(slate, scope({
    sports: new Set(['NFL']),
    sources: new Set(['NFL ML']),
  }));
  assert.deepEqual(nflMl.map(item => item.id), ['nfl-lean-win']);
  assert.deepEqual(record(nflMl), { wins: 1, losses: 0, pending: 0, net: 0.23 });

  const nflPass = rankingOverallPool(slate, scope({
    sports: new Set(['NFL']),
    decision: 'PASS',
  }));
  assert.deepEqual(nflPass.map(item => item.id), ['nfl-pass-pending']);
  assert.deepEqual(record(nflPass), { wins: 0, losses: 0, pending: 1, net: 0 });
});

test('source filters intersect sports instead of unioning unrelated buckets', () => {
  const mixed = rankingScopedPicks(slate, scope({
    sports: new Set(['CFB']),
    sources: new Set(['MLB ML', 'CFB ML']),
  }));
  assert.deepEqual(mixed.map(item => item.id), ['cfb-bet-win']);
});

test('research decision filters mirror Rankings STAKED/BET/LEAN/PASS matching', () => {
  const betWin = pick({ id: 'r-bet', source: 'Scores24MLB', scraped: true, research: true, decision: 'BET', result: 'win' });
  const leanLoss = pick({ id: 'r-lean', source: 'Scores24MLB', scraped: true, research: true, decision: 'LEAN', result: 'loss' });
  const passPending = pick({ id: 'r-pass', source: 'ForebetMLB', scraped: true, research: true, decision: 'PASS', units: 0, result: 'pending' });
  const pool = [betWin, leanLoss, passPending];
  assert.deepEqual(
    rankingScopedPicks(pool, scope({ decision: 'STAKED' })).map(item => item.id).sort(),
    ['r-bet', 'r-lean'],
  );
  assert.deepEqual(rankingScopedPicks(pool, scope({ decision: 'BET' })).map(item => item.id), ['r-bet']);
  assert.deepEqual(rankingScopedPicks(pool, scope({ decision: 'LEAN' })).map(item => item.id), ['r-lean']);
  assert.deepEqual(rankingScopedPicks(pool, scope({ decision: 'PASS' })).map(item => item.id), ['r-pass']);
  // A filtered research pool with mixed results must not collapse to an all-win card.
  const staked = rankingScopedPicks(pool, scope({ decision: 'STAKED' }));
  assert.equal(record(staked).wins, 1);
  assert.equal(record(staked).losses, 1);
});

test('Best Bets Research historically counts PASS, not only juice favorites', () => {
  const passWin = pick({
    id: 'f5-pass-win',
    sport: 'MLB',
    source: 'MLB First Five',
    decision: 'PASS',
    units: 0,
    probability: 0.74,
    result: 'win',
    odds: -110,
  });
  const passLoss = pick({
    id: 'tt-pass-loss',
    sport: 'MLB',
    source: 'MLB Team Total',
    decision: 'PASS',
    units: 0,
    probability: 0.61,
    result: 'loss',
    odds: -105,
  });
  const juiceWin = pick({
    id: 'juice-bet-win',
    sport: 'MLB',
    source: 'MLB ML',
    decision: 'BET',
    probability: 0.72,
    result: 'win',
    odds: -350,
  });
  const passMid = pick({
    id: 'f5-pass-mid',
    sport: 'MLB',
    source: 'MLB First Five',
    decision: 'PASS',
    units: 0,
    probability: 0.55,
    result: 'win',
    odds: -110,
  });
  const publishedOnly = [juiceWin];
  const posted = [passWin, passLoss, passMid, juiceWin];
  const probabilityOf = (item: Pick): number | null => (
    item.probability == null ? null : Number(item.probability)
  );
  assert.equal(isDailyResearchCandidate(passWin, 0.74), true);
  // Mid-probability PASS reaches Research only as a slate probability leader,
  // never through the queue, so the queue cannot swell to every sit-out.
  assert.equal(isDailyResearchCandidate(passMid, 0.55), false);
  assert.equal(isDailyResearchCandidate(juiceWin, 0.72), true);
  assert.deepEqual(dailyResearchPool(publishedOnly, probabilityOf).map(item => item.id), ['juice-bet-win']);
  assert.deepEqual(
    dailyResearchPool(posted, probabilityOf).map(item => item.id).sort(),
    ['f5-pass-win', 'juice-bet-win', 'tt-pass-loss'],
  );
  assert.deepEqual(record(dailyResearchPool(posted, probabilityOf)), {
    wins: 2,
    losses: 1,
    pending: 0,
    net: 0,
  });
});

