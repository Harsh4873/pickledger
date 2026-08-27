import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import {
  getTeamPicks,
  loadAllData,
  loadLatestAndNewestDated,
  setPickMode,
} from '../src/data.ts';

type CachePayload = Record<string, unknown>;

const realFetch = globalThis.fetch;

function installFetch(responses: Map<string, CachePayload>): string[] {
  const requests: string[] = [];
  globalThis.fetch = async (input: RequestInfo | URL): Promise<Response> => {
    const path = typeof input === 'string' ? input : input.toString();
    requests.push(path);
    const payload = responses.get(path);
    if (!payload) return new Response(null, { status: 404 });
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  return requests;
}

afterEach(() => {
  globalThis.fetch = realFetch;
});

test('loads a newer dated payload alongside latest.json', { concurrency: false }, async () => {
  const requests = installFetch(new Map([
    ['./data/model_cache/latest.json', { date: '2026-08-26' }],
    ['./data/model_cache/index.json', { files: ['2026-08-25.json', '2026-08-27.json'] }],
    ['./data/model_cache/2026-08-27.json', { date: '2026-08-27' }],
  ]));

  const payloads = await loadLatestAndNewestDated<{ date?: string }>(
    './data/model_cache/latest.json',
    './data/model_cache/index.json',
    './data/model_cache',
    payload => payload.date || '',
  );

  assert.deepEqual(payloads.map(payload => payload.date), ['2026-08-26', '2026-08-27']);
  assert.ok(requests.includes('./data/model_cache/index.json'));
  assert.ok(requests.includes('./data/model_cache/2026-08-27.json'));
});

test('does not refetch a dated payload when latest.json already has that date', { concurrency: false }, async () => {
  const requests = installFetch(new Map([
    ['./data/model_cache/latest.json', { date: '2026-08-27' }],
    ['./data/model_cache/index.json', { files: ['2026-08-26.json', '2026-08-27.json'] }],
  ]));

  const payloads = await loadLatestAndNewestDated<{ date?: string }>(
    './data/model_cache/latest.json',
    './data/model_cache/index.json',
    './data/model_cache',
    payload => payload.date || '',
  );

  assert.deepEqual(payloads.map(payload => payload.date), ['2026-08-27']);
  assert.ok(!requests.includes('./data/model_cache/2026-08-27.json'));
});

test('falls back to the newest dated payload when latest.json is unavailable', { concurrency: false }, async () => {
  const requests = installFetch(new Map([
    ['./data/model_cache/index.json', { files: ['2026-08-27.json'] }],
    ['./data/model_cache/2026-08-27.json', { date: '2026-08-27' }],
  ]));

  const payloads = await loadLatestAndNewestDated<{ date?: string }>(
    './data/model_cache/latest.json',
    './data/model_cache/index.json',
    './data/model_cache',
    payload => payload.date || '',
  );

  assert.deepEqual(payloads.map(payload => payload.date), ['2026-08-27']);
});

test('first paint keeps prior-day models on their original date', { concurrency: false }, async () => {
  const requests = installFetch(new Map([
    ['./data/model_cache/latest.json', {
      date: '2026-08-26',
      models: {
        mlb_new: {
          ok: true,
          picks: [{ sport: 'MLB', pick: 'Yesterday model', decision: 'BET' }],
        },
      },
    }],
    ['./data/model_cache/index.json', { files: ['2026-08-26.json', '2026-08-27.json'] }],
    ['./data/model_cache/2026-08-27.json', {
      date: '2026-08-27',
      models: {
        scores24_mlb: {
          ok: true,
          picks: [{ sport: 'MLB', pick: 'Today Scores24', decision: 'BET' }],
        },
      },
    }],
  ]));

  setPickMode('team');
  await loadAllData({ includeHistory: false });

  const priorDay = getTeamPicks().filter(pick => pick.date === '2026-08-26');
  const today = getTeamPicks().filter(pick => pick.date === '2026-08-27');
  assert.deepEqual(priorDay.map(pick => pick.pick), ['Yesterday model']);
  assert.deepEqual(today.map(pick => pick.pick), ['Today Scores24']);
  assert.ok(requests.includes('./data/model_cache/2026-08-27.json'));
});
