import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';

interface MockPayload {
  date?: string;
  slate_date?: string;
  models?: Record<string, { ok?: boolean; picks?: unknown[] }>;
  picks?: unknown[];
  [key: string]: unknown;
}

type FetchMock = (path: string) => Promise<MockPayload | null>;

function createDataModule(fetchMock: FetchMock) {
  const teamCachePayloads: MockPayload[] = [];
  const playerCachePayloads: MockPayload[] = [];
  const parlayPayloads: MockPayload[] = [];
  const profitDeskPayloads: MockPayload[] = [];

  const DATED_CACHE_FILE = /^\d{4}-\d{2}-\d{2}\.json$/;

  async function fetchJson<T>(path: string): Promise<T | null> {
    return fetchMock(path) as Promise<T | null>;
  }

  async function listDatedCacheFiles(indexPath: string): Promise<string[]> {
    const manifest = await fetchJson<{ files?: string[] }>(indexPath);
    return Array.isArray(manifest?.files)
      ? manifest.files.filter((file: string) => DATED_CACHE_FILE.test(file))
      : [];
  }

  function mergePayloadsByDate<T>(
    existing: T[],
    incoming: T[],
    dateOf: (payload: T) => string,
  ): T[] {
    const byDate = new Map<string, T>();
    for (const payload of incoming) {
      const date = dateOf(payload);
      if (date) byDate.set(date, payload);
    }
    for (const payload of existing) {
      const date = dateOf(payload);
      if (date) byDate.set(date, payload);
    }
    return [...byDate.values()].sort((left, right) =>
      dateOf(left).localeCompare(dateOf(right)),
    );
  }

  async function loadLatestAndNewestDated<T extends { date?: string }>(
    latestPath: string,
    indexPath: string,
    dir: string,
    dateOf: (payload: T) => string,
  ): Promise<T[]> {
    const [latest, files] = await Promise.all([
      fetchJson<T>(latestPath),
      listDatedCacheFiles(indexPath),
    ]);
    const sorted = [...files].sort();
    const newestFile = sorted[sorted.length - 1];
    const newestDate = newestFile?.replace(/\.json$/, '') ?? '';
    const latestDate = latest ? dateOf(latest) : '';
    const payloads: T[] = [];
    if (latest) payloads.push(latest);
    if (newestFile && newestDate > latestDate) {
      const newestPayload = await fetchJson<T>(`${dir}/${newestFile}`);
      if (newestPayload) payloads.push(newestPayload);
    } else if (!latest && newestFile) {
      const fallback = await fetchJson<T>(`${dir}/${newestFile}`);
      if (fallback) payloads.push(fallback);
    }
    return payloads;
  }

  return {
    teamCachePayloads,
    playerCachePayloads,
    parlayPayloads,
    profitDeskPayloads,
    mergePayloadsByDate,
    loadLatestAndNewestDated,
  };
}

let mockResponses: Map<string, MockPayload | null>;

beforeEach(() => {
  mockResponses = new Map();
});

afterEach(() => {
  mockResponses.clear();
});

function mockFetch(path: string): Promise<MockPayload | null> {
  return Promise.resolve(mockResponses.get(path) ?? null);
}

test('first paint includes newer dated payload when latest.json is an older complete day', async () => {
  mockResponses.set('./data/model_cache/latest.json', {
    date: '2026-08-26',
    models: {
      mlb_new: { ok: true, picks: [{ pick: 'Yankees ML', sport: 'MLB', date: '2026-08-26' }] },
    },
  });
  mockResponses.set('./data/model_cache/index.json', {
    files: ['2026-08-25.json', '2026-08-26.json', '2026-08-27.json'],
  });
  mockResponses.set('./data/model_cache/2026-08-27.json', {
    date: '2026-08-27',
    models: {
      scores24_mlb: { ok: true, picks: [{ pick: 'Red Sox ML', sport: 'MLB', date: '2026-08-27' }] },
    },
  });

  const module = createDataModule(mockFetch);
  const payloads = await module.loadLatestAndNewestDated<MockPayload>(
    './data/model_cache/latest.json',
    './data/model_cache/index.json',
    './data/model_cache',
    (p) => String(p.date || ''),
  );

  assert.equal(payloads.length, 2, 'should fetch both latest.json and newer dated file');
  assert.equal(payloads[0].date, '2026-08-26', 'first payload is from latest.json');
  assert.equal(payloads[1].date, '2026-08-27', 'second payload is the newer dated file');

  const merged = module.mergePayloadsByDate([], payloads, (p) => String(p.date || ''));
  assert.equal(merged.length, 2, 'merged result has both dates');
  assert.equal(merged[0].date, '2026-08-26');
  assert.equal(merged[1].date, '2026-08-27');
  assert.ok(
    (merged[1].models as Record<string, { picks?: unknown[] }>)?.scores24_mlb?.picks,
    'today Scores24 picks are present in merged result',
  );
});

test('first paint does not fetch duplicate when latest.json matches newest dated file', async () => {
  mockResponses.set('./data/model_cache/latest.json', {
    date: '2026-08-27',
    models: {
      mlb_new: { ok: true, picks: [{ pick: 'Yankees ML', sport: 'MLB', date: '2026-08-27' }] },
    },
  });
  mockResponses.set('./data/model_cache/index.json', {
    files: ['2026-08-25.json', '2026-08-26.json', '2026-08-27.json'],
  });

  const module = createDataModule(mockFetch);
  const payloads = await module.loadLatestAndNewestDated<MockPayload>(
    './data/model_cache/latest.json',
    './data/model_cache/index.json',
    './data/model_cache',
    (p) => String(p.date || ''),
  );

  assert.equal(payloads.length, 1, 'should only fetch latest.json when dates match');
  assert.equal(payloads[0].date, '2026-08-27');
});

test('first paint falls back to newest dated file when latest.json is missing', async () => {
  mockResponses.set('./data/model_cache/latest.json', null);
  mockResponses.set('./data/model_cache/index.json', {
    files: ['2026-08-25.json', '2026-08-26.json', '2026-08-27.json'],
  });
  mockResponses.set('./data/model_cache/2026-08-27.json', {
    date: '2026-08-27',
    models: {
      scores24_mlb: { ok: true, picks: [{ pick: 'Red Sox ML', sport: 'MLB', date: '2026-08-27' }] },
    },
  });

  const module = createDataModule(mockFetch);
  const payloads = await module.loadLatestAndNewestDated<MockPayload>(
    './data/model_cache/latest.json',
    './data/model_cache/index.json',
    './data/model_cache',
    (p) => String(p.date || ''),
  );

  assert.equal(payloads.length, 1, 'should fallback to newest dated file');
  assert.equal(payloads[0].date, '2026-08-27');
});

test('merge does not overwrite historical models with today data', async () => {
  mockResponses.set('./data/model_cache/latest.json', {
    date: '2026-08-26',
    models: {
      mlb_new: { ok: true, picks: [{ pick: 'Yankees ML', sport: 'MLB', date: '2026-08-26' }] },
      scores24_mlb: { ok: true, picks: [{ pick: 'Mets ML', sport: 'MLB', date: '2026-08-26' }] },
    },
  });
  mockResponses.set('./data/model_cache/index.json', {
    files: ['2026-08-26.json', '2026-08-27.json'],
  });
  mockResponses.set('./data/model_cache/2026-08-27.json', {
    date: '2026-08-27',
    models: {
      scores24_mlb: { ok: true, picks: [{ pick: 'Red Sox ML', sport: 'MLB', date: '2026-08-27' }] },
    },
  });

  const module = createDataModule(mockFetch);
  const payloads = await module.loadLatestAndNewestDated<MockPayload>(
    './data/model_cache/latest.json',
    './data/model_cache/index.json',
    './data/model_cache',
    (p) => String(p.date || ''),
  );

  const merged = module.mergePayloadsByDate([], payloads, (p) => String(p.date || ''));

  const aug26 = merged.find((p) => p.date === '2026-08-26');
  const aug27 = merged.find((p) => p.date === '2026-08-27');

  assert.ok(aug26, '2026-08-26 payload should exist');
  assert.ok(aug27, '2026-08-27 payload should exist');

  const aug26Models = aug26.models as Record<string, { picks?: unknown[] }>;
  const aug27Models = aug27.models as Record<string, { picks?: unknown[] }>;

  assert.ok(aug26Models.mlb_new?.picks, '8/26 mlb_new picks remain on 8/26');
  assert.ok(aug26Models.scores24_mlb?.picks, '8/26 scores24 picks remain on 8/26');
  assert.ok(!aug27Models.mlb_new, '8/27 does not have mlb_new (it only has scores24)');
  assert.ok(aug27Models.scores24_mlb?.picks, '8/27 has its own scores24 picks');
});
