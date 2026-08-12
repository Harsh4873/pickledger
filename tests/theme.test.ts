import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import { initTheme, resolveTheme } from '../src/settings.ts';

type ThemeEnvironment = {
  bodyTheme: () => string | null;
  emitPreference: (prefersLight: boolean) => void;
  rootTheme: () => string | undefined;
};

function installThemeEnvironment(saved: string | null, prefersLight: boolean): ThemeEnvironment {
  let stored = saved;
  let listener: ((event: { matches: boolean }) => void) | null = null;
  const dataset: Record<string, string> = {};
  const bodyAttributes = new Map<string, string>();

  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (key: string) => key === 'pickledger_theme' ? stored : null,
      setItem: (key: string, value: string) => {
        if (key === 'pickledger_theme') stored = value;
      },
    },
  });
  Object.defineProperty(globalThis, 'matchMedia', {
    configurable: true,
    value: () => ({
      matches: prefersLight,
      addEventListener: (_type: string, callback: (event: { matches: boolean }) => void) => {
        listener = callback;
      },
      removeEventListener: () => undefined,
    }),
  });
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {
      documentElement: { dataset },
      body: {
        getAttribute: (name: string) => bodyAttributes.get(name) ?? null,
        removeAttribute: (name: string) => bodyAttributes.delete(name),
        setAttribute: (name: string, value: string) => bodyAttributes.set(name, value),
      },
      getElementById: () => null,
      querySelector: () => null,
    },
  });

  return {
    bodyTheme: () => bodyAttributes.get('data-theme') ?? null,
    emitPreference: (next: boolean) => listener?.({ matches: next }),
    rootTheme: () => dataset.theme,
  };
}

afterEach(() => {
  Reflect.deleteProperty(globalThis, 'document');
  Reflect.deleteProperty(globalThis, 'localStorage');
  Reflect.deleteProperty(globalThis, 'matchMedia');
});

test('a fresh visitor resolves the operating-system theme', () => {
  installThemeEnvironment(null, true);
  assert.equal(resolveTheme(), 'light');
});

test('a saved choice outranks the operating-system theme', () => {
  installThemeEnvironment('dark', true);
  assert.equal(resolveTheme(), 'dark');
});

test('initialization applies and follows the OS while no choice is saved', () => {
  const environment = installThemeEnvironment(null, false);
  initTheme();
  assert.equal(environment.rootTheme(), 'dark');
  assert.equal(environment.bodyTheme(), null);

  environment.emitPreference(true);
  assert.equal(environment.rootTheme(), 'light');
  assert.equal(environment.bodyTheme(), 'light');
});

test('initialization does not let later OS changes override a saved choice', () => {
  const environment = installThemeEnvironment('light', false);
  initTheme();
  assert.equal(environment.rootTheme(), 'light');

  environment.emitPreference(false);
  assert.equal(environment.rootTheme(), 'light');
  assert.equal(environment.bodyTheme(), 'light');
});
