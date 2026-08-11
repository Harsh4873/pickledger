const MOBILE_MODE_KEY = 'pickledger_mobile_mode';
const PICK_MODE_KEY = 'pickledger_pick_mode';
const THEME_KEY = 'pickledger_theme';

export type PickMode = 'team' | 'player';
export type Theme = 'light' | 'dark';

const LIGHT_QUERY = '(prefers-color-scheme: light)';

function storedTheme(): Theme | null {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    return saved === 'light' || saved === 'dark' ? saved : null;
  } catch {
    return null;
  }
}

function systemTheme(): Theme {
  try {
    return matchMedia(LIGHT_QUERY).matches ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

/**
 * An explicit saved choice wins; with none, follow the OS; dark is the final
 * fallback. Mirrors the pre-paint resolver inlined in index.html.
 */
export function resolveTheme(): Theme {
  return storedTheme() ?? systemTheme();
}

function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  if (theme === 'light') document.body.setAttribute('data-theme', 'light');
  else document.body.removeAttribute('data-theme');

  const label = document.getElementById('theme-label');
  if (label) label.textContent = theme === 'light' ? 'LIGHT' : 'DARK';
  const toggle = document.querySelector('.theme-toggle');
  if (toggle) toggle.setAttribute('aria-pressed', String(theme === 'light'));
}

export function initTheme(): void {
  applyTheme(resolveTheme());
  try {
    matchMedia(LIGHT_QUERY).addEventListener('change', event => {
      // A saved preference always outranks the OS.
      if (storedTheme()) return;
      applyTheme(event.matches ? 'light' : 'dark');
    });
  } catch {
    // Browsers without matchMedia listeners keep the theme resolved at load.
  }
}

export function toggleTheme(): void {
  const next: Theme = document.body.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    // Preference cannot be persisted, but the theme still applies for this visit.
  }
  applyTheme(next);
}

function applyMobileMode(enabled: boolean): void {
  document.body.classList.toggle('mobile-app-mode', enabled);
  const btn = document.getElementById('mobile-mode-toggle');
  const label = document.getElementById('mobile-mode-label');
  if (btn) btn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
  if (label) label.textContent = enabled ? 'MOBILE' : 'DESK';
}

export function initMobileMode(): void {
  applyMobileMode(localStorage.getItem(MOBILE_MODE_KEY) === 'mobile');
}

export function toggleMobileMode(): void {
  const enabled = !document.body.classList.contains('mobile-app-mode');
  localStorage.setItem(MOBILE_MODE_KEY, enabled ? 'mobile' : 'desktop');
  applyMobileMode(enabled);
}

function applyPickMode(mode: PickMode): void {
  document.body.setAttribute('data-pick-mode', mode);
  document.querySelectorAll<HTMLButtonElement>('[data-pick-mode]').forEach(button => {
    const active = button.dataset.pickMode === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

export function initPickMode(): PickMode {
  const stored = localStorage.getItem(PICK_MODE_KEY);
  const mode: PickMode = stored === 'player' ? 'player' : 'team';
  applyPickMode(mode);
  return mode;
}

export function setPickMode(mode: PickMode): void {
  if (mode !== 'team' && mode !== 'player') return;
  if (document.body.getAttribute('data-pick-mode') === mode) return;
  localStorage.setItem(PICK_MODE_KEY, mode);
  applyPickMode(mode);
  document.dispatchEvent(new CustomEvent('pickledger:modechange', { detail: { mode } }));
}

export function initSettingsUI(): void {
  Object.assign(window, {
    toggleTheme,
    toggleMobileMode,
    setPickMode,
  });
}
