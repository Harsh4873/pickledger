const MOBILE_MODE_KEY = 'pickledger_mobile_mode';
const PICK_MODE_KEY = 'pickledger_pick_mode';

export type PickMode = 'team' | 'player';

export function initTheme(): void {
  if ((localStorage.getItem('pickledger_theme') || 'dark') === 'light') {
    document.body.setAttribute('data-theme', 'light');
    const label = document.getElementById('theme-label');
    if (label) label.textContent = 'LIGHT';
  }
}

export function toggleTheme(): void {
  const label = document.getElementById('theme-label');
  const light = document.body.getAttribute('data-theme') === 'light';
  if (light) {
    document.body.removeAttribute('data-theme');
    localStorage.setItem('pickledger_theme', 'dark');
    if (label) label.textContent = 'DARK';
  } else {
    document.body.setAttribute('data-theme', 'light');
    localStorage.setItem('pickledger_theme', 'light');
    if (label) label.textContent = 'LIGHT';
  }
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
