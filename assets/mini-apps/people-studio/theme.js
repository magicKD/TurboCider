// Run in the document head so the first paint uses the saved appearance.
(() => {
  const root = document.documentElement;
  const system = matchMedia('(prefers-color-scheme: dark)');
  const tint = document.querySelector('meta[name="theme-color"]');
  const preference = value => value === 'dark' || value === 'light' ? value : 'system';
  let selected = 'system';
  try { selected = preference(window.PeopleStudioNative ? PeopleStudioNative.get('theme') : localStorage.getItem('theme')); } catch { /* Storage can be disabled. */ }

  function apply() {
    const resolved = selected === 'system' ? (system.matches ? 'dark' : 'light') : selected;
    root.dataset.theme = resolved;
    if (tint) tint.content = root.dataset.tcWeb
      ? (resolved === 'dark' ? '#17191b' : '#f7f6f2') : (resolved === 'dark' ? '#111111' : '#ffffff');
    for (const button of document.querySelectorAll('[data-theme-choice]')) {
      button.setAttribute('aria-pressed', String(button.dataset.themeChoice === selected));
    }
  }
  function select(value) {
    selected = preference(value);
    apply();
    try { if (window.PeopleStudioNative) PeopleStudioNative.set('theme', selected); else localStorage.setItem('theme', selected); } catch { /* Keep the choice for this visit. */ }
  }

  apply();
  document.addEventListener('DOMContentLoaded', apply, {once: true});
  system.addEventListener('change', () => { if (selected === 'system') apply(); });
  window.addEventListener('storage', event => {
    if (event.storageArea !== localStorage || (event.key !== 'theme' && event.key !== null)) return;
    selected = preference(event.newValue);
    apply();
  });
  window.PoseWearTheme = {select, toggle() { select(root.dataset.theme === 'dark' ? 'light' : 'dark'); }};
})();
