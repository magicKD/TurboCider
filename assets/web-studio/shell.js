'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const terminal = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
  const states = { queued: '排队中', preparing: '准备中', running: '推理中', cancelling: '正在停止',
    finalizing: '保存中', succeeded: '已完成', failed: '失败', cancelled: '已取消', interrupted: '已中断',
    cleanup_pending: '等待推理进程释放' };
  const operations = { 'image.generate': '文生图', 'image.transform': '单图修改', 'image.edit': '参考编辑',
    'image.upscale': '图片超分', 'video.generate': '文生视频', 'video.image': '图生视频',
    'video.keyframes': '首尾帧视频', 'video.reference': '参考视频', 'video.upscale': '视频超分' };
  const phases = { prepare: '准备模型', denoise: '采样', ltx_step: '视频采样', ltx_block: '视频采样',
    image_encode: '编码参考图', connector: '准备生成条件', latent_upsample: '细化画面', upscale: '图像超分',
    seedvr2_encode_progress: 'VAE 编码', seedvr2_dit: '恢复细节', seedvr2_decode_progress: 'VAE 解码',
    relayvsr_restore: '流式超分', export: '保存结果', mux: '保存视频' };
  const pageNames = { create: '创作', upscale: '超分', apps: '应用', template: '创作模板', library: '素材库',
    jobs: '任务', models: '模型', settings: '设置' };
  const darkSystem = matchMedia('(prefers-color-scheme: dark)');
  let session, pendingTimer, toastTimer, connecting = false;
  let ticket = new URLSearchParams(location.hash.slice(1)).get('ticket');
  if (ticket) history.replaceState(null, '', '/#apps');

  function preference(key, value) {
    try {
      if (value !== undefined) localStorage.setItem(key, JSON.stringify(value));
      else return JSON.parse(localStorage.getItem(key));
    } catch (error) { /* A browser in private mode may disable local storage. */ }
    return null;
  }

  function syncBrowserColor() {
    const value = document.documentElement.dataset.theme;
    const dark = value === 'dark' || (value !== 'light' && darkSystem.matches);
    document.querySelector('meta[name="theme-color"]').content = dark ? '#17191b' : '#f7f6f2';
  }

  function applyTheme(value, persist = true) {
    document.documentElement.dataset.theme = value;
    if (persist) {
      preference('turbocider.theme', value);
      try {
        if (value === 'system') localStorage.removeItem('theme');
        else localStorage.setItem('theme', value);
      } catch (error) {}
    }
    const labels = { system: '跟随系统', light: '浅色', dark: '深色' };
    $('theme').title = '主题 · ' + labels[value];
    $('theme').setAttribute('aria-label', '更换色彩主题，当前为' + labels[value]);
    $('theme').replaceChildren(icon(value));
    syncBrowserColor();
  }

  function createID() {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    const hex = [...bytes].map(byte => byte.toString(16).padStart(2, '0')).join('');
    return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join('-');
  }

  async function api(path, body, options) {
    const response = await fetch(path, options || (body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    }));
    const value = await response.json();
    if (!response.ok) {
      const error = new Error(value.error || '无法连接 TurboCider。');
      error.status = response.status;
      error.code = value.code;
      if (error.status === 401 && session && !path.startsWith('/api/session')) {
        showSession({ state: 'unpaired' }, '网页服务连接已失效，请重新连接。');
      }
      throw error;
    }
    return value;
  }

  function node(tag, text, classes) {
    const item = document.createElement(tag);
    if (text !== undefined && text !== null) item.textContent = text;
    if (classes) item.className = classes;
    return item;
  }
  function icon(name) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg'), use = document.createElementNS(ns, 'use');
    svg.setAttribute('class', 'glyph');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    use.setAttribute('href', '/studio-icons.svg#' + name);
    svg.append(use);
    return svg;
  }
  function button(text, action, classes) {
    const item = node('button', text, classes);
    item.type = 'button';
    item.addEventListener('click', action);
    return item;
  }
  function link(text, href, classes) {
    const item = node('a', text, classes);
    item.href = href;
    return item;
  }
  function toast(message, error = false) {
    clearTimeout(toastTimer);
    $('toast').textContent = message;
    $('toast').classList.toggle('is-error', error);
    $('toast').hidden = false;
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 6500 : 3200);
  }
  function action(work) {
    return async event => {
      const trigger = event?.currentTarget;
      const wasDisabled = trigger?.disabled;
      if (trigger?.tagName === 'BUTTON') trigger.disabled = true;
      try { await work(event); } catch (error) { toast(error.message || '操作未完成。', true); }
      finally { if (trigger?.tagName === 'BUTTON') trigger.disabled = Boolean(wasDisabled); }
    };
  }

  function route() {
    if (!session) return;
    const [candidate, parameter] = location.hash.slice(1).split('/');
    const name = Object.hasOwn(pageNames, candidate) ? candidate : 'create';
    for (const page of document.querySelectorAll('.page')) page.hidden = page.id !== name;
    for (const tab of document.querySelectorAll('[data-nav], .mobile-settings')) {
      if ((tab.dataset.nav || 'settings') === (name === 'template' ? 'apps' : name)) tab.setAttribute('aria-current', 'page');
      else tab.removeAttribute('aria-current');
    }
    $('current-page').textContent = pageNames[name];
    document.title = pageNames[name] + ' · TurboCider';
    window.dispatchEvent(new CustomEvent('tc-route', { detail: { name, parameter } }));
    window.scrollTo({ top: 0, behavior: 'instant' });
  }

  function showSession(value, message) {
    clearTimeout(pendingTimer);
    if (value.state === 'ready') {
      const changed = session?.workspace !== value.workspace || !session;
      session = value;
      $('entry').hidden = true;
      $('workspace').hidden = false;
      $('rail-device').textContent = value.direct ? '局域网直连' : value.name || '已配对设备';
      if (changed) {
        window.dispatchEvent(new CustomEvent('tc-session', { detail: value }));
        route();
      }
      return;
    }
    if (session) window.dispatchEvent(new Event('tc-disconnect'));
    session = undefined;
    $('entry').classList.add('is-pairing');
    $('entry').hidden = false;
    $('workspace').hidden = true;
    $('pair-code').hidden = value.state !== 'pending';
    $('pair-form').hidden = !ticket || value.state !== 'unpaired';
    $('connect').hidden = value.directAvailable !== true;
    $('retry').hidden = !message || value.state === 'pending';
    if (value.state === 'pending') {
      $('pair-code').textContent = value.code;
      $('pair-status').textContent = '在 Mac 上核对验证码并允许这台设备连接。';
      pendingTimer = setTimeout(checkSession, 1200);
    } else {
      $('pair-status').textContent = message || (ticket
        ? '为这台设备起个名字，然后在 Mac 上确认配对。'
        : '在 Mac 上打开「本地 API → 网页服务」，扫描二维码或打开配对链接。');
    }
  }

  async function connect() {
    connecting = true;
    try { showSession(await api('/api/connect', {})); }
    finally { connecting = false; }
  }

  async function checkSession() {
    if (connecting) return;
    try {
      const value = await api('/api/session');
      if (value.state === 'unpaired' && value.directAvailable && !ticket) await connect();
      else showSession(value);
    } catch (error) { showSession({ state: 'unpaired' }, error.message || '无法连接到 Mac。'); }
  }

  async function upload(file, kind, path = '/api/studio/upload', extra = {}) {
    if (!file || !file.size) throw new Error('请选择有效的文件。');
    const limit = kind === 'image' ? 32 : 256;
    if (file.size > limit * 1024 * 1024) throw new Error('单个文件不能超过 ' + limit + ' MiB。');
    return api(path, undefined, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream', 'X-TurboCider-Kind': kind,
        'X-TurboCider-Filename': encodeURIComponent((file.name || 'image.png').slice(0, 160)), ...extra },
      body: file
    });
  }

  window.TC = { $, terminal, states, operations, phases, api, node, icon, button, link, toast, action, createID, preference,
    upload, route, checkSession, getSession: () => session };
  document.querySelectorAll('[data-icon]').forEach(item => item.replaceWith(icon(item.dataset.icon)));
  document.querySelectorAll('.rail [data-nav]').forEach(item => { item.title = pageNames[item.dataset.nav]; });
  $('skip').addEventListener('click', event => { event.preventDefault(); $('main').focus(); });
  $('pair-form').addEventListener('submit', async event => {
    event.preventDefault();
    const button = event.submitter;
    button.disabled = true;
    try {
      const value = await api('/api/pair', { ticket, name: $('device-name').value.trim() });
      ticket = null;
      showSession(value);
    } catch (error) { $('pair-status').textContent = error.message; }
    finally { button.disabled = false; }
  });
  $('connect').addEventListener('click', action(connect));
  $('retry').addEventListener('click', checkSession);
  $('theme').addEventListener('click', () => {
    const modes = ['system', 'light', 'dark'];
    applyTheme(modes[(modes.indexOf(document.documentElement.dataset.theme) + 1) % modes.length]);
  });
  window.addEventListener('hashchange', route);
  function restoreTheme() {
    const theme = preference('turbocider.theme');
    applyTheme(['system', 'light', 'dark'].includes(theme) ? theme : 'system', false);
  }
  restoreTheme();
  window.addEventListener('storage', event => { if (event.key === 'turbocider.theme' || event.key === null) restoreTheme(); });
  window.addEventListener('pageshow', event => { if (event.persisted) restoreTheme(); });
  darkSystem.addEventListener('change', syncBrowserColor);
  document.addEventListener('DOMContentLoaded', checkSession, { once: true });
})();
