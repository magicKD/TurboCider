'use strict';
(() => {
  const { app, base, settings, canManage } = window.TurboCiderWebConfig;
  delete window.TurboCiderWebConfig;
  document.documentElement.dataset.tcWeb = 'true';
  try {
    const theme = JSON.parse(localStorage.getItem('turbocider.theme'));
    if (['light', 'dark', 'system'].includes(theme)) settings.theme = JSON.stringify(theme);
  } catch (error) {}
  async function responseValue(response) {
    const value = await response.json();
    if (!response.ok) {
      const error = new Error(value.error || '网页请求未完成。');
      error.code = value.code;
      error.status = response.status;
      throw error;
    }
    return value;
  }
  const call = async (method, payload = {}) => responseValue(await fetch('/api/call', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ app, method, payload })
  }));
  const createRequestID = () => {
    // getRandomValues is available when an IP link is served over HTTP.
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
    return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join('-');
  };
  const mediaURL = id => base + '_media/' + encodeURIComponent(id);
  function download(url, name) {
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    return true;
  }
  window.TurboCiderStorage = settings;
  Object.defineProperty(window, 'TurboCider', {
    value: Object.freeze({
      sdkVersion: 1, platform: 'browser', canManage, call, createRequestID,
      media: Object.freeze({
        url: mediaURL,
        bundled: path => call('media.bundled', { path }),
        get: id => call('media.get', { id }),
        async import(blob, name) {
          if (blob.size > 16 * 1024 * 1024) throw new Error('单张图片上限为 16 MiB。');
          return responseValue(await fetch('/api/media', {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream', 'X-TurboCider-App': app,
              'X-TurboCider-Filename': encodeURIComponent((name || blob.name || '图片').slice(0, 160)) },
            body: blob
          }));
        },
        save: async id => download(mediaURL(id) + '?download=1', ''),
        async saveBlob(blob, name) {
          if (blob.size > 24 * 1024 * 1024) throw new Error('单次导出上限为 24 MiB。');
          const url = URL.createObjectURL(blob);
          try { return download(url, name); }
          finally { setTimeout(() => URL.revokeObjectURL(url), 30_000); }
        }
      }),
      storage: Object.freeze({
        get: key => call('storage.get', { key }),
        set(key, value) {
          if (key === 'theme') {
            try {
              const theme = JSON.parse(value);
              if (['light', 'dark', 'system'].includes(theme)) {
                localStorage.setItem('turbocider.theme', JSON.stringify(theme));
                if (theme === 'system') localStorage.removeItem('theme'); else localStorage.setItem('theme', theme);
              }
            } catch (error) {}
          }
          return call('storage.set', { key, value });
        }
      }),
      workflow: Object.freeze({ plan: options => call('workflow.plan', options) }),
      jobs: Object.freeze({
        async submit(id) {
          try { return await call('jobs.submit', { id }); }
          catch (error) {
            if (!(error instanceof TypeError)) throw error;
            return call('jobs.submit', { id });
          }
        },
        get: id => call('jobs.get', { id }),
        async *watch(id, { signal } = {}) {
          for (;;) {
            signal?.throwIfAborted();
            const state = await call('jobs.get', { id });
            yield state;
            if (['succeeded', 'failed', 'cancelled', 'interrupted'].includes(state.state)) return;
            await new Promise(resolve => setTimeout(resolve, 1100));
          }
        },
        list: () => call('jobs.list'),
        cancel: id => call('jobs.cancel', { id }),
        remove: id => call('jobs.remove', { id })
      })
    })
  });
  document.addEventListener('DOMContentLoaded', () => { document.querySelector('.tc-web-back').hidden = false; });
})();
