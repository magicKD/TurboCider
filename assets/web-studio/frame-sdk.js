'use strict';
(() => {
  const { token, origin, settings } = window.TurboCiderFrame;
  delete window.TurboCiderFrame;
  const pending = new Map(), media = new Map();
  let sequence = 0;
  const display = result => {
    if (Array.isArray(result)) return result.map(display);
    const item = result?.output || result;
    if (item?.blob instanceof Blob && item.id) {
      const previous = media.get(item.id);
      item.url = previous || URL.createObjectURL(item.blob);
      media.set(item.id, item.url); delete item.blob;
    }
    return result;
  };
  window.addEventListener('message', event => {
    const { id, token: responseToken, result, error, code } = event.data || {};
    if (event.source !== parent || event.origin !== origin || responseToken !== token) return;
    const request = pending.get(id);
    if (!request) return;
    clearTimeout(request.timer); pending.delete(id);
    if (error) { const failure = new Error(error); failure.code = code; request.reject(failure); }
    else request.resolve(display(result));
  });
  const call = (method, payload = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error('应用请求超时。')); }, 185_000);
    pending.set(id, { timer, resolve, reject });
    parent.postMessage({ id, token, method, payload }, origin);
  });
  const createRequestID = () => {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    const hex = [...bytes].map(byte => byte.toString(16).padStart(2, '0')).join('');
    return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join('-');
  };
  window.TurboCiderStorage = settings;
  Object.defineProperty(window, 'TurboCider', { value: Object.freeze({
    sdkVersion: 1, platform: 'browser', call, createRequestID,
    media: Object.freeze({ url: id => media.get(id), get: id => call('media.get', { id }),
      bundled: path => call('media.bundled', { path }),
      import: (blob, name) => call('browser.import', { blob, name: name || blob.name }),
      save: id => call('browser.save', { id }), saveBlob: (blob, name) => call('browser.saveBlob', { blob, name }) }),
    storage: Object.freeze({ get: key => call('storage.get', { key }), set: (key, value) => call('storage.set', { key, value }) }),
    workflow: Object.freeze({ plan: options => call('workflow.plan', options) }),
    jobs: Object.freeze({ submit: id => call('jobs.submit', { id }), get: id => call('jobs.get', { id }),
      async *watch(id, { signal } = {}) {
        for (;;) {
          signal?.throwIfAborted();
          const state = await call('jobs.get', { id });
          yield state;
          if (['succeeded', 'failed', 'cancelled', 'interrupted'].includes(state.state)) return;
          await new Promise(resolve => setTimeout(resolve, 1100));
        }
      }, list: () => call('jobs.list'), cancel: id => call('jobs.cancel', { id }), remove: id => call('jobs.remove', { id }) })
  }) });
})();
