'use strict';
(async () => {
  const app = location.pathname.split('/')[2];
  const status = document.getElementById('community-status');
  const frame = document.getElementById('community-frame');

  async function value(response) {
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || '应用请求失败。');
      error.code = data.code;
      throw error;
    }
    return data;
  }
  const post = (path, body) => fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) }).then(value);
  const asset = (id, exportFile = false) => {
    if (typeof id !== 'string' || !/^[\da-f-]{36}$/i.test(id)) throw new Error('无效的应用素材。');
    return '/apps/' + app + '/_media/' + id + (exportFile ? '?download=1' : '');
  };
  function save(url, name) {
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = name; document.body.append(anchor); anchor.click(); anchor.remove();
    return true;
  }
  const addMedia = async data => {
    if (Array.isArray(data)) return Promise.all(data.map(addMedia));
    const item = data?.output || data;
    if (item?.url?.startsWith('/apps/' + app + '/_media/') && item?.id) {
      const response = await fetch(asset(item.id));
      if (response.ok) {
        item.blob = await response.blob(); delete item.url;
      }
    }
    return data;
  };

  function applyTheme() {
    try {
      const setting = JSON.parse(localStorage.getItem('turbocider.theme'));
      document.documentElement.dataset.theme = ['dark', 'light'].includes(setting) ? setting : 'system';
    } catch (error) {}
  }
  applyTheme();
  window.addEventListener('storage', event => { if (event.key === 'turbocider.theme' || event.key === null) applyTheme(); });
  try {
    const config = await post('/api/frame', { app });
    document.getElementById('community-name').textContent = config.name;
    document.title = config.name + ' · TurboCider';
    frame.title = config.name;
    let pending = 0;
    window.addEventListener('message', async event => {
      const message = event.data;
      if (event.origin !== 'null' || event.source !== frame.contentWindow || message?.token !== config.token ||
          typeof message?.id !== 'number' || pending >= 12) return;
      const reply = (result, error) => frame.contentWindow?.postMessage({
        id: message.id, token: config.token, result, error: error?.message, code: error?.code
      }, '*'); // The iframe has an opaque origin. Only it knows this host-issued token.
      pending++;
      try {
        const payload = message.payload || {};
        const method = message.method;
        if (method === 'browser.import') {
          if (!(payload.blob instanceof Blob) || payload.blob.size > 16 * 1024 * 1024) throw new Error('图片超过 16 MiB。');
          const response = await fetch('/api/media', { method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream', 'X-TurboCider-App': app,
              'X-TurboCider-Filename': encodeURIComponent(String(payload.name || '图片').slice(0, 160)) },
            body: payload.blob });
          reply(await addMedia(await value(response)));
        } else if (method === 'browser.save' || method === 'browser.saveBlob') {
          if (!config.export) throw new Error('此应用没有保存权限。');
          if (method === 'browser.save') {
            const url = asset(payload.id, true);
            const response = await fetch(url);
            if (!response.ok) throw new Error('找不到此应用的素材。');
            void response.body?.cancel();
            reply(save(url, ''));
          } else {
            if (!(payload.blob instanceof Blob) || payload.blob.size > 24 * 1024 * 1024) throw new Error('素材包超过 24 MiB。');
            const name = String(payload.name || 'TurboCider.zip').slice(0, 160);
            if (!/\.(png|jpg|jpeg|webp|zip|txt|json|md)$/i.test(name)) throw new Error('此格式无法保存。');
            const url = URL.createObjectURL(payload.blob);
            try { reply(save(url, name)); } finally { setTimeout(() => URL.revokeObjectURL(url), 30_000); }
          }
        } else {
          const data = await post('/api/call', { app, method, payload });
          reply(['media.bundled', 'media.get', 'jobs.get', 'jobs.submit'].includes(method) ? await addMedia(data) : data);
        }
      } catch (error) { reply(undefined, error); }
      finally { pending--; }
    });
    frame.src = config.src;
    frame.hidden = false; status.hidden = true;
  } catch (error) { status.textContent = error.message; }
})();
