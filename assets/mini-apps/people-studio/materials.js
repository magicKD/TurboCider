'use strict';
(() => {
  const materials = window.PeopleStudioNative?.materials;
  const app = window.APP;
  if (!materials || !app) return;
  const $ = id => document.getElementById(id);
  const dialog = $('tcMaterialsDialog');
  const toggle = $('tcMaterialsOnline');
  const select = $('tcMaterialCollection');
  const progress = $('tcMaterialProgress');
  const status = $('tcMaterialStatus');
  const download = $('tcMaterialDownload');
  const cancel = $('tcMaterialCancel');
  let updating = false;
  let timer;
  let states = new Map();
  const readOnly = window.PeopleStudioNative.browser && !window.TurboCider.canManage;
  if (readOnly) {
    toggle.disabled = true;
    $('tcDesktopMaterialDownloads').hidden = true;
    $('tcMaterialsHostNote').hidden = false;
  }

  function paint(state) {
    toggle.checked = state.online;
    if (readOnly) return;
    if (!select.options.length) {
      for (const choice of state.collections.filter(item => item.downloadable)) {
        select.append(new Option(choice.title, choice.id));
      }
      if (state.collections.some(item => item.downloadable && item.id === app.st.tab)) select.value = app.st.tab;
    }
    for (const option of select.options) {
      const pack = state.collections.find(item => item.id === option.value);
      option.textContent = pack.title + (pack.phase === 'installed' ? ' · 已下载' : '');
    }
    const pack = state.collections.find(item => item.id === select.value);
    const active = state.collections.find(item => item.phase === 'downloading');
    cancel.hidden = !active;
    cancel.dataset.collection = active?.id || '';
    download.disabled = !pack || !!active || pack.phase === 'installed';
    download.textContent = ['failed', 'cancelled'].includes(pack?.phase) ? '继续下载' : '下载到本机';
    progress.hidden = !active;
    if (active?.total) {
      progress.max = active.total;
      progress.value = active.completed;
    } else progress.removeAttribute('value');
    if (active) {
      status.textContent = active.total ? `${active.title} · ${active.completed} / ${active.total} 张` : `正在读取${active.title}目录…`;
    } else if (pack?.phase === 'installed') {
      status.textContent = `${pack.title}已下载，可离线选用。`;
    } else if (pack?.phase === 'failed') {
      status.textContent = pack.error || '素材下载失败，请重试。';
    } else if (pack?.phase === 'cancelled') {
      status.textContent = '下载已停止，下次可继续。';
    } else status.textContent = '';
  }

  async function refresh() {
    if (updating) return;
    updating = true;
    try {
      const state = await materials.status();
      const changed = states.size > 0 && state.collections.some(pack => pack.phase === 'installed' && states.get(pack.id) !== 'installed');
      states = new Map(state.collections.map(pack => [pack.id, pack.phase]));
      paint(state);
      if (changed) await app.refreshMaterials();
    } catch (error) { status.textContent = error.message; }
    finally { updating = false; }
  }

  $('tcMaterialsClose').onclick = () => dialog.close();
  $('tcMaterials').onclick = async () => {
    dialog.showModal();
    await refresh();
    if (dialog.open && !timer) timer = setInterval(refresh, 700);
  };
  // Let a download finish while the browser or workbench is open. The native
  // host owns the task, so closing this dialog only stops status polling.
  dialog.addEventListener('close', () => {
    clearInterval(timer);
    timer = undefined;
    if ([...states.values()].includes('downloading')) {
      timer = setInterval(async () => {
        await refresh();
        if (![...states.values()].includes('downloading') && !dialog.open) {
          clearInterval(timer);
          timer = undefined;
        }
      }, 900);
    }
  });
  toggle.onchange = async () => {
    toggle.disabled = true;
    try {
      await materials.enable(toggle.checked);
      await app.refreshMaterials();
    } catch (error) { status.textContent = error.message; }
    finally { toggle.disabled = false; }
  };
  $('tcMaterialsRefresh').onclick = async () => {
    $('tcMaterialsRefresh').disabled = true;
    try { await materials.refresh(); await app.refreshMaterials(); }
    catch (error) { status.textContent = error.message; }
    finally { $('tcMaterialsRefresh').disabled = false; }
  };
  select.onchange = refresh;
  download.onclick = async () => {
    download.disabled = true;
    try { await materials.download(select.value); await refresh(); }
    catch (error) { status.textContent = error.message; }
  };
  cancel.onclick = async () => {
    await materials.cancel(cancel.dataset.collection).catch(error => { status.textContent = error.message; });
    await refresh();
  };
  document.addEventListener('click', event => {
    const source = event.target.closest('[data-material-source]')?.dataset.materialSource;
    if (source) void materials.openSource(source).catch(error => app.toast(error.message));
  });
})();
