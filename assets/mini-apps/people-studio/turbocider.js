'use strict';
(() => {
  const host = window.TurboCider;
  if (!host) return;
  const browser = host.platform === 'browser';
  const settings = Object.assign(Object.create(null), window.TurboCiderStorage || {});
  const get = key => {
    try { return JSON.parse(settings[key] ?? 'null'); } catch { return null; }
  };
  const set = (key, value) => {
    settings[key] = JSON.stringify(value);
    void host.storage.set(key, settings[key]).catch(error => console.error(error.message));
  };
  const engine = 'qwen-image-2.1-turbo';
  const model = 'Qwen Image 2.1 Turbo · 8 步';
  const dimensions = {
    '1:2': [384, 768], '2:3': [384, 576], '3:4': [416, 544], '9:16': [384, 672],
    '4:3': [544, 416], '16:9': [672, 384], '2:1': [768, 384]
  };
  const roleByType = { pose: 'pose', depth: 'depth', expr: 'expression', hair: 'hairstyle',
    scene: 'scene', light: 'lighting', prop: 'prop', furn: 'furniture' };
  const blobIDs = new WeakMap();
  const bundled = new Map();
  const cache = new Map();
  let visibleHistory = [];
  let settingsOpen = false;

  async function asBlob(value, signal) {
    if (value instanceof Blob) return value;
    const response = await fetch(value, { signal });
    if (!response.ok) throw new Error('无法读取本机图片。');
    return response.blob();
  }

  async function importImage(blob, name) {
    if (blobIDs.has(blob)) return blobIDs.get(blob);
    const value = await host.media.import(blob, name);
    blobIDs.set(blob, value);
    return value;
  }

  async function bind(ref, item) {
    if (ref.tcMediaID || item?.tcMediaID) return ref.tcMediaID || item.tcMediaID;
    if (ref.blob) return (await importImage(ref.blob, ref.name)).id;
    const view = item?.angles?.[ref.ai ?? item.ai];
    const materialID = view?.materialID || item?.materialID;
    if (materialID) return (await host.call('materials.claim', { id: materialID })).id;
    const path = view?.i || item?.i;
    if (!path) throw new Error('有参考图已不可用，请重新选择。');
    if (!bundled.has(path)) bundled.set(path, await host.media.bundled(path.replace(/^\//, '')));
    return bundled.get(path).id;
  }

  function imageURL(id) {
    if (!id) return null;
    return host.media.url?.(id) ?? `tcapp://${location.hostname}/_media/${id}`;
  }

  function shotGroups(batch, job, items) {
    return [
      ['persons', batch.persons, () => 'person'],
      ['garments', batch.garments, () => 'garment'],
      ['poses', job.set, ref => roleByType[items[ref.k]?.type] || 'pose'],
      ['exprs', batch.exprs, () => 'expression'],
      ['extraRefs', batch.extraRefs, ref => roleByType[items[ref.k]?.type] || 'reference']
    ];
  }

  function workflowFor(batch) {
    if (batch.garments.length && batch.jobs.some(job => job.set.length)) return 'people.outfit-pose';
    if (batch.persons.length && batch.garments.length) return 'people.outfit';
    if (batch.jobs.some(job => job.set.length)) return 'people.pose';
    return 'people.compose';
  }

  async function submit(batch, items, signal) {
    const shots = [];
    const displayIndices = [];
    const [width, height] = dimensions[batch.ar] || [512, 512];
    const canvas = batch.ar === 'auto' ? { matchReferenceAspect: true } : { width, height };
    const adapters = get('wbTurboAdapters') || [];
    const workflow = workflowFor(batch);
    let layered = false;
    for (const job of batch.jobs) {
      const groups = {};
      const inputs = { ar: batch.ar, lang: batch.lang, bm: batch.bm };
      for (const [key, group, role] of shotGroups(batch, job, items)) {
        const references = groups[key] = [];
        inputs[key] = [];
        for (const ref of group) {
          signal.throwIfAborted();
          const item = items[ref.k];
          const mediaID = await bind(ref, item);
          references.push({ role: role(ref), mediaID });
          inputs[key].push({ k: ref.k, ai: ref.ai, name: ref.name, w: ref.w, h: ref.h, mediaID,
            ...(item?.materialSource ? { sourceCollection: item.type, sourceTitle: item.ti || item.zh } : {}) });
        }
      }
      const recipe = { mode: batch.mode, inputs, prompt: job.prompt, orig: job.soft ? job.original : null, neutral: job.soft };
      const settings = { ...canvas, referencePixelBudget: 512 * 512, adapters };
      const canSplit = !!job.stages;
      if (canSplit) {
        layered = true;
        const dependency = shots.length;
        shots.push({
          ...settings,
          references: [...groups.persons, ...groups.poses, ...groups.exprs, ...groups.extraRefs],
          prompt: job.stages.pose + '\nMaintain the clothing from the person reference.',
          label: job.label ? `${job.label} · 姿势与场景` : '姿势与场景',
          context: JSON.stringify({ ...recipe, intermediate: true })
        });
        displayIndices.push(shots.length);
        shots.push({
          ...settings,
          references: [{ role: 'person', shotIndex: dependency }, ...groups.garments],
          prompt: job.stages.garment + '\nKeep the pose, expression, framing and scenery of the first image.',
          label: job.label ? `${job.label} · 换装` : '换装',
          context: JSON.stringify(recipe)
        });
      } else {
        displayIndices.push(shots.length);
        shots.push({ ...settings, prompt: job.prompt, references: Object.values(groups).flat(), label: job.label,
          context: JSON.stringify(recipe) });
      }
    }
    if (shots.length > 12) throw new Error('单批最多 12 个任务，请减少每个姿势的出图数量。');
    signal.throwIfAborted();
    const plan = await host.workflow.plan({ requestID: host.createRequestID(),
      workflow: layered ? `${workflow}.staged` : workflow, engine, shots });
    signal.throwIfAborted();
    const jobs = await host.jobs.submit(plan.id);
    const cancel = () => Promise.allSettled(jobs.map(job => host.jobs.cancel(job.id)));
    if (signal.aborted) await cancel();
    else signal.addEventListener('abort', cancel, { once: true });
    signal.throwIfAborted();
    return displayIndices.map(index => jobs[index]);
  }

  async function wait(initial, signal, status) {
    let job = initial;
    let delay = browser ? 1050 : 550;
    while (!['succeeded', 'failed', 'cancelled', 'interrupted'].includes(job.state)) {
      signal.throwIfAborted();
      status(job);
      await new Promise(resolve => setTimeout(resolve, delay));
      signal.throwIfAborted();
      try {
        job = await host.jobs.get(job.id);
        delay = browser ? 1050 : 550;
      } catch (error) {
        if (!browser || !(error instanceof TypeError)) throw error;
        delay = Math.min(6000, delay * 2);
      }
    }
    signal.throwIfAborted();
    if (job.state !== 'succeeded' || !job.output) throw new Error(job.error || '本机任务未完成。');
    return job;
  }

  function record(job) {
    let recipe = {};
    try { recipe = JSON.parse(job.context || '{}'); } catch {}
    const inputs = recipe.inputs;
    const source = inputs?.persons?.[0];
    return {
      t: job.createdAt, tcID: job.id, tcOutput: job.output.id, prov: 'local',
      model, mode: recipe.mode || 'gen', label: job.label, prompt: job.prompt,
      orig: recipe.orig || recipe.prompt, neutral: recipe.neutral, inputs, intermediate: !!recipe.intermediate,
      width: job.output.width, height: job.output.height, referencePixels: job.referencePixels,
      before: imageURL(source?.mediaID), after: job.output.url
    };
  }

  async function history() {
    const hidden = new Set(get('wbHiddenJobs') || []);
    visibleHistory = (await host.jobs.list())
      .filter(job => job.state === 'succeeded' && job.output && !hidden.has(job.id))
      .map(job => cache.get(job.id) || record(job))
      .filter(rec => !rec.intermediate);
    return visibleHistory;
  }

  async function transaction(_mode, fn) {
    let result;
    const hide = entries => set('wbHiddenJobs', [...new Set([...(get('wbHiddenJobs') || []), ...entries])].slice(-500));
    const store = {
      getAll: () => { result = history(); },
      delete: t => hide(visibleHistory.filter(item => item.t === t).map(item => item.tcID)),
      put: rec => {
        if (!rec.tcID) return;
        set('wbHiddenJobs', (get('wbHiddenJobs') || []).filter(id => id !== rec.tcID));
      },
      clear: () => hide(visibleHistory.map(item => item.tcID))
    };
    fn(store);
    return result;
  }

  async function settingsDialog() {
    if (settingsOpen) return;
    settingsOpen = true;
    const dialog = document.querySelector('#tcSettings');
    const status = document.querySelector('#tcModelStatus');
    const list = document.querySelector('#tcAdapters');
    const save = document.querySelector('#tcSaveAdapters');
    dialog.showModal();
    const update = async () => {
      list.replaceChildren();
      try {
        const models = await host.call('models.list');
        status.textContent = models.find(item => item.id === engine)?.installed ? model
          : browser ? '请在 Mac 的 TurboCider 中选择 Qwen Image 2.1 Turbo 模型目录。' : '请选择已下载的 Qwen Image 2.1 Turbo 模型目录。';
        if (!models.find(item => item.id === engine)?.installed) return;
        const adapters = await host.call('adapters.list', { engine });
        if (!adapters.length) {
          const empty = document.createElement('p');
          empty.textContent = '在 TurboCider 的「模型」中导入 LoRA 后，兼容项会显示在这里。';
          list.append(empty);
        }
        const active = new Map((get('wbTurboAdapters') || []).map(adapter => [adapter.id, adapter.strength]));
        const selections = [];
        for (const item of adapters) {
          const label = document.createElement('label');
          const check = document.createElement('input');
          const strength = document.createElement('input');
          check.type = 'checkbox';
          check.checked = active.has(item.id);
          strength.type = 'number'; strength.min = -8; strength.max = 8; strength.step = 0.1;
          strength.value = active.get(item.id) ?? item.strength;
          strength.setAttribute('aria-label', item.name + ' 强度');
          label.append(check, document.createTextNode(item.name));
          const row = document.createElement('div');
          row.className = 'tc-adapter';
          row.append(label, strength);
          list.append(row);
          selections.push({ item, check, strength });
        }
        save.onclick = () => {
          const chosen = selections.filter(row => row.check.checked);
          if (chosen.length > 8 || chosen.some(row => !row.strength.checkValidity() || !row.strength.value)) {
            status.textContent = '最多选择 8 项。强度须在 −8 到 8 之间。';
            return;
          }
          set('wbTurboAdapters', chosen.map(row => ({ id: row.item.id, strength: Number(row.strength.value) })));
          dialog.close();
        };
      } catch (error) { status.textContent = error.message; }
    };
    const choose = document.querySelector('#tcChooseModel');
    choose.hidden = browser;
    choose.onclick = async () => {
      try { await host.call('models.chooseTurbo'); await update(); }
      catch (error) { status.textContent = error.message; }
    };
    document.querySelector('#tcCloseSettings').onclick = () => dialog.close();
    save.onclick = () => dialog.close();
    dialog.addEventListener('close', () => { settingsOpen = false; }, { once: true });
    await update();
  }

  function remember(rec) {
    if (rec.tcID) {
      cache.set(rec.tcID, rec);
      if (cache.size > 12) cache.delete(cache.keys().next().value);
    }
  }

  const materialCollections = new Set(['pose', 'depth', 'costume', 'anime', 'portrait', 'expr', 'hair', 'dyn', 'scene', 'light', 'prop', 'furn']);
  const materials = {
    catalog: collection => materialCollections.has(collection) ? host.call('materials.catalog', { collection }) : [],
    status: () => host.call('materials.status'),
    refresh: () => host.call('materials.refresh'),
    enable: enabled => host.call('materials.online', { enabled }),
    download: collection => host.call('materials.download', { collection }),
    cancel: collection => host.call('materials.cancel', { collection }),
    async openSource(source) {
      if (!browser) return host.call('materials.openSource', { source });
      const page = window.open('about:blank', '_blank');
      if (page) page.opener = null;
      try {
        const url = await host.call('materials.openSource', { source });
        (page || window).location.assign(url);
      } catch (error) {
        page?.close();
        throw error;
      }
    }
  };
  window.PeopleStudioNative = { model, browser, get, set, asBlob, importImage, imageURL, submit, wait, record, remember,
    transaction, settingsDialog, materials };
})();
