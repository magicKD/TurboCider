'use strict';
(()=>{
  const A = window.APP;
  if (!A) return;
  const desktop = window.PeopleStudioNative;
  const asBlob = value => desktop ? desktop.asBlob(value) : value;
  const $ = s => document.querySelector(s), $$ = s => document.querySelectorAll(s), esc = A.esc, toast = A.toast;
  const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
  const p2 = n => String(n).padStart(2, '0');
  const stamp = d => `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-${p2(d.getHours())}${p2(d.getMinutes())}`;
  const PROV_NAME = { openai: 'OpenAI', gemini: 'Gemini', seedream: '豆包', custom: '自定义' };
  const providerName = r => r.prov === 'local' ? (r.model || '本机引擎') : (PROV_NAME[r.prov] || r.prov);

  let L = [];
  let filterMode = 'all';
  let filterQuery = '';
  const sel = new Set();
  let curIndex = -1;
  let blobUrls = [];
  let lbUrls = [];
  let opener = null;

  const visibleRecords = () => {
    const words = filterQuery.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return L.filter(r => {
      if (filterMode === 'tryon' && r.mode == 'gen') return false;
      if (filterMode === 'gen' && r.mode != 'gen') return false;
      if (!words.length) return true;
      const hay = [r.label, r.prompt, r.orig, r.model, providerName(r)].filter(Boolean).join(' ').toLowerCase();
      return words.every(w => hay.includes(w));
    });
  };

  const U = b => {
    if (!b) return '';
    if (typeof b === 'string') return b;
    const u = URL.createObjectURL(b);
    blobUrls.push(u);
    return u;
  };
  function copyRecord(r,text){
    if(r.inputs)return window.WB.copyStudioBrief(text,r.inputs,r.inputs.lang||'zh');
    return A.copy(text,'提示词');
  }
  const cleanupUrls = () => {
    blobUrls.forEach(u => URL.revokeObjectURL(u));
    blobUrls = [];
  };
  const cleanupLbUrls = () => {
    lbUrls.forEach(u => URL.revokeObjectURL(u));
    lbUrls = [];
  };

  // Database access
  let dbp;
  const getDB = () => dbp || (dbp = new Promise((res, rej) => {
    const q = indexedDB.open('lk-studio', 1);
    q.onupgradeneeded = () => q.result.createObjectStore('h', { keyPath: 't' });
    q.onsuccess = () => res(q.result);
    q.onerror = () => rej(q.error);
  }));

  const tx = async (mode, fn) => {
    if (window.WB?.db?.tx) return window.WB.db.tx(mode, fn);
    const d = await getDB();
    return new Promise((res, rej) => {
      const t = d.transaction('h', mode), s = t.objectStore('h'), q = fn(s);
      t.oncomplete = () => res(q && q.result);
      t.onerror = () => rej(t.error);
    });
  };

  const getRecords = async () => {
    const all = await tx('readonly', s => s.getAll());
    return all.sort((a, b) => b.t - a.t);
  };

  const delRecord = async t => {
    if (window.WB?.db?.del) {
      await window.WB.db.del(t);
      if (window.WB.db.refreshHist) window.WB.db.refreshHist();
    } else {
      await tx('readwrite', s => s.delete(t));
    }
  };

  const putRecord = async rec => {
    if (window.WB?.db?.put) {
      await window.WB.db.put(rec);
      if (window.WB.db.refreshHist) window.WB.db.refreshHist();
    } else {
      await tx('readwrite', s => s.put(rec));
    }
  };

  const clearAll = async () => {
    if (window.WB?.db?.clear) {
      await window.WB.db.clear();
      if (window.WB.db.refreshHist) window.WB.db.refreshHist();
    } else {
      await tx('readwrite', s => s.clear());
    }
  };

  function updateToolbar() {
    const V = visibleRecords();
    const total = L.length;
    const shown = V.length;
    const cntEl = $('#histCount');
    if (cntEl) cntEl.innerHTML = total ? (shown === total ? `<b>${total}</b> 条记录` : `<b>${shown}</b> / ${total} 条记录`) : '';

    const selN = sel.size;
    const selNEl = $('#histSelN');
    if (selNEl) {
      selNEl.hidden = selN === 0;
      if (selN > 0) selNEl.innerHTML = `已选 <b>${selN}</b> 张`;
    }

    const visSel = V.filter(r => sel.has(r.t)).length;
    const selAllBox = $('#histSelAll');
    if (selAllBox) {
      selAllBox.checked = shown > 0 && visSel === shown;
      selAllBox.indeterminate = visSel > 0 && visSel < shown;
      selAllBox.disabled = shown === 0;
    }

    const zipBtn = $('#histZip');
    if (zipBtn) zipBtn.disabled = selN === 0;
    const expBtn = $('#histExp');
    if (expBtn) expBtn.disabled = selN === 0;
    const clrBtn = $('#histClr');
    if (clrBtn) clrBtn.disabled = total === 0;

    $$('#histFilter [data-hf]').forEach(b => {
      const on = b.dataset.hf === filterMode;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    });

    const nHist = $('#nHist');
    if (nHist) nHist.textContent = total || '';
  }

  function renderStats() {
    const el = $('#histStats');
    if (!el) return;
    if (!L.length) {
      el.innerHTML = '';
      return;
    }
    const tryon = L.filter(r => r.mode != 'gen').length;
    const gen = L.length - tryon;
    const provCount = {};
    L.forEach(r => {
      const p = providerName(r);
      provCount[p] = (provCount[p] || 0) + 1;
    });
    const provStr = Object.entries(provCount).map(([k, v]) => `${esc(k)} ${v}`).join(' / ');
    el.innerHTML = `<span class="hist-ratio" aria-hidden="true"><i style="width:${100*tryon/L.length}%"></i></span>
      <span class="hist-key"><i></i><span>试穿</span> <b>${tryon}</b></span>
      <span class="hist-key gen"><i></i><span>生图</span> <b>${gen}</b></span>
      <span class="hist-providers">${provStr}</span>`;
  }

  function renderGrid() {
    cleanupUrls();
    const grid = $('#histGrid');
    const empty = $('#histEmpty');
    const bar = $('#histBar');
    const stats = $('#histStats');
    if (!grid) return;

    if (!L.length) {
      grid.innerHTML = '';
      if (empty) empty.hidden = false;
      if (bar) bar.hidden = true;
      if (stats) stats.hidden = true;
      updateToolbar();
      return;
    }

    if (empty) empty.hidden = true;
    if (bar) bar.hidden = false;
    if (stats) stats.hidden = false;

    const V = visibleRecords();
    if (!V.length) {
      grid.innerHTML = `<div class="hist-nomatch"><p>没有找到匹配的生成记录</p><button type="button" class="sm" id="histResetFilter">显示全部</button></div>`;
      $('#histResetFilter').onclick = () => {
        filterMode = 'all';
        filterQuery = '';
        const q = $('#histQ');
        if (q) q.value = '';
        renderGrid();
      };
      renderStats();
      updateToolbar();
      return;
    }

    grid.innerHTML = V.map((r, i) => {
      const u = U(r.after);
      const isSel = sel.has(r.t);
      const d = new Date(r.t);
      const timeStr = `${d.getMonth() + 1}/${d.getDate()} ${p2(d.getHours())}:${p2(d.getMinutes())}`;
      const prov = providerName(r);
      const label = r.label || (r.prompt ? r.prompt.slice(0, 40) : (r.mode == 'gen' ? '自由生图' : '虚拟试穿'));
      const sub = [timeStr, prov, r.model !== prov && r.model].filter(Boolean).join(' · ');

      return `<article class="hist-card${isSel ? ' selected' : ''}" data-t="${r.t}" data-i="${i}">
        <div class="hph">
          <img src="${u}" alt="${esc(label)}" loading="lazy">
          <button class="hopen" type="button" aria-label="查看大图：${esc(label)}"></button>
          <input type="checkbox" class="hsel" data-t="${r.t}" aria-label="选择此记录" ${isSel ? 'checked' : ''}>
          <div class="hcbtns">
            <button class="icon" data-act="dl" title="下载图片" aria-label="下载图片">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            </button>
            <button class="icon" data-act="regen" title="用相同设置重新生成" aria-label="重新生成">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.19"/></svg>
            </button>
            <button class="icon" data-act="del" title="删除记录" aria-label="删除记录">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </div>
          <span class="hmode">${r.mode == 'gen' ? '生图' : '试穿'}</span>
        </div>
        <div class="hinfo">
          <h3 title="${esc(label)}">${esc(label)}</h3>
          <div class="sub" title="${esc(sub)}">${esc(sub)}</div>
        </div>
      </article>`;
    }).join('');

    renderStats();
    updateToolbar();
  }

  function dlOne(r) {
    if (!r?.after) return;
    const ts = stamp(new Date(r.t));
    const name = `posewear-${r.mode == 'gen' ? 'gen' : 'tryon'}-${ts}.png`;
    if (desktop && r.tcOutput) {
      void window.TurboCider.media.save(r.tcOutput).catch(error => toast(error.message));
    } else {
      window.X.dl(r.after, name);
      toast('已开始下载图片');
    }
  }

  async function delOne(r) {
    if (!r) return;
    await delRecord(r.t);
    sel.delete(r.t);
    toast('已删除 1 条记录', async () => {
      await putRecord(r);
      await refresh();
    });
    await refresh();
  }

  async function regen(r) {
    if (!r || !window.WB) return;
    try {
      const missing = await WB.restoreRecord(r);
      A.lbClose();
      await A.setTab('studio');
      toast(['已载入历史提示词和可用参考图', ...missing].join('；'));
    } catch {
      toast('历史选材未能载入，请重试');
    }
  }

  async function asPerson(r) {
    if (!r?.after || !window.WB) return;
    await WB.show();
    const W = WB._W;
    if (W) {
      let w = 0, h = 0;
      const blob = await asBlob(r.after);
      const bmp = await createImageBitmap(blob).catch(() => null);
      if (bmp) { w = bmp.width; h = bmp.height; bmp.close(); }
      W.mode = 'tryon';
      W.view = 'board';
      W.persons = [{
        blob,
        tcMediaID: r.tcOutput,
        url: typeof r.after === 'string' ? r.after : URL.createObjectURL(blob),
        w: w || undefined,
        h: h || undefined
      }];
    }
    A.lbClose(true);
    A.setTab('studio');
    toast('已将结果设为新的人物照片');
  }

  function openLb(index, srcCard) {
    const V = visibleRecords();
    if (index < 0 || index >= V.length) return;
    curIndex = index;
    const r = V[curIndex];
    if (!r) return;

    const lb = $('#lb');
    if (!lb) return;
    A.lbResetView?.(true);
    opener = srcCard?.querySelector('.hopen');
    lb.hidden = false;
    document.body.style.overflow = 'hidden';

    drawLb();

    if (!motionPreference.matches && srcCard) {
      const ph = srcCard.querySelector('.hph');
      if (ph) {
        const rect = ph.getBoundingClientRect();
        $('.lb-bg').animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
        const el = $('.lb-box');
        const dx = rect.left + rect.width / 2 - innerWidth / 2;
        const dy = rect.top + rect.height / 2 - innerHeight / 2;
        const s = Math.max(rect.width / el.offsetWidth, 0.08);
        el.animate([
          { transform: `translate(${dx}px,${dy}px) scale(${s})`, opacity: 0.2, borderRadius: '40px' },
          { transform: 'none', opacity: 1, borderRadius: '20px' }
        ], { duration: 520, easing: 'cubic-bezier(.16,1,.3,1)', fill: 'both' });
      }
    }
    setTimeout(() => $('.lb-x')?.focus({ preventScroll: true }), 60);
  }

  function drawLbRefs(r, container) {
    if (!container || !r.inputs) return;
    const inputs = r.inputs;
    const refs = [
      ...(inputs.persons || []).filter(x => x.k || x.blob),
      ...(inputs.garments || []),
      ...(inputs.poses || []),
      ...(inputs.exprs || []),
      ...(inputs.extraRefs || [])
    ];
    if (!refs.length) return;
    const keys = refs.filter(x => x.k).map(x => x.k);
    const paint = () => {
      if (!container.isConnected || visibleRecords()[curIndex] !== r) return;
      const cards = refs.map(ref => {
        if (ref.blob) {
          const u = URL.createObjectURL(ref.blob);
          lbUrls.push(u);
          return `<span class="hlb-ref" title="${esc(ref.name || '本地上传')}"><img src="${u}" alt=""></span>`;
        }
        const it = A.ITEMS[ref.k];
        if (!it) return '';
        const src = it.angles ? A.ath(it.angles[ref.ai ?? it.ai].t) : A.thumb(it);
        const ang = it.angles?.[ref.ai ?? it.ai]?.zh;
        const label = [A.title(it), ang].filter(Boolean).join(' · ');
        return `<a class="hlb-ref" href="#${esc(it.type)}/${esc(it.id)}" data-href-ref title="${esc(label)}"><img src="${esc(src)}" alt="${esc(label)}"${it.ref ? ` referrerpolicy="${esc(it.ref)}"` : ''}></a>`;
      }).filter(Boolean);
      if (!cards.length) return;
      container.innerHTML = `<small class="hlb-refs-h">所用参考图</small><div class="hlb-refs-row">${cards.join('')}</div>`;
      container.hidden = false;
    };
    if (keys.every(k => A.ITEMS[k])) paint();
    else A.needKeys(keys).then(paint).catch(() => {});
  }

  function drawLb() {
    const V = visibleRecords();
    const r = V[curIndex];
    if (!r) return;
    cleanupLbUrls();

    const img = $('#lbImg');
    const box = $('.lb-img');
    if (box) box.classList.remove('loading', 'ext', 'fail');
    const afterUrl = typeof r.after === 'string' ? r.after : URL.createObjectURL(r.after);
    if (typeof r.after !== 'string') lbUrls.push(afterUrl);

    if (img) {
      delete img.dataset.src;
      img.onerror = null;
      img.classList.remove('lo');
      img.removeAttribute('srcset');
      img.src = afterUrl;
      img.alt = r.label || (r.mode == 'gen' ? '自由生图' : '虚拟试穿');
    }

    let cmpHtml = '';
    if (r.before && r.mode != 'gen') {
      const beforeUrl = typeof r.before === 'string' ? r.before : URL.createObjectURL(r.before);
      if (typeof r.before !== 'string') lbUrls.push(beforeUrl);
      cmpHtml = `
      <div class="hlb-cmp" style="--x:50%">
        <img src="${afterUrl}" alt="生成后">
        <div class="hlb-bef"><img src="${beforeUrl}" alt="原图"></div>
        <div class="hlb-hdl"><span></span></div>
        <input type="range" min="0" max="100" value="50" aria-label="前后对比">
        <em class="l">原图</em><em class="r">生成后</em>
      </div>`;
    }

    const d = new Date(r.t);
    const dateStr = `${d.getFullYear()}/${p2(d.getMonth() + 1)}/${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}`;
    const prov = providerName(r);

    const side = $('#lbSide');
    if (!side) return;
    const cpSvg = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M15 5V4H5v11h1"/></svg>';
    side.innerHTML = `
      <div class="mu">生成记录 · 第 ${curIndex + 1} / ${V.length} 项</div>
      <h2>${r.mode == 'gen' ? '自由生图' : '虚拟试穿'} · ${esc(prov)}</h2>
      <div class="mu">${dateStr} · 模型 ${esc(r.model || '默认')}${r.label ? ' · ' + esc(r.label) : ''}</div>
      ${cmpHtml}
      <div class="hlb-refs" id="hlbRefs" hidden></div>
      <div class="pbox" style="margin-top:14px">
        <div class="pbox-h"><small>提示词 · PROMPT${r.neutral ? '（已减少误拦截）' : ''}</small>${r.prompt ? `<button class="sm pbox-cp pri" data-hlb="cp">${cpSvg}<span>复制提示词</span></button>` : ''}</div>
        <p>${esc(r.prompt || '（无提示词）')}</p>
      </div>
      ${r.orig ? `<details class="raw" style="margin-top:10px"><summary>原始提示词</summary><pre>${esc(r.orig)}</pre><button class="sm ghost" data-hlb="cporig">复制</button></details>` : ''}
      <div class="acts" style="margin-top:16px">
        <button class="pri" data-hlb="dl">下载图片</button>
        <button data-hlb="regen">用此设置重新生成</button>
        ${r.mode != 'gen' && r.after ? '<button data-hlb="person">用作人物照片</button>' : ''}
        <button data-hlb="del" class="ghost hlb-del">删除此记录</button>
      </div>
      <p class="mu lb-keys"><kbd>←</kbd><kbd>→</kbd>切换记录　<kbd>C</kbd>复制提示词　<kbd>Esc</kbd>关闭</p>
    `;

    drawLbRefs(r, $('#hlbRefs'));

    const slider = side.querySelector('.hlb-cmp input[type=range]');
    if (slider) {
      slider.oninput = e => {
        const wrap = e.target.closest('.hlb-cmp');
        if (wrap) wrap.style.setProperty('--x', e.target.value + '%');
      };
    }
  }

  function stepLb(delta) {
    const V = visibleRecords();
    if (!V.length) return;
    curIndex = (curIndex + delta + V.length) % V.length;
    drawLb();
    if (!motionPreference.matches) {
      const img = $('#lbImg');
      if (img) {
        img.animate([
          { opacity: 0.3, transform: `translateX(${delta * 30}px)` },
          { opacity: 1, transform: 'none' }
        ], { duration: 320, easing: 'cubic-bezier(.16,1,.3,1)' });
      }
    }
  }

  // Intercept prev / next buttons when in history view
  $('.prev')?.addEventListener('click', e => {
    if (A.st.tab == 'history' && !$('#lb')?.hidden) {
      e.stopImmediatePropagation();
      stepLb(-1);
    }
  }, true);

  $('.next')?.addEventListener('click', e => {
    if (A.st.tab == 'history' && !$('#lb')?.hidden) {
      e.stopImmediatePropagation();
      stepLb(1);
    }
  }, true);

  // Intercept keyboard shortcuts when in history lightbox view
  addEventListener('keydown', e => {
    if (A.st.tab == 'history' && !$('#lb')?.hidden) {
      if ($('#helpBox:not([hidden]),#ageBox:not([hidden]),#privBox:not([hidden])')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const ae = document.activeElement;
      if (ae && /^(INPUT|SELECT|TEXTAREA)$/.test(ae.tagName)) return;
      const k = (e.key || '').toLowerCase();
      if (e.key == 'ArrowLeft') {
        e.preventDefault();
        e.stopImmediatePropagation();
        stepLb(-1);
      } else if (e.key == 'ArrowRight') {
        e.preventDefault();
        e.stopImmediatePropagation();
        stepLb(1);
      } else if (e.key == 'ArrowUp' || e.key == 'ArrowDown') {
        e.preventDefault();
        e.stopImmediatePropagation();
      } else if (k == 'c') {
        e.preventDefault();
        e.stopImmediatePropagation();
        const r = visibleRecords()[curIndex];
        if (r?.prompt) copyRecord(r,r.prompt);
      } else if (k == 'f' || k == 'a' || k == 's' || k == 'z' || k == 'h' || k == 'g' || k == 'v') {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    }
  }, true);

  // Hook side actions
  $('#lbSide')?.addEventListener('click', async e => {
    if (A.st.tab != 'history') return;
    if (e.target.closest('[data-href-ref]')) {
      A.lbClose(true);
      return;
    }
    const b = e.target.closest('[data-hlb]');
    if (!b) return;
    const act = b.dataset.hlb;
    const r = visibleRecords()[curIndex];
    if (!r) return;

    if (act == 'cp') {
      copyRecord(r,r.prompt);
      b.classList.add('copied');clearTimeout(b._ct);b._ct=setTimeout(()=>b.classList.remove('copied'),1200);
    }
    if (act == 'cporig') copyRecord(r,r.orig);
    if (act == 'dl') dlOne(r);
    if (act == 'regen') {
      A.lbClose(true);
      await regen(r);
    }
    if (act == 'person') {
      await asPerson(r);
    }
    if (act == 'del') {
      A.lbClose();
      await delOne(r);
    }
  });

  // Grid click interactions
  $('#histGrid')?.addEventListener('click', async e => {
    const card = e.target.closest('.hist-card');
    if (!card) return;
    const t = +card.dataset.t;
    const i = +card.dataset.i;
    const V = visibleRecords();
    const r = V.find(x => x.t == t) || V[i];
    if (!r) return;

    if (e.target.classList.contains('hsel')) {
      if (e.target.checked) {
        sel.add(t);
        card.classList.add('selected');
      } else {
        sel.delete(t);
        card.classList.remove('selected');
      }
      updateToolbar();
      return;
    }

    const btn = e.target.closest('button[data-act]');
    if (btn) {
      e.stopPropagation();
      const act = btn.dataset.act;
      if (act == 'dl') dlOne(r);
      if (act == 'del') await delOne(r);
      if (act == 'regen') await regen(r);
      return;
    }

    const ph = e.target.closest('.hph');
    if (ph) {
      openLb(i, card);
    }
  });

  // Filter & Search events
  $('#histFilter')?.addEventListener('click', e => {
    const b = e.target.closest('[data-hf]');
    if (!b || b.dataset.hf === filterMode) return;
    filterMode = b.dataset.hf;
    renderGrid();
  });
  let hqTimer;
  $('#histQ')?.addEventListener('input', e => {
    clearTimeout(hqTimer);
    hqTimer = setTimeout(() => {
      filterQuery = e.target.value;
      renderGrid();
    }, 120);
  });

  // Toolbar events
  $('#histSelAll')?.addEventListener('change', e => {
    const V = visibleRecords();
    if (e.target.checked) {
      V.forEach(r => sel.add(r.t));
    } else {
      V.forEach(r => sel.delete(r.t));
    }
    $$('#histGrid .hist-card').forEach(c => {
      const isSel = sel.has(+c.dataset.t);
      c.classList.toggle('selected', isSel);
      const cb = c.querySelector('.hsel');
      if (cb) cb.checked = isSel;
    });
    updateToolbar();
  });

  $('#histZip')?.addEventListener('click', async () => {
    const selected = L.filter(r => sel.has(r.t));
    if (!selected.length) return toast('请先勾选要下载的记录');
    if (!window.makeZip) return toast('未加载打包组件');
    const btn = $('#histZip');
    btn.disabled = true;
    btn.textContent = '打包中…';
    try {
      const ts = stamp(new Date());
      const files = await Promise.all(selected.map(async (r, i) => ({
        name: `posewear-${r.mode == 'gen' ? 'gen' : 'tryon'}-${stamp(new Date(r.t))}-${String(i + 1).padStart(2, '0')}.png`,
        blob: await asBlob(r.after)
      })));
      const zipBlob = await window.makeZip(files);
      window.X.dl(zipBlob, `posewear-history-${ts}.zip`);
      toast(`已打包下载 ${files.length} 张图片`);
    } catch (err) {
      toast('打包失败，请重试');
    } finally {
      btn.disabled = false;
      btn.textContent = '打包下载';
    }
  });

  $('#histExp')?.addEventListener('click', async () => {
    const selected = L.filter(r => sel.has(r.t));
    if (!selected.length) return toast('请先勾选要导出的记录');
    if (!window.makeZip) return toast('未加载打包组件');
    const btn = $('#histExp');
    btn.disabled = true;
    btn.textContent = '导出中…';
    try {
      const ts = stamp(new Date());
      const metaList = [];
      const files = [];
      for (const [i, r] of selected.entries()) {
        const fileName = `posewear-${r.mode == 'gen' ? 'gen' : 'tryon'}-${stamp(new Date(r.t))}-${String(i + 1).padStart(2, '0')}.png`;
        files.push({ name: fileName, blob: await asBlob(r.after) });
        metaList.push({
          time: new Date(r.t).toISOString(),
          timestamp: r.t,
          mode: r.mode || 'tryon',
          provider: r.prov,
          model: r.model,
          prompt: r.prompt,
          originalPrompt: r.orig || null,
          neutral: !!r.neutral,
          label: r.label || '',
          fileName
        });
      }
      const metaBlob = new Blob([JSON.stringify({
        app: '衣姿 PoseWear',
        type: 'generation-history',
        exportTime: new Date().toISOString(),
        totalCount: selected.length,
        records: metaList
      }, null, 2)], { type: 'application/json' });
      files.push({ name: 'metadata.json', blob: metaBlob });

      const zipBlob = await window.makeZip(files);
      window.X.dl(zipBlob, `posewear-export-${ts}.zip`);
      toast(`已导出 ${selected.length} 条记录（含元数据）`);
    } catch (err) {
      toast('导出失败，请重试');
    } finally {
      btn.disabled = false;
      btn.textContent = '导出 ZIP';
    }
  });

  $('#histClr')?.addEventListener('click', async () => {
    if (!L.length) return toast('还没有生成记录');
    const oldList = L.slice();
    await clearAll();
    sel.clear();
    await refresh();
    toast(`已清空 ${oldList.length} 条记录`, async () => {
      for (const r of oldList) await putRecord(r);
      await refresh();
    });
  });

  $('#histEmpty button[data-go="studio"]')?.addEventListener('click', () => {
    A.setTab('studio');
  });

  async function refresh() {
    try {
      L = await getRecords();
    } catch {
      $('#histEmpty').hidden = true;
      $('#histBar').hidden = true;
      $('#histStats').hidden = true;
      const grid = $('#histGrid');
      grid.innerHTML = '<p>生成记录读取失败。<button type="button" class="sm">重新加载</button></p>';
      grid.querySelector('button').onclick = refresh;
      return;
    }
    const currentKeys = new Set(L.map(r => r.t));
    for (const k of sel) {
      if (!currentKeys.has(k)) sel.delete(k);
    }
    renderGrid();
  }

  async function show() {
    await refresh();
  }

  function closed(restoreFocus) {
    cleanupLbUrls();
    if (restoreFocus) {
      const target = opener?.isConnected ? opener : $('#histGrid .hopen, #histEmpty button');
      target?.focus({preventScroll:true});
    }
    opener = null;
  }
  window.HIS = { show, refresh, closed, getList: () => L };

  // Init badge count on load
  getRecords().then(list => {
    L = list;
    const nHist = $('#nHist');
    if (nHist) nHist.textContent = L.length || '';
    if (document.body.classList.contains('histview') || location.hash == '#history') {
      show();
    }
  }).catch(() => {});
})();
