'use strict';
// Gallery enhancements: micro-interactions, batch selection, search suggestions,
// lightbox similar recommendations, image download/copy, and PWA update notice.
(() => {
  const A = window.APP;
  const C = window.PoseWearCatalog;
  if (!A || !C) return;
  const { $, $$, readLS: LS, writeLS: SV } = C;
  const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
  const EZ = 'cubic-bezier(.16,1,.3,1)';
  const grid = $('#grid');

  const dl = (blob, name) => {
    if (window.PeopleStudioNative) {
      return window.TurboCider.media.saveBlob(blob, name).catch(error => { A.toast(error.message); return false; });
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 3000);
  };

  const X = (window.X = { dl });

  // ---------- Micro-interactions ----------
  function burst(el) {
    if (motionPreference.matches || !el) return;
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    for (let i = 0; i < 8; i++) {
      const p = document.createElement('i');
      p.className = 'pt';
      p.style.cssText = `left:${cx}px;top:${cy}px;background:${i % 2 ? '#ff2442' : '#ff8fa3'}`;
      document.body.append(p);
      const angle = (i / 8) * Math.PI * 2;
      const dist = 18 + Math.random() * 10;
      p.animate(
        [
          { transform: 'translate(-50%,-50%) scale(1)', opacity: 1 },
          { transform: `translate(calc(-50% + ${Math.cos(angle) * dist}px),calc(-50% + ${Math.sin(angle) * dist}px)) scale(.2)`, opacity: 0 },
        ],
        { duration: 520, easing: EZ }
      ).finished.then(() => p.remove());
    }
  }

  function pop(el) {
    if (!motionPreference.matches && el) {
      el.animate(
        [{ transform: 'scale(1)' }, { transform: 'scale(1.32)' }, { transform: 'scale(.92)' }, { transform: 'scale(1)' }],
        { duration: 460, easing: EZ }
      );
    }
  }

  function fly(img, target) {
    if (motionPreference.matches || !img || !target) return;
    const a = img.getBoundingClientRect();
    const b = target.getBoundingClientRect();
    if (!a.width) return;
    const clone = img.cloneNode();
    clone.className = 'flyimg';
    clone.style.cssText = `left:${a.left}px;top:${a.top}px;width:${a.width}px;height:${a.height}px`;
    document.body.append(clone);
    const dx = b.left + b.width / 2 - (a.left + a.width / 2);
    const dy = b.top + b.height / 2 - (a.top + a.height / 2);
    clone.animate(
      [
        { transform: 'none', opacity: 1, borderRadius: '14px' },
        { transform: `translate(${dx * 0.6}px,${dy * 0.6 - 40}px) scale(.35)`, opacity: 0.9, offset: 0.6 },
        { transform: `translate(${dx}px,${dy}px) scale(.06)`, opacity: 0.2, borderRadius: '50%' },
      ],
      { duration: 640, easing: 'cubic-bezier(.5,0,.2,1)' }
    ).finished.then(() => {
      clone.remove();
      pop(target);
    });
  }

  X.fly = fly;
  X.pop = pop;

  const favTab = () => $('#personalMenu summary');

  // ---------- Batch Selection Mode ----------
  const selectedKeys = new Set();
  X.batchPicked = k => selectedKeys.has(k);
  X.filterBatch = items => {
    if (!selectedKeys.size) return;
    const valid = new Set(items.map(it => it.k));
    for (const k of selectedKeys) {
      if (!valid.has(k)) selectedKeys.delete(k);
    }
    updateBatchBar();
  };

  function togglePick(k) {
    if (selectedKeys.has(k)) selectedKeys.delete(k);
    else selectedKeys.add(k);
    updateBatchBar();
  }

  grid.addEventListener('click', e => {
    const btn = e.target.closest('[data-a]');
    const card = e.target.closest('.card');
    if (!card) return;
    const key = card.dataset.show || card.dataset.k;

    if (document.body.classList.contains('batch')) {
      e.preventDefault();
      e.stopPropagation();
      togglePick(key);
      grid.querySelectorAll('.card').forEach(c =>
        c.classList.toggle('picked', selectedKeys.has(c.dataset.show || c.dataset.k))
      );
      return;
    }

    if (btn && btn.dataset.a === 'fav') {
      const adding = !A.st.fav.has(key);
      const img = card.querySelector('img');
      requestAnimationFrame(() => {
        const nextBtn = grid.querySelector(`.card[data-k="${CSS.escape(card.dataset.k)}"] [data-a=fav]`);
        if (adding) {
          pop(nextBtn);
          burst(nextBtn);
          fly(img, favTab());
        }
      });
    }

    if (btn && btn.dataset.a === 'sel' && !$('#trayFab').hidden && !A.st.sel.includes(key)) {
      fly(card.querySelector('img'), $('#trayFab'));
    }
  }, true);

  const toolbar = $('.tbar');
  const batchBtn = document.createElement('button');
  batchBtn.className = 'fbtn';
  batchBtn.id = 'batchBtn';
  batchBtn.innerHTML =
    '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="7" height="7" rx="2"/><rect x="13" y="4" width="7" height="7" rx="2"/><rect x="4" y="13" width="7" height="7" rx="2"/><path d="m14 16.5 2 2 4-4"/></svg>批量';
  toolbar.querySelector('.sp').after(batchBtn);

  const batchBar = document.createElement('div');
  batchBar.id = 'bbar';
  batchBar.hidden = true;
  batchBar.innerHTML =
    '<b id="bbN">已选 0</b>' +
    '<button data-b="all">全选</button>' +
    '<button data-b="fav">收藏</button>' +
    '<button data-b="unfav">取消收藏</button>' +
    '<button data-b="zip">打包下载</button>' +
    '<button data-b="stu">去试穿</button>' +
    '<button data-b="cp">复制</button>' +
    '<button data-b="x" class="pri">完成</button>';
  document.body.append(batchBar);

  function setBatchMode(on) {
    document.body.classList.toggle('batch', on);
    batchBar.querySelector('[data-b=fav]').hidden = A.st.tab === 'fav';
    batchBar.querySelector('[data-b=unfav]').hidden = A.st.tab !== 'fav';
    batchBar.hidden = !on;
    batchBtn.classList.toggle('on', on);
    batchBtn.setAttribute('aria-pressed', String(on));
    if (!on) {
      selectedKeys.clear();
      $$('.card.picked').forEach(c => c.classList.remove('picked'));
    }
    updateBatchBar();
  }

  function updateBatchBar() {
    $('#bbN').textContent = '已选 ' + selectedKeys.size;
    batchBar.querySelectorAll('[data-b]:not([data-b=x]):not([data-b=all])').forEach(b => {
      b.disabled = !selectedKeys.size;
    });
  }

  batchBtn.onclick = () => setBatchMode(!document.body.classList.contains('batch'));

  batchBar.onclick = async e => {
    const target = e.target.closest('[data-b]');
    if (!target) return;
    const items = [...selectedKeys].map(k => A.ITEMS[k]).filter(Boolean);
    const mode = target.dataset.b;

    if (mode === 'x') return setBatchMode(false);
    if (mode === 'all') {
      A.st.list.forEach(it => selectedKeys.add(A.cardFace(it).k));
      $$('#grid .card').forEach(c => c.classList.add('picked'));
      return updateBatchBar();
    }
    if (mode === 'fav') {
      let added = 0;
      items.forEach(it => {
        if (!A.st.fav.has(it.k)) {
          A.st.fav.add(it.k);
          added++;
        }
      });
      SV('fav', [...A.st.fav]);
      $('#nFav').textContent = A.st.fav.size;
      pop(favTab());
      A.toast(`已收藏 ${added} 项`);
      setBatchMode(false);
      A.apply();
      return;
    }
    if (mode === 'unfav') {
      items.forEach(it => A.st.fav.delete(it.k));
      SV('fav', [...A.st.fav]);
      const removedCount = items.length;
      const removedKeys = items.map(i => i.k);
      setBatchMode(false);
      A.apply();
      $('#nFav').textContent = A.st.fav.size || '';
      A.toast(`已取消 ${removedCount} 项收藏`, () => {
        removedKeys.forEach(k => A.st.fav.add(k));
        SV('fav', [...A.st.fav]);
        $('#nFav').textContent = A.st.fav.size;
        A.apply();
      });
      return;
    }
    if (mode === 'zip') {
      const local = items.filter(it => !it.ext && (!it.materialSource || it.materialDownloadable));
      if (!local.length) return A.toast(items.length ? '这些素材支持按需选用。可打开单张图片保存。' : '先选几张');
      if (!window.makeZip) return;
      target.disabled = true;
      target.textContent = '打包中…';
      try {
        const files = [];
        for (const it of local.slice(0, 60)) {
          const resp = await fetch(A.big(it));
          if (!resp.ok) throw Error(resp.status);
          const blob = await resp.blob();
          const ext = blob.type === 'image/png' ? 'png' : blob.type === 'image/jpeg' ? 'jpg' : 'webp';
          files.push({ name: `posewear-${it.type}-${it.id}.${ext}`, blob });
        }
        window.X.dl(await makeZip(files), `posewear-${A.st.tab}-${files.length}.zip`);
        A.toast(
          `已打包 ${files.length} 张` +
            (local.length > 60 ? '（单次最多 60 张）' : '') +
            (local.length < items.length ? '，外链图片请到原站查看' : '')
        );
      } catch {
        A.toast('打包失败，请重试');
      }
      target.disabled = false;
      target.textContent = '打包下载';
      return;
    }
    if (mode === 'cp') {
      A.copyBrief(items.map(it => A.prompt(it, A.st.lang)).join('\n'), items, A.st.lang, `${items.length} 条提示词`);
      return;
    }
    if (mode === 'stu') {
      if (await A.sendToStudio(items, true)) setBatchMode(false);
    }
  };

  addEventListener('keydown', e => {
    if (e.key === 'Escape' && document.body.classList.contains('batch')) setBatchMode(false);
  });
  X.closeBatch = () => {
    if (document.body.classList.contains('batch')) setBatchMode(false);
  };

  // ---------- Search Suggestions ----------
  const searchInput = $('#q');
  const suggestBox = document.createElement('div');
  suggestBox.id = 'qsug';
  suggestBox.hidden = true;
  searchInput.parentNode.append(suggestBox);

  function renderSuggestions() {
    const history = LS('qh') || [];
    const value = searchInput.value.trim();
    const hot = (A.suggestions[A.st.tab] || []).filter(x => !value || x.includes(value));
    const recentHtml =
      history.length && !value
        ? `<div class="sgh"><small>最近搜索</small><button data-clr>清除</button></div>` +
          `<div class="sgc">${history.map(x => `<button data-q="${A.esc(x)}">${A.esc(x)}</button>`).join('')}</div>`
        : '';
    const hotHtml = hot.length
      ? `<div class="sgh"><small>热门</small></div>` +
        `<div class="sgc">${hot.map((x, i) => `<button data-q="${x}"><em>${i + 1}</em>${x}</button>`).join('')}</div>`
      : '';
    suggestBox.innerHTML = recentHtml + hotHtml;
    suggestBox.hidden = !suggestBox.innerHTML;
  }

  const saveSearchQuery = () => {
    const value = searchInput.value.trim();
    if (value.length < 1) return;
    SV('qh', [value, ...(LS('qh') || []).filter(x => x !== value)].slice(0, 8));
  };

  searchInput.addEventListener('focus', renderSuggestions);
  searchInput.addEventListener('input', () => {
    renderSuggestions();
    clearTimeout(searchInput._h);
    searchInput._h = setTimeout(saveSearchQuery, 1500);
  });
  searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      saveSearchQuery();
      suggestBox.hidden = true;
      searchInput.blur();
    }
  });
  searchInput.parentNode.addEventListener('focusout', e => {
    if (!searchInput.parentNode.contains(e.relatedTarget)) suggestBox.hidden = true;
  });
  searchInput.parentNode.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !suggestBox.hidden) {
      e.stopPropagation();
      searchInput.focus();
      suggestBox.hidden = true;
    }
  });
  suggestBox.addEventListener('mousedown', e => e.preventDefault());
  suggestBox.onclick = e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    if (btn.dataset.clr != null) {
      SV('qh', []);
      return renderSuggestions();
    }
    searchInput.focus();
    searchInput.value = btn.dataset.q;
    searchInput.dispatchEvent(new Event('input'));
    saveSearchQuery();
    suggestBox.hidden = true;
  };

  // ---------- Result Count Animation ----------
  const countEl = $('#count');
  new MutationObserver(() => {
    if (!motionPreference.matches) {
      countEl.animate([{ opacity: 0.2, transform: 'translateY(4px)' }, { opacity: 1, transform: 'none' }], {
        duration: 360,
        easing: EZ,
      });
    }
  }).observe(countEl, { childList: true, characterData: true, subtree: true });

  // ---------- Lightbox Similar Recommendations / Download / Copy Image ----------
  const COLOR_CHARS = /[黑白红粉蓝灰绿黄紫金银棕米]/g;
  X.lb = it => {
    const list = A.st.list;
    const cols = new Set((it.zh || '').match(COLOR_CHARS) || []);
    const score = candidate =>
      it.type === 'portrait'
        ? (candidate.len === it.len ? 2 : 0) + (candidate.st && candidate.st === it.st ? 4 : 0) + (candidate.sec === it.sec ? 1 : 0)
        : (candidate.cat === it.cat ? 3 : 0) +
          (candidate.line && candidate.line === it.line ? 1 : 0) +
          ((candidate.zh || '').match(COLOR_CHARS) || []).filter(c => cols.has(c)).length;

    const similar = list
      .map((item, index) => [item, index])
      .filter(([item]) => item.k !== it.k && item.type === it.type)
      .map(([item, index]) => [item, index, score(item) + Math.random() * 0.5])
      .sort((a, b) => b[2] - a[2])
      .slice(0, 8);

    const slot = $('#lbSim');
    if (!slot) return;
    slot.innerHTML = similar.length
      ? `<small class="simh">相似推荐</small><div class="sim">${similar
          .map(
            ([x, i]) =>
              `<button data-si="${i}" title="${A.esc(A.title(x))}"><img src="${A.esc(A.thumb(x))}" alt="${A.esc(A.title(x))}" loading="lazy"${x.ref ? ` referrerpolicy="${A.esc(x.ref)}"` : ''}></button>`
          )
          .join('')}</div>`
      : '';
  };

  async function fetchCurrentBlob() {
    const it = A.previewItem();
    const resp = await fetch(A.big(it));
    if (!resp.ok) throw Error(resp.status);
    return [await resp.blob(), it];
  }

  $('#lbSide').addEventListener('click', async e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    if (btn.dataset.si) {
      A.lbOpen(+btn.dataset.si);
      if (!motionPreference.matches) {
        $('#lbImg').animate([{ opacity: 0, transform: 'scale(.97)' }, { opacity: 1, transform: 'none' }], {
          duration: 340,
          easing: EZ,
        });
      }
      return;
    }
    if (btn.dataset.x === 'dlx') {
      try {
        const [blob, it] = await fetchCurrentBlob();
        const ext = blob.type === 'image/png' ? 'png' : blob.type === 'image/jpeg' ? 'jpg' : 'webp';
        await dl(blob, `posewear-${it.type}-${it.id}.${ext}`);
        A.toast('已开始下载');
      } catch {
        A.toast('下载失败');
      }
      return;
    }
    if (btn.dataset.x === 'cpimg') {
      try {
        const [blob] = await fetchCurrentBlob();
        const bitmap = await createImageBitmap(blob);
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        canvas.getContext('2d').drawImage(bitmap, 0, 0);
        const png = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': png })]);
        A.toast('图片已复制到剪贴板');
      } catch {
        A.toast('当前浏览器不支持复制图片');
      }
    }
  });

  // ---------- Random Pick Card Riffle Animation ----------
  const randBtn = $('#rand');
  if (randBtn) {
    randBtn.addEventListener('click', () => {
      if (motionPreference.matches) return;
      const cards = [...$$('#grid .card')]
        .filter(c => {
          const r = c.getBoundingClientRect();
          return r.top < innerHeight && r.bottom > 0;
        })
        .slice(0, 12);
      cards.forEach((c, i) =>
        c.animate(
          [{ transform: 'none' }, { transform: `translateY(-6px) rotate(${(i % 2 ? 1 : -1) * 1.2}deg)` }, { transform: 'none' }],
          { duration: 420, delay: i * 22, easing: EZ }
        )
      );
    }, true);
  }

  // ---------- Service Worker Update Notice ----------
  if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      $('#noticeReload').hidden = false;
    });
    $('#noticeReload').onclick = () => location.reload();
  }
})();
