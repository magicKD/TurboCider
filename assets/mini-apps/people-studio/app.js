'use strict';
(() => {
  const C = window.PoseWearCatalog;
  if (!C) return;
  const {
    $, $$, esc, readLS: LS, writeLS: SV, shuffle,
    TABS, TNAME, PFX, KT, TL, FDEF, FLAB, COLS, COL_SWATCH, ITEMK, SUGG, STUDIO_SLOT, studioSlot,
  } = C;

  // Build-injected asset host & catalog manifest filenames.
  // Kept verbatim so post.py, scripts/preview.mjs, scripts/catalog-source.mjs,
  // and tests/check_csp.py can discover the asset host and all 12 catalog manifests.
  const AH='';
  const desktop = window.PeopleStudioNative;
  const CATALOG_FILES = {
    pose: 'poses.json',
    depth: 'depths.json',
    costume: 'costumes.json',
    anime: 'anime.json',
    portrait: 'portraits.json',
    expr: 'expressions.json',
    hair: 'hair.json',
    dyn: 'dynasties.json',
    scene: 'scenes.json',
    light: 'lights.json',
    prop: 'props.json',
    furn: 'furniture.json',
  };

  const params = new URLSearchParams(location.search);
  let Q0 = (params.get('q') || '').slice(0, 60);
  const QT = params.get('t') || '';

  const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
  const ITEMS = {};
  const DATA = {};
  const LOAD = {};
  const materialErrors = new Map();
  let galleryRevision = 0;

  const st = {
    tab: 'pose',
    q: '',
    f: {},
    sort: 'def',
    lang: LS('lang') || 'zh',
    tl: LS('tl') || 'zh',
    fav: new Set(LS('fav') || []),
    sel: LS('sel') || [],
    rec: LS('rec') || [],
    recT: LS('recT') || {},
    ang: LS('ang') || {},
    list: [],
    n: 0,
    adult: LS('adult') === true,
  };

  const ok = it => it && (st.adult || !it.a);
  const visibleItem = it => ok(it) && (!it.materialID || it.materialRevision === galleryRevision);

  function localEntry(type, media) {
    const image = { t: media.url, m: media.url, i: media.url, o: media.url };
    const title = media.name || '本机图片';
    const angles = ['pose', 'depth', 'dyn'].includes(type) ? [{ ...image, zh: '原图', en: 'original' }] : undefined;
    return { ...image, id: 'local-' + media.id, ti: title, zh: title, en: title, cat: '本机导入',
      angles, tcMediaID: media.id, local: true };
  }

  function addLocalItem(type, media) {
    const item = localEntry(type, media);
    item.type = type;
    item.k = PFX[type] + item.id;
    item.ai = 0;
    item.ord = DATA[type]?.length || 0;
    const stored = LS('wbLocalItems') || [];
    if (!stored.some(entry => entry.type === type && entry.media.id === media.id)) {
      SV('wbLocalItems', [...stored, { type, media }]);
      if (DATA[type] && !ITEMS[item.k]) DATA[type].push(item);
    }
    ITEMS[item.k] = item;
    counts();
    return item.k;
  }

  function load(type) {
    if (LOAD[type]) return LOAD[type];
    const revision = galleryRevision;
    const bundledRequest = fetch(CATALOG_FILES[type]).then(response => {
      if (!response.ok) throw Error(response.status);
      return response.json();
    });
    const onlineRequest = Promise.resolve(desktop?.materials.catalog(type) || []).catch(error => {
      if (revision === galleryRevision) materialErrors.set(type, error.message);
      return [];
    });
    const request = Promise.all([bundledRequest, onlineRequest]).then(([raw, available]) => {
      if (revision !== galleryRevision) return load(type);
      const bundled = Array.isArray(raw) ? raw : (raw.items || []);
      const local = desktop ? (LS('wbLocalItems') || []).filter(item => item.type === type)
        .map(item => localEntry(type, item.media)) : [];
      const list = [...bundled, ...local, ...available];
      if (type === 'depth') DATA.depthMeta = raw.pack || null;
      list.forEach((it, index) => {
        it.type = type;
        it.k = PFX[type] + it.id;
        it.ord = index;
        if (it.materialID) it.materialRevision = revision;
        if (it.angles) it.ai = Math.min(st.ang[it.k] || 0, it.angles.length - 1);
        ITEMS[it.k] = it;
      });
      DATA[type] = list;
      counts();
      return list;
    }).catch(error => {
      if (LOAD[type] === request) delete LOAD[type];
      throw error;
    });
    LOAD[type] = request;
    return request;
  }

  async function refreshMaterials() {
    galleryRevision += 1;
    materialErrors.clear();
    for (const type of Object.keys(CATALOG_FILES)) {
      delete LOAD[type];
      delete DATA[type];
    }
    flippedCards.clear();
    if (st.tab && !isWorkspace(st.tab)) await setTab(st.tab);
  }

  function counterpartType(item) {
    if (item.type === 'pose') return 'depth';
    if (item.type === 'depth' && ['gia-depth', 'turbocider-samples', 'shisei'].includes(item.pack)) return 'pose';
    return null;
  }

  function poseCounterpart(item, index = item.ai) {
    const type = counterpartType(item);
    if (!type || !ok(item)) return null;
    const other = ITEMS[PFX[type] + item.id];
    if (!visibleItem(other) || !counterpartType(other)) return null;
    const angle = other.angles.findIndex(a => a.zh === item.angles[index]?.zh);
    return angle < 0 ? null : { item: other, angle };
  }

  const flippedCards = new Map();
  function cardFace(origin) {
    const face = flippedCards.get(origin.k);
    return ok(face) ? face : origin;
  }

  const needKeys = keys => Promise.all([...new Set(keys.map(k => KT[k[0]]).filter(Boolean))].map(load));
  const ath = url => (url.startsWith('http') ? url : AH + url);
  const thumb = it => ath(it.angles ? it.angles[it.ai].t : it.t);
  const big = it => (it.angles ? it.angles[it.ai].i : (innerWidth * (devicePixelRatio || 1) > 1500 ? it.i : it.m));
  const orig = it => {
    const path = it.angles ? it.angles[it.ai].o : it.o;
    return it.local || it.materialID ? path : it.ext ? '' : (AH || '/') + path;
  };

  function prompt(it, lang) {
    const angle = it.angles?.[it.ai];
    return [it[lang], angle && (lang === 'en' ? it.lineEn : it.line), angle?.[lang]]
      .filter(Boolean)
      .join(lang === 'en' ? ', ' : '，');
  }

  const title = it => (it.type === 'pose' || it.type === 'depth' ? it.zh : it.ti);
  const subtitle = (it, layout = it.type) => {
    if (layout === 'depth') return it.ja;
    if (st.lang === 'en') return prompt(it, 'en');
    if (layout === 'pose') return `${it.cat} · ${it.angles.length} 个角度`;
    if (layout === 'portrait') return `${it.sec} · ${it.len}${it.rk ? ' · 人气第 ' + it.rk : ''}`;
    return it.zh;
  };

  function recordNotice(text) {
    const last = $('#noticeAction');
    last.textContent = text;
    last.hidden = false;
  }

  function toast(text, undo) {
    recordNotice(text);
    const el = $('#toast');
    el.textContent = text;
    if (undo) {
      const btn = document.createElement('button');
      btn.textContent = '撤销';
      btn.onclick = () => {
        el.classList.remove('on', 'act');
        undo();
      };
      el.append(btn);
    }
    el.classList.toggle('act', !!undo);
    el.classList.remove('on');
    void el.offsetWidth;
    el.classList.add('on');
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove('on', 'act'), undo ? 4500 : 1800);
  }

  const EMPTY_ICONS = {
    fav: '<svg viewBox="0 0 24 24"><path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z"/></svg>',
    rec: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/></svg>',
    q: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  };

  async function copy(text, label) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const field = document.createElement('textarea');
      const opener = document.activeElement;
      field.value = text;
      field.style.cssText = 'position:fixed;opacity:0';
      document.body.append(field);
      field.select();
      const copied = document.execCommand('copy');
      field.remove();
      opener?.focus({ preventScroll: true });
      if (!copied) return toast('复制失败，请检查剪贴板权限');
    }
    toast('已复制' + (label ? `「${label}」` : ''));
  }

  // Copied references use the public deployment origin, including when previewing locally.
  const publicSite = document.querySelector('link[rel=canonical]')?.href || new URL('.', location.href).href;
  const copyRefs = $('#copyRefs');
  copyRefs.checked = LS('copyRefs') !== false;
  copyRefs.onchange = () => SV('copyRefs', copyRefs.checked);

  function copyBrief(text, items, lang, label) {
    if (!copyRefs.checked) return copy(text, label);
    const refs = items.map(it => it && (!desktop || it.materialID)
      ? PoseWearReferences.reference(it, it.ai ?? 0, publicSite, lang, AH || publicSite) : null);
    return copy(PoseWearReferences.formatBrief(text, refs, lang), label);
  }

  function counts() {
    const bars = $$('#libraryMenu [data-tab]');
    const quantities = [];
    for (const button of bars) {
      const stat = button.querySelector('[data-stat]');
      const items = DATA[button.dataset.tab];
      if (items) {
        stat.dataset.n = items.filter(it => !it.a).length;
        stat.dataset.na = items.length;
      }
      const count = Number((st.adult && stat.dataset.na) || stat.dataset.n);
      stat.textContent = String(count);
      quantities.push(count);
    }
    const maximum = Math.max(1, ...quantities);
    bars.forEach((button, index) => button.style.setProperty('--share', quantities[index] / maximum));
    const favCount = [...st.fav].filter(k => !ITEMS[k] || ok(ITEMS[k])).length;
    $('#nFav').textContent = favCount || '';
    $('#nRec').textContent = st.rec.filter(k => !ITEMS[k] || ok(ITEMS[k])).length || '';
  }

  function base() {
    const list =
      st.tab === 'fav'
        ? [...st.fav].reverse().map(k => ITEMS[k])
        : st.tab === 'recent'
          ? st.rec.map(k => ITEMS[k])
          : DATA[st.tab] || [];
    return list.filter(visibleItem);
  }

  function val(it, key) {
    if (key === 'ang') return it.angles.map(a => a.zh);
    if (key === 'type') return [TL(it)];
    if (key === 'col') {
      if (!it.kw) return [];
      if (!it._c) {
        const padded = ` ${it.kw} `;
        it._c = Object.keys(COLS).filter(c => padded.includes(` ${COLS[c]} `));
      }
      return it._c;
    }
    if (key === 'item') {
      if (!it.kw) return [];
      if (!it._i) {
        const padded = ` ${it.en.replace(/,/g, ' ')} `;
        it._i = Object.keys(ITEMK).filter(z => padded.includes(` ${ITEMK[z]} `));
      }
      return it._i;
    }
    return it[key] ? [it[key]] : [];
  }

  function drawFilters() {
    const filters = $('#filters');
    const positions = new Map();
    let focusedChip;
    if (filters.dataset.filterTab === st.tab) {
      for (const row of filters.querySelectorAll('.frow')) {
        positions.set(row.dataset.filter, row.scrollLeft);
      }
      const active = document.activeElement;
      if (filters.contains(active) && active.matches('.chip')) focusedChip = active.dataset;
    }

    const items = base();
    const rowList = FDEF[st.tab]
      .map(([key, label]) => {
        const freq = {};
        items.forEach(it => val(it, key).forEach(v => { freq[v] = (freq[v] || 0) + 1; }));
        const entries = Object.entries(freq).sort((a, b) => b[1] - a[1]);
        if (entries.length < 2 && !st.f[key]) return '';
        const chips = entries
          .map(([v, n]) => {
            const swatch = key === 'col' && COL_SWATCH[v]
              ? `<span class="csw" style="--sw:${COL_SWATCH[v]}" aria-hidden="true"></span>`
              : '';
            const active = st.f[key] === v;
            return `<button class="chip ${active ? 'on' : ''}" data-k="${key}" data-v="${esc(v)}" aria-pressed="${active}">${swatch}${esc(v)}<i>${n}</i></button>`;
          })
          .join('');
        return `<div class="frow" role="group" data-filter="${key}" aria-label="${label}"><label>${label}</label><button class="chip ${!st.f[key] ? 'on' : ''}" data-k="${key}" data-v="" aria-pressed="${!st.f[key]}">全部</button>${chips}</div>`;
      })
      .filter(Boolean);

    filters.innerHTML = '<div class="dh"><b>筛选</b><button class="sm pri" data-done>完成</button></div>' + rowList.join('');
    filters.dataset.filterTab = st.tab;

    const hasMoreRows = rowList.length > 1;
    $('#fBtn').hidden = !hasMoreRows;
    if (!hasMoreRows) drawer(false);

    for (const row of filters.querySelectorAll('.frow')) {
      row.scrollLeft = positions.get(row.dataset.filter) ?? 0;
    }
    if (focusedChip) {
      const replacement = [...filters.querySelectorAll('.chip')].find(
        chip => chip.dataset.k === focusedChip.k && chip.dataset.v === focusedChip.v
      );
      replacement?.focus({ preventScroll: true });
    }

    const activeFilters = Object.entries(st.f).filter(([, v]) => v);
    $('#active').innerHTML = activeFilters
      .map(([k, v]) => `<button data-k="${k}" title="移除此条件">${FLAB[k]}：${esc(v)} ✕</button>`)
      .join('');
    $('#fN').textContent = activeFilters.length || '';
    $('#clearAll').hidden = !(activeFilters.length || st.q);
  }

  function countMarkup(list, all) {
    const imageCount = arr => arr.reduce((n, it) => n + (it.angles ? it.angles.length : 1), 0);
    if (st.tab === 'depth') {
      return list.length === all.length
        ? `共 <b>${all.length}</b> 组 · ${imageCount(all)} 张`
        : `<b>${list.length}</b> / ${all.length} 组 · ${imageCount(list)} 张`;
    }
    return list.length === all.length
      ? `共 <b>${all.length}</b> 项`
      : `<b>${list.length}</b> / ${all.length} 项`;
  }

  function searchHaystack(it) {
    if (!it._s) {
      it._s = [
        it.id, it.ti, it.zh, it.en, it.ja, it.kw,
        it.cat, it.catEn, it.catJa, it.line, it.lineJa,
        it.no, it.sec, it.len, it.st,
        ...(it.angles || []).flatMap(a => [a.zh, a.en, a.ja]),
      ].join(' ').toLowerCase();
    }
    return it._s;
  }

  function apply(previousView) {
    const words = st.q.toLowerCase().trim().split(/\s+/).filter(Boolean);
    let list = base().filter(
      it =>
        Object.entries(st.f).every(([k, v]) => !v || val(it, k).includes(v)) &&
        words.every(w => searchHaystack(it).includes(w))
    );

    if (st.f.ang) {
      for (const it of list) {
        const index = it.angles.findIndex(a => a.zh === st.f.ang);
        if (index < 0) continue;
        it.ai = index;
        if (flippedCards.has(it.k)) {
          const mate = poseCounterpart(it, index);
          if (mate) mate.item.ai = mate.angle;
          else flippedCards.delete(it.k);
        }
      }
      tray();
    }

    if (st.sort === 'id') {
      list = list.slice().reverse();
    } else if (st.sort === 'shuf') {
      const order = previousView?.order;
      list = order
        ? list.sort((a, b) => (order.get(a.k) ?? Infinity) - (order.get(b.k) ?? Infinity))
        : shuffle(list);
    } else if (st.sort === 'rank') {
      list = list.slice().sort((a, b) => (a.rk || (a.rc && a.rc + 10) || 999) - (b.rk || (b.rc && b.rc + 10) || 999));
    } else if (st.sort === 'fav') {
      list = list.slice().sort((a, b) => st.fav.has(b.k) - st.fav.has(a.k));
    }

    st.list = list;
    window.X?.filterBatch?.(list.map(cardFace));

    const all = base();
    const total = all.length;
    $('#count').innerHTML = countMarkup(list, all);
    $('#grid').innerHTML = '';
    st.n = 0;

    const empty = !list.length;
    $('#empty').hidden = !empty;
    $('#clearAll').hidden = !(Object.values(st.f).some(Boolean) || st.q);
    if ($('#sketchBtn')) $('#sketchBtn').hidden = list.length < 2;
    document.body.classList.toggle('none0', (st.tab === 'fav' || st.tab === 'recent') && !total);

    if (empty) {
      const none = (st.tab === 'fav' || st.tab === 'recent') && !total;
      $('#emptyT').textContent = none
        ? (st.tab === 'fav' ? '还没有收藏' : '还没有浏览记录')
        : '没有找到匹配的内容';
      $('#emptyP').textContent = none
        ? (st.tab === 'fav'
            ? '看到喜欢的，点卡片右上角的 ♡。收藏只存在这台设备的浏览器里，可以导出备份。'
            : '打开过的大图会按时间排在这里，最多保留 100 条。')
        : '换个关键词试试，或者看看这些热门搜索：';
      $('#sugg').innerHTML = none
        ? ''
        : (SUGG[st.tab] || []).map(s => `<button class="chip" data-s="${s}">${s}</button>`).join('');
      $('#reset').hidden = none;
      $('#empty .emo').innerHTML = none ? (st.tab === 'fav' ? EMPTY_ICONS.fav : EMPTY_ICONS.rec) : EMPTY_ICONS.q;
      $('#emptyGo').innerHTML = none
        ? '<button class="sm" data-go="pose">逛姿势库</button>' +
          '<button class="sm" data-go="depth">逛深度姿势</button>' +
          '<button class="sm" data-go="costume">逛服装库</button>' +
          '<button class="sm" data-go="anime">逛动画服装</button>' +
          '<button class="sm" data-go="portrait">逛人像库</button>' +
          '<button class="sm" data-go="expr">逛表情库</button>' +
          '<button class="sm" data-go="dyn">逛历代服饰</button>' +
          '<button class="sm" data-go="scene">逛场景库</button>'
        : '';
    }

    more(previousView?.count || 30);
  }

  function tryLabel(it, selecting = false) {
    if (selecting) return st.sel.includes(it.k) ? '移出工作台' : '加入工作台';
    const slot = STUDIO_SLOT[it.type];
    return { portrait: '用作人物参考', garment: '试穿这套', pose: '用作姿势参考', expr: '用作表情参考' }[slot] || '加入生图组合';
  }

  function useReference(it) {
    if (STUDIO_SLOT[it.type]) return window.WB?.pick(it);
    if (!st.sel.includes(it.k)) toggleSel(it.k);
    openTray();
  }

  function cardBadges(it) {
    const mate = poseCounterpart(it);
    const label = it.type === 'pose' ? '切换到深度图' : '切换到线稿';
    const hint = it.type === 'pose' ? '深度图' : '线稿';
    const number = mate
      ? `<button class="badge card-flip" type="button" data-a="flip" title="${label}" aria-label="${label}">` +
        `<span>${esc(it.no)}</span>` +
        `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 0 1 15-6l2 2m0-5v5h-5M21 12a9 9 0 0 1-15 6l-2-2m0 5v-5h5"/></svg>` +
        `<em class="flip-hint" aria-hidden="true">${hint}</em></button>`
      : `<span class="badge">${esc(it.no)}</span>`;
    return `<div class="card-badges">${number}${it.a ? '<span class="badge age">18+</span>' : ''}</div>`;
  }

  function loadCardPairs() {
    if (!['pose', 'depth', 'fav', 'recent'].includes(st.tab)) return;
    const missing = new Set(base().map(counterpartType).filter(type => type && !DATA[type]));
    for (const type of missing) {
      load(type).then(() => {
        for (const badges of grid.querySelectorAll('.card-badges')) {
          const it = cardFace(ITEMS[badges.closest('.card').dataset.k]);
          if (counterpartType(it) === type && poseCounterpart(it)) badges.outerHTML = cardBadges(it);
        }
      }, () => {});
    }
  }

  function cardAngles(origin, it) {
    const angles = it.angles.map((angle, index) => ({ name: angle.zh, index }));
    if (it !== origin) {
      const order = new Map(origin.angles.map((angle, index) => [angle.zh, index]));
      angles.sort((a, b) => (order.get(a.name) ?? order.size) - (order.get(b.name) ?? order.size));
    }
    return angles
      .map(({ name, index }) =>
        `<button type="button" data-a="ang" data-i="${index}" aria-pressed="${index === it.ai}" class="${index === it.ai ? 'on' : ''}">${esc(name)}</button>`
      )
      .join('');
  }

  function card(origin, delay) {
    const it = cardFace(origin);
    const isSel = st.sel.includes(it.k);
    const isFav = st.fav.has(it.k);
    const isPose = it.type === 'pose';
    const isDepth = it.type === 'depth';
    const hasAngles = isPose || isDepth || !!it.angles;
    const label = title(it);
    const credit = it.materialSource ? `<div class="sub tc-material-credit">${esc(it.materialCredit)}</div>` : '';
    const miss = isDepth && !desktop ? `<a class="miss" href="${esc(it.site)}" target="_blank" rel="noopener">请前往原站查看</a>` : '';
    const w = hasAngles ? 300 : it.type === 'portrait' ? 400 : 360;
    const h = hasAngles ? 400 : it.type === 'portrait' ? 500 : 540;
    const topBadge = hasAngles
      ? (isPose || isDepth ? cardBadges(it) : `<span class="badge">${esc(it.no)}</span>${it.a ? '<span class="badge age">18+</span>' : ''}`) +
        `<div class="angs">${cardAngles(origin, it)}</div>`
      : '';

    return (
      `<article class="card ${window.X?.batchPicked?.(it.k) ? 'picked' : ''} ${isSel ? 'sel' : ''}" data-k="${origin.k}"${origin !== it ? ` data-show="${it.k}"` : ''} style="animation-delay:${delay || 0}ms">` +
      `<div class="ph ${isDepth ? 'depth' : isPose ? 'pose' : 'cos'}" data-a="open">` +
      `<img loading="lazy" decoding="async"${isDepth ? ` referrerpolicy="${esc(it.ref || 'no-referrer')}"` : ''} src="${esc(thumb(it))}" alt="${esc(label)}" width="${w}" height="${h}" draggable="false">` +
      `${miss}<button class="phk" data-a="open" aria-label="${esc(label)}"></button>` +
      `${it.rk ? `<span class="badge rk">TOP ${it.rk}</span>` : ''}${topBadge}` +
      `${st.tab === 'recent' ? '<button data-a="unrec" class="unrec" title="从记录中移除" aria-label="从记录中移除">✕</button>' : ''}` +
      `<div class="cbtns">` +
      `<button data-a="fav" class="${isFav ? 'on' : ''}" title="${isFav ? '取消收藏' : '收藏'}" aria-label="收藏" aria-pressed="${isFav}">${isFav ? '♥' : '♡'}</button>` +
      (it.previewOnly ? '' : `<button data-a="sel" class="${isSel ? 'on' : ''}" title="${isSel ? '移出组合' : '加入组合'}" aria-label="加入组合" aria-pressed="${isSel}">${isSel ? '✓' : '＋'}</button>`) +
      `</div>${isDepth || it.previewOnly ? '' : `<button class="try" data-a="try">${tryLabel(it, trayOpen)}</button>`}</div>` +
      `<div class="info"><div class="card-heading"><h3 title="${esc(label)}">${esc(label)}</h3>` +
      `<button class="card-copy" data-a="copy" title="复制提示词" aria-label="复制提示词"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M15 5V4H5v11h1"/></svg></button></div>` +
      `<div class="sub">${esc(subtitle(it, origin.type))}</div>${credit}` +
      `${origin.type === 'depth' ? `<div class="sub dim">${esc(it.line)} · ${esc(it.cat)}</div>` : ''}</div></article>`
    );
  }

  function recentGroup(key) {
    const ts = st.recT[key];
    if (!ts) return '更早';
    const dayStart = new Date();
    dayStart.setHours(0, 0, 0, 0);
    if (ts >= dayStart) return '今天';
    if (ts >= dayStart - 864e5) return '昨天';
    if (ts >= dayStart - 6 * 864e5) return '近 7 天';
    return '更早';
  }

  function more(count = 30) {
    if (st.n >= st.list.length) return;
    const grouped = st.tab === 'recent' && !st.q && !Object.values(st.f).some(Boolean);
    let prevGroup = grouped && st.n ? recentGroup(st.list[st.n - 1].k) : '';
    const html = st.list
      .slice(st.n, st.n + count)
      .map((it, i) => {
        let heading = '';
        if (grouped) {
          const g = recentGroup(it.k);
          if (g !== prevGroup) {
            heading = `<h4 class="rgrp">${g}</h4>`;
            prevGroup = g;
          }
        }
        return heading + card(it, motionPreference.matches ? 0 : Math.min(i, 14) * 35);
      })
      .join('');
    st.n += count;
    $('#grid').insertAdjacentHTML('beforeend', html);
  }

  const grid = $('#grid');
  grid.addEventListener('load', e => {
    if (e.target.tagName !== 'IMG') return;
    e.target.classList.add('ld');
    e.target.parentNode.classList.remove('err');
    e.target.parentNode.classList.add('done');
  }, true);
  grid.addEventListener('error', e => {
    if (e.target.tagName === 'IMG') e.target.parentNode.classList.add('err', 'done');
  }, true);
  new IntersectionObserver(entries => entries[0].isIntersecting && more(), { rootMargin: '1400px' }).observe($('#sentinel'));

  function replaceCard(el, prepared) {
    const origin = ITEMS[el.dataset.k];
    const oldImage = el.querySelector('img');
    const wrapper = document.createElement('div');
    wrapper.innerHTML = card(origin);
    const nextCard = wrapper.firstElementChild;
    const ph = nextCard.querySelector('.ph');
    const img = ph.querySelector('img');
    nextCard.style.animation = 'none';
    const active = document.activeElement;
    const action = el.contains(active) && active.dataset.a;

    if (prepared) {
      for (const name of ['alt', 'width', 'height', 'draggable']) {
        prepared.setAttribute(name, img.getAttribute(name));
      }
      prepared.classList.add('ld');
      img.replaceWith(prepared);
      ph.classList.add('done');
      if (!motionPreference.matches && oldImage.complete && oldImage.naturalWidth) {
        const outgoing = oldImage.cloneNode();
        outgoing.className = 'ld flip-previous';
        outgoing.alt = '';
        outgoing.setAttribute('aria-hidden', 'true');
        ph.classList.add('pair-turn');
        prepared.after(outgoing);
        outgoing.addEventListener('animationend', () => {
          outgoing.remove();
          ph.classList.remove('pair-turn');
        }, { once: true });
      }
    } else if (img.getAttribute('src') === oldImage.getAttribute('src')) {
      const oldFace = oldImage.parentElement;
      ph.classList.toggle('done', oldFace.classList.contains('done'));
      ph.classList.toggle('err', oldFace.classList.contains('err'));
      img.replaceWith(oldImage);
    }

    el.replaceWith(nextCard);
    if (action) {
      const sel = `button[data-a="${CSS.escape(action)}"]` + (active.dataset.i != null ? `[data-i="${CSS.escape(active.dataset.i)}"]` : '');
      nextCard.querySelector(sel)?.focus({ preventScroll: true });
    }
  }

  function refreshCard(key) {
    const escaped = CSS.escape(key);
    grid.querySelectorAll(`.card[data-show="${escaped}"],.card[data-k="${escaped}"]:not([data-show])`).forEach(el => replaceCard(el));
  }

  function updateCardAngle(it) {
    const escaped = CSS.escape(it.k);
    const nextSrc = thumb(it);
    grid.querySelectorAll(`.card[data-show="${escaped}"],.card[data-k="${escaped}"]:not([data-show])`).forEach(el => {
      const origin = ITEMS[el.dataset.k];
      const ph = el.querySelector('.ph');
      const img = ph.querySelector('img');
      if (img && img.getAttribute('src') !== nextSrc) {
        ph.classList.remove('err');
        img.src = nextSrc;
      }
      el.querySelectorAll('.angs [data-a="ang"]').forEach(btn => {
        const on = +btn.dataset.i === it.ai;
        btn.classList.toggle('on', on);
        btn.setAttribute('aria-pressed', String(on));
      });
      const badges = el.querySelector('.card-badges');
      if (badges && (it.type === 'pose' || it.type === 'depth')) badges.outerHTML = cardBadges(it);
      const sub = el.querySelector('.info .sub');
      if (sub && origin) sub.textContent = subtitle(it, origin.type);
    });
  }

  async function flipCard(el, button, it) {
    const mate = poseCounterpart(it);
    if (!mate || button.hasAttribute('aria-busy')) return;
    button.setAttribute('aria-busy', 'true');
    const image = new Image();
    const view = mate.item.angles[mate.angle];
    if (mate.item.type === 'depth') image.referrerPolicy = mate.item.ref || 'no-referrer';
    image.src = ath(view.t);
    try {
      await image.decode();
    } catch {
      if (button.isConnected) {
        button.removeAttribute('aria-busy');
        toast('图片加载失败，请重试');
      }
      return;
    }
    if (!button.isConnected || !ok(mate.item) || document.body.classList.contains('batch')) {
      button.removeAttribute('aria-busy');
      return;
    }
    if (mate.item.ai !== mate.angle) setAng(mate.item, mate.angle);
    if (mate.item.k === el.dataset.k) flippedCards.delete(el.dataset.k);
    else flippedCards.set(el.dataset.k, mate.item);
    replaceCard(el, image);
  }

  function setAng(it, i) {
    it.ai = i;
    st.ang[it.k] = i;
    SV('ang', st.ang);
    updateCardAngle(it);
    if (st.sel.includes(it.k)) tray();
  }

  function dropCard(k) {
    const el = grid.querySelector(`.card[data-k="${CSS.escape(k)}"]`);
    const idx = st.list.findIndex(x => x.k === k);
    if (idx >= 0) {
      st.list.splice(idx, 1);
      st.n = Math.max(0, st.n - 1);
    }
    const finalize = () => {
      el?.remove();
      if (!st.list.length) {
        apply();
      } else {
        $('#count').innerHTML = countMarkup(st.list, base());
        grid.querySelectorAll('.rgrp').forEach(heading => {
          const next = heading.nextElementSibling;
          if (!next || next.classList.contains('rgrp')) heading.remove();
        });
      }
    };
    if (el && !motionPreference.matches) {
      el.animate([{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'scale(.92)' }], {
        duration: 220,
        easing: 'cubic-bezier(.4,0,.2,1)',
      }).finished.then(finalize);
    } else {
      finalize();
    }
  }

  function toggleFav(k, quiet) {
    const on = !st.fav.has(k);
    if (on) st.fav.add(k);
    else st.fav.delete(k);
    SV('fav', [...st.fav]);
    counts();
    if (!quiet) {
      toast(on ? '已加入收藏' : '已取消收藏', on ? null : () => {
        toggleFav(k, 1);
        if (st.tab === 'fav') apply();
      });
    }
    refreshCard(k);
    if (st.tab === 'fav') {
      if (!$('#lb').hidden) pend.add(k);
      else if (on) apply();
      else dropCard(k);
    }
    if (!$('#lb').hidden) lbDraw();
  }

  function unRec(k) {
    const old = st.rec.slice();
    st.rec = st.rec.filter(x => x !== k);
    SV('rec', st.rec);
    counts();
    dropCard(k);
    toast('已从记录中移除', () => {
      st.rec = old;
      SV('rec', old);
      counts();
      apply();
    });
  }

  function toggleSel(k) {
    if (ITEMS[k]?.previewOnly) return toast('这张素材支持原站预览。');
    const index = st.sel.indexOf(k);
    const added = index < 0;
    if (added) st.sel.push(k);
    else st.sel.splice(index, 1);
    SV('sel', st.sel);
    refreshCard(k);
    tray(added ? k : null);
    const message = added ? '已加入组合' : '已移出组合';
    if (trayOpen) recordNotice(message);
    else toast(message);
    if (!$('#lb').hidden) lbDraw();
  }

  // ---------- Floating Workspace Dock ----------
  function visibleSelection() {
    const items = st.sel.map(k => ITEMS[k]).filter(visibleItem);
    const poses = items.filter(counterpartType);
    if (!poses.some(it => it.type === 'pose') || !poses.some(it => it.type === 'depth')) return items;
    const groups = new Map();
    for (const it of poses) {
      if (!groups.has(it.id)) groups.set(it.id, {});
      groups.get(it.id)[it.type] = it;
    }
    const ordered = [...groups.values()].flatMap(({ pose, depth }) => [pose, depth].filter(Boolean));
    const rest = items.filter(it => !counterpartType(it));
    rest.splice(items.indexOf(poses[0]), 0, ...ordered);
    return rest;
  }

  function compactSelection(selected) {
    if (selected.length <= 3) return selected;
    const recent = new Set();
    for (let i = st.sel.length - 1; i >= 0 && recent.size < 3; i--) {
      const it = ITEMS[st.sel[i]];
      if (!visibleItem(it) || recent.has(it)) continue;
      recent.add(it);
      const type = counterpartType(it);
      const other = type && ITEMS[PFX[type] + it.id];
      if (recent.size < 3 && other && counterpartType(other) && selected.includes(other)) recent.add(other);
    }
    return selected.filter(it => recent.has(it));
  }

  function combo(lang) {
    return visibleSelection()
      .map(it => {
        const { label } = PoseWearReferences.reference(it, it.ai ?? 0, publicSite, lang, AH || publicSite);
        return `${label} — ${prompt(it, lang)}`;
      })
      .join('\n');
  }

  let trayOpen = false;
  const trayThumbnail = item =>
    `<img data-type="${item.type}" src="${esc(thumb(item))}"${item.ref ? ` referrerpolicy="${esc(item.ref)}"` : ''} alt="">`;

  function trayItem(it) {
    const type = counterpartType(it);
    const name = type ? [it.no, title(it), it.angles[it.ai].zh].join(' · ') : title(it);
    const action = type === 'depth' ? '添加同视角深度姿势' : '添加同视角姿势';
    const pairButton = type
      ? `<button class="tray-pair" data-pair="${esc(it.k)}" title="${action}" aria-label="${action}" hidden>＋</button>`
      : '';
    const removeLabel = esc(type ? `${TL(it)} · ${name}` : name);
    return `<div title="${esc(name)}">${trayThumbnail(it)}<em>${TL(it)}</em>${pairButton}<button data-k="${esc(it.k)}" type="button" title="移除" aria-label="移除 ${removeLabel}">×</button></div>`;
  }

  function showTrayPairs() {
    for (const button of $$('#trayItems [data-pair]')) {
      const mate = poseCounterpart(ITEMS[button.dataset.pair]);
      button.hidden = !mate || st.sel.includes(mate.item.k);
    }
  }

  function tray(addedKey) {
    const selected = visibleSelection();
    const n = selected.length;
    const fab = $('#trayFab');
    const badge = $('#trayFabN');
    fab.hidden = trayOpen;
    fab.setAttribute('aria-expanded', String(trayOpen));
    badge.textContent = n;
    badge.hidden = !n;
    $('#trayFilm').innerHTML = compactSelection(selected).map(trayThumbnail).join('');
    if (addedKey && !trayOpen && !motionPreference.matches) {
      fab.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.12)' }, { transform: 'scale(1)' }], { duration: 300 });
    }
    $('#tray').hidden = !trayOpen;
    document.body.classList.toggle('tray', trayOpen);
    if (!trayOpen) return;

    $('#tgEmpty').hidden = n > 0;
    $('#trayClear').hidden = !n;
    $('#trayComboBox').hidden = !n;
    $('#trayN').textContent = n ? String(n) : '';
    $$('[data-tl]').forEach(b => b.classList.toggle('on', b.dataset.tl === st.tl));

    const strip = $('#trayItems');
    strip.innerHTML = selected.map(trayItem).join('');
    showTrayPairs();
    const missing = new Set(selected.map(counterpartType).filter(t => t && !DATA[t]));
    if (missing.size) Promise.all([...missing].map(load)).then(showTrayPairs, () => {});

    if (addedKey) {
      const added = strip.querySelector(`[data-k="${CSS.escape(addedKey)}"]`)?.parentElement;
      if (added) {
        const bounds = strip.getBoundingClientRect();
        const rect = added.getBoundingClientRect();
        const shift = rect.left < bounds.left ? rect.left - bounds.left : Math.max(0, rect.right - bounds.right);
        if (shift) strip.scrollBy({ left: shift, behavior: motionPreference.matches ? 'instant' : 'smooth' });
        if (!motionPreference.matches) {
          added.animate([{ opacity: 0.35, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 220 });
        }
      }
    }

    $('#trayText').textContent = combo(st.tl);
    tgSync();
  }

  $('#trayItems').onclick = e => {
    const source = e.target.closest('button[data-pair]');
    if (source) {
      const mate = poseCounterpart(ITEMS[source.dataset.pair]);
      if (!mate || st.sel.includes(mate.item.k)) return;
      setAng(mate.item, mate.angle);
      toggleSel(mate.item.k);
      $('#trayItems').querySelector(`[data-k="${CSS.escape(mate.item.k)}"]`).focus({ preventScroll: true });
      return;
    }
    const button = e.target.closest('button[data-k]');
    if (!button) return;
    const pos = [...$$('#trayItems [data-k]')].indexOf(button);
    toggleSel(button.dataset.k);
    const buttons = $$('#trayItems [data-k]');
    (buttons[Math.min(pos, buttons.length - 1)] || $('#tgAddRow button')).focus({ preventScroll: true });
  };
  $('#trayCopyEn').onclick = () => copyBrief(combo('en'), visibleSelection(), 'en', '组合英文提示词');
  $('#trayCopyZh').onclick = () => copyBrief(combo('zh'), visibleSelection(), 'zh', '组合中文描述');
  $$('[data-tl]').forEach(b => {
    b.onclick = () => {
      st.tl = b.dataset.tl;
      SV('tl', st.tl);
      tray();
    };
  });

  function setTrayOpen(open) {
    trayOpen = open;
    tray();
    for (const button of grid.querySelectorAll('[data-a="try"]')) {
      const item = cardFace(ITEMS[button.closest('.card').dataset.k]);
      button.textContent = tryLabel(item, open);
    }
  }
  function openTray() { setTrayOpen(true); $('#tray').focus({ preventScroll: true }); }
  function closeTray(restore = true) { setTrayOpen(false); if (restore) $('#trayFab').focus({ preventScroll: true }); }

  $('#trayMin').onclick = () => closeTray();
  $('#trayFab').onclick = openTray;
  $('#tgAddRow').onclick = e => {
    const button = e.target.closest('[data-add]');
    if (!button || !window.WB) return;
    const menu = button.closest('details');
    if (menu) {
      menu.open = false;
      menu.querySelector('summary').focus({ preventScroll: true });
    }
    WB.openPicker(button.dataset.add, {
      keys: () => st.sel.slice(),
      toggle: (k, ai) => {
        const item = ITEMS[k];
        if (ai != null && item.ai !== ai && !st.sel.includes(k)) setAng(item, ai);
        toggleSel(k);
        return st.sel.includes(k);
      },
    });
  };
  $('#trayClear').onclick = () => {
    const prev = st.sel;
    st.sel = [];
    SV('sel', []);
    prev.forEach(refreshCard);
    tray();
    $('#tgAddRow button').focus({ preventScroll: true });
    toast('已清空组合');
  };

  // Floating dock quick generation
  const trayDraft = LS('trayDraft');
  let tgAutoPrompt = '';
  let tgEdited = trayDraft?.prompt != null;
  let tgBusy = false;
  let tgRecs = [];
  let tgUrls = [];
  let tgCur = 0;

  if (trayDraft) {
    $('#tgPrompt').value = trayDraft.prompt ?? '';
    $('#tgN').value = trayDraft.count;
    $('#tgAR').value = trayDraft.ratio;
  }

  function saveTrayDraft() {
    SV('trayDraft', { prompt: tgEdited ? $('#tgPrompt').value : null, count: $('#tgN').value, ratio: $('#tgAR').value });
  }

  function groupSelection(items) {
    const groups = { portrait: [], garment: [], pose: [], expr: [], extra: [] };
    for (const it of items) groups[studioSlot(it.type)].push(it);
    return groups;
  }
  const traySelection = () => groupSelection(visibleSelection());

  function tgSync() {
    const editor = $('#tgPrompt');
    const reset = $('#tgResetPrompt');
    $('#tgProv').textContent = window.WB && WB.configured() ? WB.provLabel() : '未配置 API';
    $('#tgCfg .dot').classList.toggle('on', !!window.WB?.configured());
    if (!visibleSelection().length) {
      tgAutoPrompt = '';
      if (!tgEdited) editor.value = '';
      if (reset) reset.hidden = true;
      return;
    }
    if (!window.WB) return;
    tgAutoPrompt = WB.selectionPrompt(traySelection(), st.tl);
    if (!tgEdited) editor.value = tgAutoPrompt;
    if (reset) reset.hidden = !tgEdited;
  }

  $('#tgResetPrompt').onclick = () => {
    tgEdited = false;
    $('#tgPrompt').value = tgAutoPrompt;
    saveTrayDraft();
    $('#tgResetPrompt').hidden = true;
    toast('已按当前选择重新生成提示词');
  };
  $('#tgCopyDraft').onclick = () => {
    const g = traySelection();
    const inputs = [...g.portrait.slice(0, 1), ...g.garment.slice(0, 2), ...g.pose.slice(0, 6), ...g.expr.slice(0, 1)];
    const other = visibleSelection().filter(it => !inputs.includes(it));
    copyBrief($('#tgPrompt').value, [...inputs, ...other], st.tl, '提示词');
  };
  $('#tgPrompt').oninput = e => {
    tgEdited = e.target.value !== tgAutoPrompt;
    if ($('#tgResetPrompt')) $('#tgResetPrompt').hidden = !tgEdited || !visibleSelection().length;
    saveTrayDraft();
  };
  $('#tgN').onchange = $('#tgAR').onchange = saveTrayDraft;

  function tgStat(text) {
    const el = $('#tgStat');
    if (!el) return;
    el.hidden = !text;
    if (text) el.textContent = text;
  }

  function tgRender() {
    tgUrls.forEach(u => URL.revokeObjectURL(u));
    tgUrls = tgRecs.map(r => typeof r.after === 'string' ? r.after : URL.createObjectURL(r.after));
    $('#tgRes').innerHTML = tgUrls
      .map((u, i) => `<button data-tg="${i}" title="查看大图"><img src="${u}" alt="生成结果 ${i + 1}"></button>`)
      .join('');
  }

  $('#tgRes').onclick = e => {
    const b = e.target.closest('[data-tg]');
    if (!b) return;
    tgCur = +b.dataset.tg;
    $('#tgViewImg').src = tgUrls[tgCur];
    openDialog($('#tgView'), $('#tgViewX'), b);
  };
  const tgClose = () => closeDialog($('#tgView'));
  $('#tgViewX').onclick = tgClose;
  $('#tgView').addEventListener('click', e => { if (e.target.classList.contains('tgv-bg')) tgClose(); });
  $('#tgDl').onclick = () => {
    const r = tgRecs[tgCur];
    if (!r) return;
    if (desktop && r.tcOutput) void window.TurboCider.media.save(r.tcOutput).catch(error => toast(error.message));
    else if (window.X) window.X.dl(r.after, `posewear-gen-${tgCur + 1}.png`);
  };
  $('#tgHist').onclick = () => { tgClose(); closeTray(false); navigate('history'); };
  $('#tgCfg').onclick = () => window.WB?.cfgOpen();
  $('#cfg').addEventListener('click', e => {
    if (trayOpen && e.target.closest('#cfSave,#cfClr')) tgSync();
  });

  async function sendToStudio(items, append = false, draft) {
    const groups = groupSelection(items);
    try {
      if (items.length || draft) await (append ? WB.addSelection : WB.setSel)(groups, '', draft);
      await navigate('studio');
      return true;
    } catch {
      toast('素材未能载入，请重试');
      return false;
    }
  }

  $('#tgStudio').onclick = async () => {
    const items = visibleSelection();
    const promptVal = $('#tgPrompt').value.trim();
    const draft = items.length || promptVal
      ? { prompt: tgEdited ? promptVal : '', lang: st.tl, count: $('#tgN').value, ratio: $('#tgAR').value }
      : null;
    if (await sendToStudio(items, false, draft)) {
      closeTray(false);
      $('#stT').focus({ preventScroll: true });
    }
  };
  $('#tgStop').onclick = () => window.WB?.stop();
  $('#tgGo').onclick = async () => {
    if (tgBusy || !window.WB) return;
    if (!WB.configured()) return WB.cfgOpen();
    if (!visibleSelection().length && !$('#tgPrompt').value.trim()) return $('#tgPrompt').focus();
    tgSync();
    tgBusy = true;
    const go = $('#tgGo');
    go.disabled = true;
    go.textContent = '生成中…';
    $('#tgStop').hidden = false;
    tgStat('正在准备素材…');
    const pv = $('#tgPrompt').value.trim();
    const opts = {
      selection: traySelection(),
      lang: st.tl,
      n: +$('#tgN').value || 1,
      ar: $('#tgAR').value,
      onTick: (d, t) => tgStat(`生成中… 已完成 ${d} / ${t}`),
    };
    if (pv) opts.prompt = pv;
    try {
      const r = await WB.quick(opts);
      if (r.got.length) {
        tgRecs = r.got;
        tgRender();
      }
      const messages = [];
      if (r.got.length) {
        messages.push(
          r.unsaved
            ? `完成 ${r.got.length} 张，其中 ${r.unsaved} 张未存入生成记录，请立即下载`
            : `完成 ${r.got.length} 张，已存入生成记录`
        );
      }
      if (r.cancelled) {
        messages.push('已取消');
      } else if (r.errs.length) {
        const err = r.errs.find(x => x.safe) || r.errs[0];
        messages.push(err.safe ? '被服务商的安全审核拦下了' : '生成失败：' + WB.nice(err).split('\n')[0]);
      }
      if (r.warns?.length) messages.push(...r.warns);
      if (r.nokey) messages.push('请先配置 API');
      tgStat(messages.join('；'));
    } catch (err) {
      tgStat('生成失败：' + (err.message || err));
    } finally {
      tgBusy = false;
      go.disabled = false;
      go.textContent = '生成图片';
      $('#tgStop').hidden = true;
    }
  };

  // ---------- Grid Interactions & Navigation ----------
  grid.onclick = e => {
    if (e.target.closest('a.miss')) return;
    const actEl = e.target.closest('[data-a]');
    if (!actEl) return;
    const cardEl = e.target.closest('.card');
    const origin = ITEMS[cardEl.dataset.k];
    const it = cardFace(origin);
    e.stopPropagation();

    const ph = cardEl.querySelector('.ph');
    if (actEl.dataset.a === 'open' && ph.classList.contains('err')) {
      const im = ph.querySelector('img');
      ph.classList.remove('err', 'done');
      im.src = `${thumb(it)}?r=${Date.now()}`;
      return;
    }

    const actions = {
      open: () => lbOpen(st.list.indexOf(origin), cardEl),
      flip: () => flipCard(cardEl, actEl, it),
      fav: () => toggleFav(it.k),
      unrec: () => unRec(origin.k),
      sel: () => toggleSel(it.k),
      copy: () => copyBrief(prompt(it, st.lang), [it], st.lang, '提示词'),
      try: () => (trayOpen ? toggleSel(it.k) : useReference(it)),
      ang: () => setAng(it, +actEl.dataset.i),
    };
    actions[actEl.dataset.a]?.();
  };

  function viewT(fn) {
    if (motionPreference.matches) {
      fn();
      return Promise.resolve();
    }
    if (document.startViewTransition) {
      const transition = document.startViewTransition(fn);
      transition.ready.catch(() => {}); // Transitions may be skipped while navigating quickly.
      return transition.updateCallbackDone;
    }
    fn();
    grid.classList.add('fade');
    requestAnimationFrame(() => requestAnimationFrame(() => grid.classList.remove('fade')));
    return Promise.resolve();
  }

  function drawer(open) {
    $('#filters').classList.toggle('open', open);
    $('#fBtn').classList.toggle('on', open);
    $('#fBtn').setAttribute('aria-expanded', String(open));
  }

  function scrollHorizontal(event) {
    const mode = event.deltaMode;
    if (!event.cancelable || event.ctrlKey || event.shiftKey || event.deltaX || !event.deltaY) return;
    const row = event.target.closest('.tabs, .frow');
    if (!row || row.scrollWidth <= row.clientWidth) return;
    const start = row.scrollLeft;
    const unit = mode === WheelEvent.DOM_DELTA_LINE ? 16 : mode === WheelEvent.DOM_DELTA_PAGE ? row.clientWidth : 1;
    row.scrollLeft += event.deltaY * unit;
    if (row.scrollLeft !== start) event.preventDefault();
  }

  $('.tabs').addEventListener('wheel', scrollHorizontal, { passive: false });
  $('#filters').addEventListener('wheel', scrollHorizontal, { passive: false });
  $('#fBtn').onclick = () => drawer(!$('#filters').classList.contains('open'));
  $('#drawerBg').onclick = () => drawer(false);
  $('#filters').onclick = e => {
    if (e.target.closest('[data-done]')) return drawer(false);
    const chip = e.target.closest('.chip');
    if (!chip) return;
    st.f[chip.dataset.k] = chip.dataset.v;
    viewT(() => { drawFilters(); apply(); });
  };
  $('#active').onclick = e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    st.f[btn.dataset.k] = '';
    viewT(() => { drawFilters(); apply(); });
  };

  const clearAll = () => {
    st.f = {};
    st.q = '';
    $('#q').value = '';
    $('#qx').hidden = true;
    viewT(() => { drawFilters(); apply(); });
  };
  $('#reset').onclick = clearAll;
  $('#clearAll').onclick = clearAll;
  $('#sugg').onclick = e => {
    const s = e.target.closest('[data-s]');
    if (!s) return;
    st.f = {};
    $('#q').value = st.q = s.dataset.s;
    $('#qx').hidden = false;
    drawFilters();
    apply();
  };
  $('#sort').onchange = e => { st.sort = e.target.value; viewT(apply); };
  $('#qDone').onclick = () => $('#libraryMenu summary').focus({ preventScroll: true });

  let qt;
  $('#q').oninput = e => {
    $('#qx').hidden = !e.target.value;
    clearTimeout(qt);
    qt = setTimeout(() => { st.q = e.target.value; apply(); }, 160);
  };
  $('#qx').onclick = () => {
    $('#q').value = '';
    st.q = '';
    $('#qx').hidden = true;
    apply();
    $('#q').focus();
  };

  function applySearch(query) {
    $('#q').value = query;
    st.q = query;
    $('#qx').hidden = !query;
    apply();
  }

  function randPick() {
    if (!st.list.length) return toast('当前没有可选的内容');
    lbOpen((Math.random() * st.list.length) | 0);
  }
  $('#rand').onclick = randPick;

  // Description language toggle
  function setLang(lang, quiet) {
    st.lang = lang;
    SV('lang', lang);
    document.documentElement.dataset.lang = lang;
    $$('button[data-lang]').forEach(b => b.classList.toggle('on', b.dataset.lang === lang));
    grid.querySelectorAll('.card').forEach(c => {
      const origin = ITEMS[c.dataset.k];
      if (origin) c.querySelector('.sub').textContent = subtitle(cardFace(origin), origin.type);
    });
    if (!quiet) toast(lang === 'zh' ? '描述语言：中文' : '描述语言：英文提示词');
  }
  $$('button[data-lang]').forEach(b => { b.onclick = () => setLang(b.dataset.lang); });

  // Tabs & shell menus
  function selectTab(tab) {
    const active = $(`.tabs [data-tab="${tab}"]`);
    const library = active.closest('#libraryMenu');
    const personal = active.closest('#personalMenu');
    $('#libraryLabel').textContent = library ? TNAME[tab] : '素材库';
    $('#personalLabel').textContent = personal ? TNAME[tab] : '我的';
    $('#collectionTitle').textContent = TNAME[tab];
    for (const button of $$('.tabs [data-tab]')) {
      button.classList.toggle('on', button === active);
      if (button === active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    }
    for (const menu of $$('.tabs details')) {
      menu.classList.toggle('current', menu === library || menu === personal);
      menu.open = false;
    }
  }

  document.addEventListener('pointerdown', event => {
    for (const menu of $$('.shell-menu[open]')) {
      if (!menu.contains(event.target)) menu.open = false;
    }
  });
  for (const menu of $$('.shell-menu')) {
    menu.addEventListener('focusout', event => {
      if (event.relatedTarget && !menu.contains(event.relatedTarget)) menu.open = false;
    });
    menu.addEventListener('keydown', event => {
      if (event.key === 'Escape' && menu.open) {
        event.preventDefault();
        event.stopPropagation();
        menu.open = false;
        menu.querySelector('summary').focus();
      }
      const items = [...menu.querySelectorAll('[data-tab],[data-theme-choice]')];
      if (!items.length) return;
      const index = items.indexOf(event.target);
      let next;
      if (event.key === 'ArrowDown') next = (index + 1) % items.length;
      else if (event.key === 'ArrowUp') next = index <= 0 ? items.length - 1 : index - 1;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = items.length - 1;
      else return;
      event.preventDefault();
      event.stopPropagation();
      menu.open = true;
      items[next].focus();
    });
  }

  // Keep the gallery position while the user works in the studio or checks a result.
  let galleryReturn = null;
  let tabRequest = 0;
  const isWorkspace = tab => tab === 'studio' || tab === 'history';

  function navigate(tab) {
    const hash = '#' + tab;
    if (location.hash !== hash) {
      if (history.state?.galleryItem) history.replaceState(null, '', hash);
      else history.pushState(null, '', hash);
    }
    const openingStudio = tab === 'studio' && st.tab !== 'studio';
    const ready = setTab(tab);
    if (openingStudio) $('#stT').focus({ preventScroll: true });
    return ready;
  }

  async function setTab(t, openId, workbenchId) {
    const request = ++tabRequest;
    const changed = st.tab !== t;
    clearTimeout(qt);
    if (changed && st.tab && !isWorkspace(st.tab) && isWorkspace(t)) {
      galleryReturn = {
        tab: st.tab,
        query: $('#q').value,
        filters: { ...st.f },
        count: st.n,
        scroll: scrollY,
        cardKey: document.activeElement.closest('.card')?.dataset.k,
        order: st.sort === 'shuf' ? new Map(st.list.map((item, i) => [item.k, i])) : null,
      };
    }
    const previous = isWorkspace(st.tab) && !openId && workbenchId === undefined && galleryReturn?.tab === t
      ? galleryReturn
      : null;
    if (changed && !isWorkspace(t)) galleryReturn = null;
    if (changed) {
      window.X?.closeBatch?.();
      flippedCards.clear();
    }

    st.tab = t;
    tray();
    document.body.classList.toggle('depth', t === 'depth');

    if (t === 'studio') {
      if ($('#tcCatalogError')) $('#tcCatalogError').hidden = true;
      selectTab(t);
      document.title = desktop ? '工作台 · 人物工作室' : '工作台 · 衣姿 PoseWear';
      history.replaceState(null, '', '#studio');
      document.body.classList.add('studio');
      document.body.classList.remove('histview');
      $('#studio').hidden = false;
      $('#histSec').hidden = true;
      scrollTo({ top: 0 });
      window.WB?.show();
      return;
    }

    if (t === 'history') {
      if ($('#tcCatalogError')) $('#tcCatalogError').hidden = true;
      selectTab(t);
      document.title = desktop ? '生成记录 · 人物工作室' : '生成记录 · 衣姿 PoseWear';
      history.replaceState(null, '', '#history');
      document.body.classList.remove('studio');
      document.body.classList.add('histview');
      $('#studio').hidden = true;
      $('#histSec').hidden = false;
      scrollTo({ top: 0 });
      window.HIS?.show();
      return;
    }

    if (document.body.classList.contains('studio')) {
      document.body.classList.remove('studio');
      $('#studio').hidden = true;
    }
    if (document.body.classList.contains('histview')) {
      document.body.classList.remove('histview');
      $('#histSec').hidden = true;
    }
    if (changed) grid.classList.add('gbusy');

    st.f = previous?.filters || {};
    st.q = previous?.query || '';
    $('#q').value = st.q;
    $('#qx').hidden = !st.q;
    selectTab(t);
    document.title = `${TNAME[t]} · ${desktop ? '人物工作室' : '衣姿 PoseWear'}`;
    if (!openId) history.replaceState(null, '', '#' + t);

    $('#rand').hidden = t === 'fav' || t === 'recent';
    $('#favTools').hidden = t !== 'fav';
    $('#clrRec').hidden = t !== 'recent';
    $('#sort').hidden = t === 'recent';
    $('#q').placeholder = t === 'depth' ? '搜姿势名、系列、分类' : (innerWidth <= 760 ? '搜索' : '搜姿势、颜色、单品');
    $('#sort option[value=rank]').hidden = t !== 'portrait';
    if (t !== 'portrait' && st.sort === 'rank') {
      st.sort = 'def';
      $('#sort').value = 'def';
    }
    document.body.classList.toggle('compact', t === 'fav' || t === 'recent');

    if ((!DATA[t] && !(t === 'pose' && grid.querySelector('.ssr'))) || t === 'fav' || t === 'recent') {
      grid.innerHTML = '<div class="skel"></div>'.repeat(10);
      $('#empty').hidden = true;
    }

    try {
      if (t === 'fav') await needKeys([...st.fav]);
      else if (t === 'recent') await needKeys(st.rec);
      else await load(t);
    } catch {
      if (request !== tabRequest) return;
      grid.innerHTML = '';
      $('#empty').hidden = false;
      $('#emptyT').textContent = '数据加载失败';
      $('#emptyP').textContent = '请检查网络后刷新页面重试';
      $('#sugg').innerHTML = '';
      $('#reset').hidden = true;
      $('#empty .emo').innerHTML = EMPTY_ICONS.q;
      grid.classList.remove('gbusy');
      $('#emptyGo').innerHTML = '<button class="sm pri" id="retry">重新加载</button>';
      $('#retry').onclick = () => setTab(t, openId, workbenchId);
      return;
    }

    if (request !== tabRequest) return;
    const upstream = $('#tcCatalogError');
    if (upstream) {
      upstream.hidden = !materialErrors.has(t);
      if (!upstream.hidden) upstream.textContent = '在线素材暂不可用。' + materialErrors.get(t);
    }
    if (Q0) {
      st.q = Q0;
      $('#q').value = Q0;
      $('#qx').hidden = false;
      Q0 = '';
    }

    const go = () => {
      if (request !== tabRequest) return;
      drawFilters();
      apply(previous);
      loadCardPairs();
    };
    if (changed) await viewT(go);
    else go();
    if (request !== tabRequest) return;
    grid.classList.remove('gbusy');

    if (workbenchId !== undefined) {
      const it = ITEMS[PFX[t] + workbenchId];
      if (it && !it.a && !st.sel.includes(it.k)) toggleSel(it.k);
      openTray();
    } else if (openId) {
      openById(openId);
    } else if (previous) {
      scrollTo({ top: previous.scroll, behavior: 'instant' });
      if (previous.cardKey) {
        grid.querySelector(`.card[data-k="${CSS.escape(previous.cardKey)}"] .phk`)?.focus({ preventScroll: true });
      }
    } else if (changed && scrollY > 0) {
      scrollTo({ top: 0, behavior: 'instant' });
    }
  }

  $('.tabs').onclick = e => {
    const button = e.target.closest('[data-tab]');
    if (!button) return;
    const tab = button.dataset.tab;
    const menu = button.closest('details');
    if (menu) {
      menu.open = false;
      menu.querySelector('summary').focus({ preventScroll: true });
    }
    if (tab === st.tab) {
      scrollTo({ top: 0, behavior: motionPreference.matches ? 'auto' : 'smooth' });
      return;
    }
    navigate(tab);
  };

  // Dialog & scroll utilities
  const hdr = () => document.documentElement.style.setProperty('--hdr', $('#top').offsetHeight + 'px');
  addEventListener('resize', () => hdr());
  new ResizeObserver(hdr).observe($('#top'));

  let scrolled = false;
  addEventListener('scroll', () => {
    const s = scrollY > 24;
    if (s !== scrolled) {
      scrolled = s;
      document.body.classList.toggle('scrolled', s);
      setTimeout(hdr, 420);
    }
    $('#toTop').classList.toggle('on', scrollY > 900);
  }, { passive: true });
  $('#toTop').onclick = () => scrollTo({ top: 0, behavior: motionPreference.matches ? 'auto' : 'smooth' });

  const dialogOpeners = new WeakMap();
  function openDialog(dialog, initial, opener = document.activeElement) {
    dialogOpeners.set(dialog, opener);
    dialog.hidden = false;
    (initial || dialog.querySelector('button')).focus({ preventScroll: true });
  }
  function closeDialog(dialog) {
    dialog.hidden = true;
    dialogOpeners.get(dialog)?.focus({ preventScroll: true });
    dialogOpeners.delete(dialog);
  }
  function trapTab(e, dialog) {
    const focusable = [...dialog.querySelectorAll('button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),textarea:not(:disabled),summary,[tabindex="0"]')]
      .filter(el => el.tabIndex >= 0 && el.getClientRects().length);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (!dialog.contains(active) || (e.shiftKey ? active === first : active === last)) {
      e.preventDefault();
      (e.shiftKey ? last : first).focus();
    }
  }

  $('#themeMenu').addEventListener('click', event => {
    const option = event.target.closest('[data-theme-choice]');
    if (!option) return;
    window.PoseWearTheme.select(option.dataset.themeChoice);
    $('#themeMenu').open = false;
    $('#theme').focus({ preventScroll: true });
  });
  $('#help').onclick = () => {
    const menu = $('#noticeMenu');
    menu.open = false;
    openDialog($('#helpBox'), null, menu.querySelector('summary'));
  };
  $('#helpBox').onclick = e => {
    if (e.target.id === 'helpBox' || e.target.closest('[data-close]')) closeDialog($('#helpBox'));
  };
  $('.skip').onclick = e => {
    e.preventDefault();
    const target = $(st.tab === 'studio' ? '#studio' : st.tab === 'history' ? '#histSec' : '#grid');
    target.tabIndex = -1;
    target.focus({ preventScroll: true });
    target.scrollIntoView({ block: 'start' });
  };

  // Favorites import/export & recent storage sync
  $('#exp').onclick = () => {
    if (!st.fav.size) return toast('还没有收藏可导出');
    const items = [...st.fav]
      .map(k => ITEMS[k])
      .filter(Boolean)
      .map(it => ({ 类型: TL(it), 编号: it.id, 名称: title(it), 中文: prompt(it, 'zh'), 英文: prompt(it, 'en'), key: it.k }));
    const blob = new Blob(
      [JSON.stringify({ app: '衣姿 PoseWear', 导出时间: new Date().toLocaleString('zh-CN'), 收藏: items }, null, 2)],
      { type: 'application/json' }
    );
    if (desktop) { void window.X.dl(blob, '人物工作室-收藏.json'); return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `衣姿-收藏-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    toast(`已导出 ${items.length} 条收藏`);
  };
  $('#imp').onclick = () => $('#impF').click();
  $('#impF').onchange = async e => {
    try {
      const json = JSON.parse(await e.target.files[0].text());
      const keys = (json.收藏 || []).map(x => x.key).filter(k => /^[pcradehysltf]:/.test(k));
      await needKeys(keys);
      let added = 0;
      keys.forEach(k => {
        if (ITEMS[k] && !st.fav.has(k)) {
          st.fav.add(k);
          added++;
        }
      });
      SV('fav', [...st.fav]);
      counts();
      setTab('fav');
      toast(`已导入 ${added} 条收藏`);
    } catch {
      toast('导入失败：文件格式不正确');
    }
    e.target.value = '';
  };
  $('#clrRec').onclick = () => {
    if (!st.rec.length) return;
    const old = st.rec.slice();
    st.rec = [];
    SV('rec', []);
    counts();
    viewT(apply);
    toast(`已清空 ${old.length} 条记录`, () => {
      st.rec = old;
      SV('rec', old);
      counts();
      viewT(apply);
    });
  };
  $('#clrFav').onclick = () => {
    if (!st.fav.size) return toast('还没有收藏');
    const old = [...st.fav];
    st.fav = new Set();
    SV('fav', []);
    counts();
    viewT(apply);
    toast(`已清空 ${old.length} 条收藏`, () => {
      st.fav = new Set(old);
      SV('fav', old);
      counts();
      viewT(apply);
    });
  };

  document.addEventListener('click', e => {
    const b = e.target.closest('[data-go]');
    if (b) navigate(b.dataset.go);
  });
  addEventListener('offline', () => toast('网络已断开，看过的图片还能继续浏览'));
  addEventListener('online', () => toast('网络已恢复'));

  function addRec(k) {
    st.rec = [k, ...st.rec.filter(x => x !== k)].slice(0, 100);
    st.recT[k] = Date.now();
    const nextTimes = {};
    st.rec.forEach(x => { if (st.recT[x]) nextTimes[x] = st.recT[x]; });
    st.recT = nextTimes;
    SV('rec', st.rec);
    SV('recT', nextTimes);
    counts();
  }

  addEventListener('storage', e => {
    if (!['fav', 'rec', 'recT', 'adult', 'sel'].includes(e.key)) return;
    st.fav = new Set(LS('fav') || []);
    st.rec = LS('rec') || [];
    st.recT = LS('recT') || {};
    st.sel = LS('sel') || [];
    if (e.key === 'adult' && st.adult !== (LS('adult') === true)) ageSet(LS('adult') === true);
    counts();
    needKeys([...st.fav, ...st.rec, ...st.sel])
      .then(() => {
        tray();
        if ((st.tab === 'fav' || st.tab === 'recent') && $('#lb').hidden) apply();
      })
      .catch(() => {});
  });

  // ---------- Lightbox & Artist Inspection Studio ----------
  let li = 0;
  let srcCard = null;
  const preCache = new Map();

  function preload(src) {
    if (preCache.has(src)) return preCache.get(src);
    const promise = new Promise(resolve => {
      const img = new Image();
      if (/^https?:/i.test(src)) img.referrerPolicy = 'no-referrer';
      img.onload = () => resolve(src);
      img.onerror = () => resolve(null);
      img.src = src;
    });
    preCache.set(src, promise);
    if (preCache.size > 80) preCache.delete(preCache.keys().next().value);
    return promise;
  }

  function rectAnim(el, from, reverse) {
    const to = el.getBoundingClientRect();
    if (!from || !to.width) {
      return el.animate(
        [{ opacity: 0, transform: 'scale(.94)' }, { opacity: 1, transform: 'none' }],
        { duration: reverse ? 220 : 380, easing: 'cubic-bezier(.16,1,.3,1)', direction: reverse ? 'reverse' : 'normal', fill: 'both' }
      );
    }
    const dx = from.left + from.width / 2 - (to.left + to.width / 2);
    const dy = from.top + from.height / 2 - (to.top + to.height / 2);
    const scale = Math.max(from.width / to.width, 0.08);
    return el.animate(
      [
        { transform: `translate(${dx}px,${dy}px) scale(${scale})`, opacity: 0.2, borderRadius: '40px' },
        { transform: 'none', opacity: 1, borderRadius: '20px' },
      ],
      {
        duration: reverse ? 320 : 520,
        easing: reverse ? 'cubic-bezier(.4,0,.2,1)' : 'cubic-bezier(.16,1,.3,1)',
        direction: reverse ? 'reverse' : 'normal',
        fill: 'both',
      }
    );
  }

  // li always indexes the gallery the user entered from, even while its paired image is on screen.
  let lbVariant = null;
  let lbVisitedPair = false;
  const previewItem = () => lbVariant || st.list[li];
  const lbV = { zoom: 1, x: 0, y: 0, flip: false, grid: 0, gray: false, cmp: false };

  function lbApplyView() {
    const stage = $('#lbStage');
    const pImg = $('#lbPairImg');
    const gridEl = $('#lbGrid');
    const tools = $('#lbTools');
    if (!stage) return;

    const zoomed = lbV.zoom > 1;
    stage.classList.toggle('zoomed', zoomed);
    stage.classList.toggle('flipped', lbV.flip);
    stage.classList.toggle('gray', lbV.gray);
    stage.classList.toggle('comparing', lbV.cmp);
    stage.style.setProperty('--lb-tx', zoomed ? lbV.x + 'px' : '0px');
    stage.style.setProperty('--lb-ty', zoomed ? lbV.y + 'px' : '0px');
    stage.style.setProperty('--lb-z', String(lbV.zoom));
    if (gridEl) gridEl.dataset.mode = ['', 'thirds', 'cross'][lbV.grid] || '';

    const cmpWrap = $('#lbPairCmp');
    const hdl = $('#lbPairHdl');
    const rng = $('#lbPairRange');
    if (cmpWrap) cmpWrap.hidden = !lbV.cmp;
    if (hdl) hdl.hidden = !lbV.cmp;
    if (rng) rng.hidden = !lbV.cmp;
    if (!lbV.cmp && pImg) pImg.removeAttribute('src');

    if (tools) {
      const setBtn = (k, on) => {
        const b = tools.querySelector(`[data-lt="${k}"]`);
        if (!b) return;
        b.classList.toggle('on', !!on);
        b.setAttribute('aria-pressed', String(!!on));
      };
      setBtn('zoom', zoomed);
      setBtn('flip', lbV.flip);
      setBtn('grid', lbV.grid > 0);
      setBtn('gray', lbV.gray);
      setBtn('cmp', lbV.cmp);
    }
  }

  function lbResetView(hideTools = false) {
    lbV.zoom = 1;
    lbV.x = 0;
    lbV.y = 0;
    lbV.flip = false;
    lbV.grid = 0;
    lbV.gray = false;
    lbV.cmp = false;
    const stage = $('#lbStage');
    const rng = $('#lbPairRange');
    const tools = $('#lbTools');
    const box = $('.lb-img');
    if (stage) stage.style.setProperty('--x', '50%');
    if (rng) rng.value = '50';
    if (hideTools && tools) tools.hidden = true;
    if (hideTools && box) box.removeAttribute('data-img-type');
    lbApplyView();
  }

  function lbDrawPalette(it) {
    window.PoseWearPalette?.render($('#lbPal'), $('#lbImg'), it, () => !$('#lb').hidden && previewItem() === it);
  }

  function lbSyncCompare(it) {
    const cmpBtn = $('#lbTools [data-lt="cmp"]');
    const grayBtn = $('#lbTools [data-lt="gray"]');
    const mate = it && poseCounterpart(it);
    const mono = it && (it.type === 'pose' || it.type === 'depth');
    if (grayBtn) grayBtn.hidden = !!mono;
    if (mono && lbV.gray) {
      lbV.gray = false;
      lbApplyView();
    }
    if (cmpBtn) cmpBtn.hidden = !mate;
    if (!mate) {
      if (lbV.cmp) {
        lbV.cmp = false;
        lbApplyView();
      }
      return;
    }
    if (lbV.cmp) {
      const pImg = $('#lbPairImg');
      const view = mate.item.angles[mate.angle];
      pImg.referrerPolicy = mate.item.ref || '';
      pImg.src = view.i || view.t;
      $('#lbPairCmp').dataset.pairType = mate.item.type;
    }
  }

  function lbToggleTool(action) {
    if (st.tab === 'history' || $('#lb').hidden) return;
    const it = previewItem();
    if (!ok(it)) return;
    if (action === 'zoom') {
      if (lbV.cmp) lbV.cmp = false;
      lbV.zoom = lbV.zoom > 1 ? 1 : 2;
      lbV.x = 0;
      lbV.y = 0;
    } else if (action === 'flip') {
      lbV.flip = !lbV.flip;
    } else if (action === 'grid') {
      lbV.grid = (lbV.grid + 1) % 3;
    } else if (action === 'gray') {
      if (it.type === 'pose' || it.type === 'depth') return;
      lbV.gray = !lbV.gray;
    } else if (action === 'cmp') {
      const mate = poseCounterpart(it);
      if (!mate) return;
      lbV.cmp = !lbV.cmp;
      if (lbV.cmp) {
        lbV.zoom = 1;
        lbV.x = 0;
        lbV.y = 0;
        $('#lbStage').style.setProperty('--x', '50%');
        $('#lbPairRange').value = '50';
        lbSyncCompare(it);
      }
    }
    lbApplyView();
  }

  $('#lbTools')?.addEventListener('click', e => {
    const b = e.target.closest('[data-lt]');
    if (b) lbToggleTool(b.dataset.lt);
  });
  $('#lbPairRange')?.addEventListener('input', e => {
    $('#lbStage').style.setProperty('--x', e.target.value + '%');
  });

  {
    const stage = $('#lbStage');
    let drag = null;
    stage?.addEventListener('dblclick', e => {
      if (st.tab === 'history' || lbV.cmp || e.target.closest('#lbPairRange')) return;
      lbToggleTool('zoom');
    });
    stage?.addEventListener('pointerdown', e => {
      if (lbV.zoom <= 1 || e.button !== 0 || e.target.closest('#lbPairRange')) return;
      drag = { id: e.pointerId, sx: e.clientX, sy: e.clientY, ox: lbV.x, oy: lbV.y };
      stage.setPointerCapture(e.pointerId);
      stage.classList.add('panning');
    });
    stage?.addEventListener('pointermove', e => {
      if (!drag || e.pointerId !== drag.id) return;
      const r = stage.getBoundingClientRect();
      const limX = r.width * 0.42;
      const limY = r.height * 0.42;
      lbV.x = Math.max(-limX, Math.min(limX, drag.ox + (e.clientX - drag.sx)));
      lbV.y = Math.max(-limY, Math.min(limY, drag.oy + (e.clientY - drag.sy)));
      stage.style.setProperty('--lb-tx', lbV.x + 'px');
      stage.style.setProperty('--lb-ty', lbV.y + 'px');
    });
    const endDrag = e => {
      if (drag && e.pointerId === drag.id) {
        drag = null;
        stage.classList.remove('panning');
      }
    };
    stage?.addEventListener('pointerup', endDrag);
    stage?.addEventListener('pointercancel', endDrag);
  }

  function drawPosePair(it, slot) {
    const type = counterpartType(it);
    if (!type || !ok(it)) return;
    const show = () => {
      if (!slot.isConnected || $('#lb').hidden || closing || previewItem() !== it || !poseCounterpart(it)) return;
      slot.innerHTML =
        '<div class="seg" role="group" aria-label="姿势图类型">' +
        [['pose', '线稿'], ['depth', '深度图']]
          .map(([kind, label]) => `<button data-variant="${kind}" class="${kind === it.type ? 'on' : ''}" aria-pressed="${kind === it.type}">${label}</button>`)
          .join('') +
        '</div>';
      lbSyncCompare(it);
    };
    if (DATA[type]) show();
    else load(type).then(show, () => {});
  }

  let lbPushed = false;
  const pend = new Set();

  function lbOpen(i, cardEl) {
    if (!ok(st.list[i])) return;
    li = i;
    lbVariant = cardEl?.dataset.show ? ITEMS[cardEl.dataset.show] : null;
    lbVisitedPair = !!lbVariant;
    const it = previewItem();
    const h = `#${it.type}/${it.id}`;
    if (location.hash !== h && !location.hash.includes('/')) {
      history.pushState({ galleryItem: true }, '', h);
      lbPushed = true;
    }
    const lb = $('#lb');
    if (lb.hidden) srcCard = document.activeElement;
    lbResetView();
    lb.hidden = false;
    document.body.style.overflow = 'hidden';
    lbDraw();
    if (!motionPreference.matches) {
      $('.lb-bg').animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
      rectAnim($('.lb-box'), cardEl && cardEl.querySelector('.ph').getBoundingClientRect());
    }
    $('.lb-x').focus({ preventScroll: true });
  }

  let closing = false;
  function lbClose(navigateOut = false, immediate = false) {
    if ($('#lb').hidden || closing) return;
    const leaving = navigateOut === true;
    const fromHistory = st.tab === 'history';
    if (lbPushed) {
      lbPushed = false;
      if (!leaving) history.back();
    }
    const it = st.list[li];
    const cardEl = it && grid.querySelector(`.card[data-k="${CSS.escape(it.k)}"]`);
    const rect = cardEl && cardEl.querySelector('.ph').getBoundingClientRect();
    const visible = rect && rect.bottom > 0 && rect.top < innerHeight;

    const done = () => {
      closing = false;
      lbVariant = null;
      lbResetView(true);
      $('#lb').hidden = true;
      document.body.style.overflow = '';
      $('.lb-box').getAnimations().forEach(a => a.cancel());
      if (!leaving && location.hash.includes('/')) history.replaceState(null, '', '#' + st.tab);
      if (!leaving && lbVisitedPair && (st.tab === 'recent' || (st.tab === 'fav' && pend.size))) apply();
      else if (st.tab === 'fav') pend.forEach(k => !st.fav.has(k) && dropCard(k));
      lbVisitedPair = false;
      pend.clear();
      if (fromHistory) {
        window.HIS?.closed?.(!leaving);
      } else {
        const img = $('#lbImg');
        delete img.dataset.src;
        img.removeAttribute('srcset');
        img.removeAttribute('src');
        $('#lbSide').replaceChildren();
        if (!leaving) {
          const back = srcCard?.isConnected && srcCard !== document.body
            ? srcCard
            : it && grid.querySelector(`.card[data-k="${CSS.escape(it.k)}"] .phk`);
          back?.focus({ preventScroll: true });
        }
      }
    };

    if (motionPreference.matches || leaving || immediate) return done();
    closing = true;
    $('.lb-bg').animate([{ opacity: 1 }, { opacity: 0 }], { duration: 220, fill: 'both' }).finished.then(a => a.cancel());
    rectAnim($('.lb-box'), visible ? rect : null, true).finished.then(done);
  }

  function lbDraw() {
    const it = previewItem();
    if (!ok(it) || st.tab === 'history') return;
    const img = $('#lbImg');
    const box = $('.lb-img');
    const src = big(it);
    const isDepth = it.type === 'depth';
    addRec(it.k);
    history.replaceState(history.state, '', `#${it.type}/${it.id}`);

    box.dataset.imgType = it.type;
    box.classList.toggle('ext', !!it.ext);
    box.classList.remove('fail');
    if (it.site) $('#lbMiss').href = it.site;
    if ($('#lbTools')) $('#lbTools').hidden = false;
    lbSyncCompare(it);
    lbApplyView();

    img.referrerPolicy = it.ref || '';
    img.onerror = () => {
      if (isDepth && img.dataset.src === src) {
        box.classList.add('fail');
        box.classList.remove('loading');
      }
    };
    if (img.dataset.src !== src) {
      img.dataset.src = src;
      img.removeAttribute('srcset');
      img.src = thumb(it);
      img.classList.add('lo');
      box.classList.add('loading');
      preload(src).then(loaded => {
        if (img.dataset.src !== src) return;
        if (loaded) {
          img.src = loaded;
          if (it.type === 'costume' || it.type === 'anime') {
            img.sizes = '(max-width:760px) 100vw, 60vw';
            img.srcset = `${it.m} 1080w, ${it.i} 1366w`;
          }
          img.classList.remove('lo');
          box.classList.remove('loading', 'fail');
          lbDrawPalette(it);
        } else {
          box.classList.toggle('fail', isDepth);
          box.classList.remove('loading');
          img.classList.remove('lo');
        }
      });
    }

    [1, -1, 2].forEach(d => {
      const next = st.list[(li + d + st.list.length) % st.list.length];
      if (ok(next)) preload(big(next));
    });
    if (it.angles) it.angles.forEach(a => preload(a.i));

    const isPose = it.type === 'pose';
    const isFav = st.fav.has(it.k);
    const isSel = st.sel.includes(it.k);
    img.alt = title(it);

    const cpSvg = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M15 5V4H5v11h1"/></svg>';
    const zhBox = `<div class="pbox"><div class="pbox-h"><small>中文描述</small><button class="sm pbox-cp pri" data-c="zh">${cpSvg}<span>复制中文</span></button></div><p>${esc(prompt(it, 'zh'))}</p></div>`;
    const enBox = `<div class="pbox en"><div class="pbox-h"><small>英文提示词 · PROMPT</small><button class="sm pbox-cp" data-c="en">${cpSvg}<span>复制英文</span></button></div><p>${esc(prompt(it, 'en'))}</p></div>`;

    const hasAng = !!it.angles;
    const meta = hasAng
      ? `${it.line} · ${it.cat}${isDepth ? ' · ' + it.angles[it.ai].zh : ''} · 编号 ${it.no}`
      : it.type === 'portrait'
        ? `${it.sec} · ${it.len} · 编号 ${it.no}${it.rk ? ' · 人气第 ' + it.rk + ' 名' : ''}`
        : `${it.cat} · 编号 ${it.id}`;

    const jaLine = isDepth
      ? `<div class="jp">日文原名：${esc(it.ja)}</div><div class="jp">${esc(it.lineJa)} · ${esc(it.catJa)} · ${esc(it.angles[it.ai].ja)}</div>`
      : (isPose ? `<div class="jp">日文原名：${esc(it.ja)}</div>` : '');

    const sourceCredit = it.materialSource
      ? `<p class="creditln">${esc(it.materialCredit)} · <button type="button" class="sm ghost" data-material-source="${esc(it.materialSource)}">在原站查看</button></p>`
      : isDepth
      ? `<p class="creditln">${esc(it.unofficial || '非官方汉化')} · ${esc(it.credit || '')} <a href="${esc(it.site)}" target="_blank" rel="noopener">在原站查看</a></p>`
      : it.type === 'portrait'
        ? '<p class="creditln">CC0 1.0，AI 生成的虚构人物，请勿用于冒充真人</p>'
        : '';

    const angHtml = hasAng
      ? `<div class="angsel" role="group" aria-label="拍摄角度">${it.angles.map((a, i) => `<button class="chip ${i === it.ai ? 'on' : ''}" data-ai="${i}" aria-pressed="${i === it.ai}">${esc(a.zh)}</button>`).join('')}</div>`
      : '';

    const taobaoQuery = encodeURIComponent([...val(it, 'col').slice(0, 1), ...val(it, 'item').slice(0, 2)].join(' ') || `女装 ${it.cat}`);
    const downloadActs = desktop
      ? (it.previewOnly ? '<span class="mu">原站预览</span>' : '<button data-x="dlx">保存图片</button><button data-x="cpimg">复制图片</button>')
      : isDepth ? '' : (it.ext
          ? ''
          : `<button data-x="dlx">下载图片</button><button data-x="cpimg">复制图片</button>` +
            `<a href="${orig(it)}" download="posewear-${it.type}-${it.id}${isPose ? '-' + it.ai : ''}.png" class="dl"><button tabindex="-1">下载原图 PNG</button></a>`) +
        (isPose || it.type === 'portrait'
          ? ''
          : `<a href="https://s.taobao.com/search?q=${taobaoQuery}" target="_blank" rel="noopener" class="dl"><button tabindex="-1">淘宝搜同款 ↗</button></a>`);

    $('#lbSide').innerHTML =
      `<div class="mu">${TNAME[st.tab]} · 第 ${li + 1} / ${st.list.length} 项</div>` +
      `<h2>${esc(title(it))}</h2>` +
      `<div class="mu">${esc(meta)}</div>${jaLine}` +
      `<div id="lbPal" class="lb-pal" hidden></div>` +
      `${isPose || isDepth ? '<div id="lbPair"></div>' : ''}${angHtml}` +
      `${st.lang === 'en' ? enBox + zhBox : zhBox + enBox}` +
      `${hasAng ? '' : it.type === 'portrait' ? `<div class="jp">日文原名：${esc(it.ja)}</div>` : `<div class="jp">原始关键词：${esc(it.kw)}</div>`}` +
      `<div class="mact"><button data-x="fav" class="${isFav ? 'on' : ''}" aria-label="${isFav ? '取消收藏' : '收藏'}">${isFav ? '♥' : '♡'}</button><button data-x="share">分享</button>${isDepth || it.previewOnly ? '' : `<button data-x="try" class="pri">${tryLabel(it)}</button>`}</div>` +
      `<div class="acts"><button data-x="fav">${isFav ? '♥ 已收藏' : '♡ 收藏'}</button>${it.previewOnly ? '' : `<button data-x="sel" class="${isSel ? '' : 'pri'}">${isSel ? '✓ 已在组合中' : '＋ 加入组合'}</button>`}${isDepth || it.previewOnly ? '' : `<button data-x="try">${tryLabel(it)}</button>`}<button data-x="share">${desktop ? '复制素材提示词' : '分享链接'}</button>${downloadActs}</div>` +
      `<div id="lbSim"></div>` +
      `<p class="mu lb-keys"><kbd>←</kbd><kbd>→</kbd>切换　${hasAng ? '<kbd>↑</kbd><kbd>↓</kbd>角度　' : ''}<kbd>Z</kbd>缩放　<kbd>H</kbd>镜像　<kbd>G</kbd>参考线　${isPose || isDepth ? '<kbd>V</kbd>对比　' : '<kbd>B</kbd>灰度　'}<kbd>F</kbd>收藏　<kbd>A</kbd>组合　<kbd>C</kbd>复制　<kbd>S</kbd>分享</p>` +
      sourceCredit;

    if (isPose || isDepth) drawPosePair(it, $('#lbPair'));
    else lbDrawPalette(it);
    window.X?.lb?.(it);
  }

  function share() {
    const it = previewItem();
    if (desktop) { copyBrief(prompt(it, st.lang), [it], st.lang, '素材提示词'); return; }
    const url = new URL(location.href);
    if (counterpartType(it)) url.hash = `${it.type}/${it.id}/${encodeURIComponent(it.angles[it.ai].zh)}`;
    copy(url.href, '分享链接');
  }

  $('#lbSide').onclick = e => {
    const it = previewItem();
    const btn = e.target.closest('button');
    if (!btn || !ok(it) || st.tab === 'history') return;
    if (btn.dataset.hex) {
      copy(btn.dataset.hex, btn.dataset.hex);
      btn.classList.add('copied');
      clearTimeout(btn._ct);
      btn._ct = setTimeout(() => btn.classList.remove('copied'), 1000);
      return;
    }
    if (btn.dataset.variant) {
      const pair = poseCounterpart(it);
      if (!pair || pair.item.type !== btn.dataset.variant) return;
      lbVisitedPair = true;
      lbVariant = pair.item;
      setAng(pair.item, pair.angle);
      lbDraw();
      $('#lbPair [aria-pressed="true"]')?.focus({ preventScroll: true });
      return;
    }
    if (btn.dataset.ai) {
      setAng(it, +btn.dataset.ai);
      lbDraw();
      $(`#lbSide [data-ai="${it.ai}"]`).focus({ preventScroll: true });
    }
    if (btn.dataset.c) {
      copyBrief(prompt(it, btn.dataset.c), [it], btn.dataset.c, btn.dataset.c === 'en' ? '英文提示词' : '中文描述');
      btn.classList.add('copied');
      clearTimeout(btn._ct);
      btn._ct = setTimeout(() => btn.classList.remove('copied'), 1200);
    }
    if (btn.dataset.x === 'fav') toggleFav(it.k);
    if (btn.dataset.x === 'sel') toggleSel(it.k);
    if (btn.dataset.x === 'share') share();
    if (btn.dataset.x === 'try') {
      lbClose(!!STUDIO_SLOT[it.type]);
      useReference(it);
    }
  };

  const step = delta => {
    li = (li + delta + st.list.length) % st.list.length;
    lbVariant = null;
    lbResetView();
    lbDraw();
    if (!motionPreference.matches) {
      $('#lbImg').animate(
        [{ opacity: 0.3, transform: `translateX(${delta * 30}px)` }, { opacity: 1, transform: 'none' }],
        { duration: 320, easing: 'cubic-bezier(.16,1,.3,1)' }
      );
    }
  };

  $('.lb-bg').onclick = lbClose;
  $('.lb-x').onclick = lbClose;
  $('.prev').onclick = () => step(-1);
  $('.next').onclick = () => step(1);

  // ---------- Global Keyboard Shortcuts ----------
  addEventListener('keydown', e => {
    const k = (e.key || '').toLowerCase();
    if ((e.ctrlKey || e.metaKey) && !e.altKey && k === 'k') {
      e.preventDefault();
      window.PoseWearCommandPalette?.isOpen()
        ? window.PoseWearCommandPalette.close()
        : window.PoseWearCommandPalette?.open();
      return;
    }
    if (window.PoseWearCommandPalette?.handleKey(e)) return;
    if (window.PoseWearSketch?.handleKey(e)) return;

    if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('.shell-menu[open]')) return;
    const ae = document.activeElement;
    const inInput = ae && /^(INPUT|SELECT|TEXTAREA)$/.test(ae.tagName);
    if (!$('#cfg').hidden || !$('#pk').hidden) return;

    const modal = $('#ageBox:not([hidden]),#privBox:not([hidden]),#helpBox:not([hidden]),#tgView:not([hidden])');
    if (modal) {
      if (e.key === 'Escape' || (modal.id === 'helpBox' && e.key === '?')) {
        e.preventDefault();
        closeDialog(modal);
        if (modal.id === 'ageBox') ageCb = null;
      } else if (e.key === 'Tab') {
        trapTab(e, modal);
      }
      return;
    }

    if (trayOpen && $('#lb').hidden && $('#tray').contains(ae) && e.key === 'Escape') {
      e.preventDefault();
      closeTray();
      return;
    }
    if (!$('#lb').hidden && e.key === 'Tab') {
      trapTab(e, $('#lb'));
      return;
    }
    if (e.key === 'Escape') {
      if (!$('#lb').hidden) return lbClose();
      drawer(false);
      if (inInput) ae.blur();
      return;
    }
    if (inInput) return;
    if (e.key === '?') {
      openDialog($('#helpBox'));
      return;
    }

    if ($('#lb').hidden) {
      if (e.key === '/' && FDEF[st.tab]) { e.preventDefault(); $('#q').focus(); }
      else if (k === 'r' && FDEF[st.tab] && !$('#rand').hidden) randPick();
      else if (k === 't') window.PoseWearTheme.toggle();
      else if (k === 'l') setLang(st.lang === 'zh' ? 'en' : 'zh');
      else if ('123456789'.includes(e.key) && TABS[e.key - 1]) navigate(TABS[e.key - 1]);
      return;
    }

    if (st.tab === 'history') return;
    const it = previewItem();
    if (!ok(it)) return;
    if (e.key === 'ArrowLeft') step(-1);
    else if (e.key === 'ArrowRight') step(1);
    else if (/Arrow(Up|Down)/.test(e.key) && it.angles) {
      e.preventDefault();
      setAng(it, (it.ai + (e.key === 'ArrowUp' ? -1 : 1) + it.angles.length) % it.angles.length);
      lbDraw();
    } else if (k === 'z') lbToggleTool('zoom');
    else if (k === 'h') lbToggleTool('flip');
    else if (k === 'g') lbToggleTool('grid');
    else if (k === 'b') lbToggleTool('gray');
    else if (k === 'v') lbToggleTool('cmp');
    else if (k === 't') window.PoseWearTheme.toggle();
    else if (k === 'f') toggleFav(it.k);
    else if (k === 'a') toggleSel(it.k);
    else if (k === 'c') copyBrief(prompt(it, 'zh'), [it], 'zh', '中文描述');
    else if (k === 's') share();
  });

  // Touch swipe inside lightbox
  let tx = 0;
  let ty = 0;
  let tSkip = false;
  $('.lb-img').addEventListener('touchstart', e => {
    tSkip = lbV.zoom > 1 || !!e.target.closest('#lbPairRange,#lbTools');
    tx = e.touches[0].clientX;
    ty = e.touches[0].clientY;
  }, { passive: true });
  $('.lb-img').addEventListener('touchend', e => {
    if (tSkip) return;
    const dx = e.changedTouches[0].clientX - tx;
    const dy = e.changedTouches[0].clientY - ty;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) (dx < 0 ? $('.next') : $('.prev')).click();
    else if (dy > 90) lbClose();
  });

  // ---------- Routing ----------
  function openById(id) {
    const it = st.list.find(x => x.id === id) || (DATA[st.tab] || []).find(x => x.id === id);
    if (!it) return toast('未找到该条目');
    if (!ok(it)) return ageAsk(() => openById(id));
    const sharedAngle = location.hash.split('/')[2];
    if (sharedAngle && counterpartType(it)) {
      const angle = it.angles.findIndex(a => encodeURIComponent(a.zh) === sharedAngle);
      if (angle >= 0) setAng(it, angle);
    }
    if (!st.list.includes(it)) {
      st.f = {};
      st.q = '';
      $('#q').value = '';
      drawFilters();
      apply();
    }
    lbOpen(st.list.indexOf(it));
  }

  function route() {
    const [t, id, ref] = location.hash.slice(1).split('/');
    if (t === 'workbench') return setTab(Object.hasOwn(PFX, id) ? id : 'pose', null, ref || '');
    const tab = TABS.includes(t) ? t : 'pose';
    if (!id && tab === st.tab && st.list.length) return;
    if (tab !== st.tab || !st.list.length || !id) setTab(tab, id && decodeURIComponent(id));
    else openById(decodeURIComponent(id));
  }

  // ---------- Adult Content Switch & Privacy Modal ----------
  function ageSet(on, animate) {
    st.adult = on;
    SV('adult', on);
    if (on) SV('ageOk', true);
    document.body.classList.toggle('adult', on);
    $('#adultSw').setAttribute('aria-checked', on);
    $('#adultSw').title = on ? '成人向内容已显示，点击关闭' : '成人向内容已隐藏，点击打开';
    counts();

    const origin = !$('#lb').hidden && st.tab !== 'history' && st.list[li];
    if (origin && (!ok(origin) || !ok(previewItem()))) lbClose(false, true);
    if (FDEF[st.tab]) {
      const update = () => {
        if (!FDEF[st.tab]) return;
        drawFilters();
        apply();
        tray();
        if (origin && !$('#lb').hidden && !closing) {
          li = st.list.indexOf(origin);
          if (li < 0) lbClose(false, true);
          else lbDraw();
        }
      };
      if (animate) viewT(update);
      else update();
    }
    if (st.tab === 'studio') window.WB?.show?.();
  }

  let ageCb = null;
  function ageAsk(cb) {
    ageCb = cb || null;
    const seen = LS('ageOk');
    $('#ageT').textContent = cb ? '这是成人向内容' : '显示成人向内容？';
    $('#ageP').textContent =
      (cb
        ? '要查看它，需要打开顶部的「18+」开关。'
        : '包括双人互动、性感类姿势、深度姿势里的双人图，以及内衣、性感类服装。') +
      (seen ? '打开后随时可以再关上。' : '仅适合已满 18 岁的用户。只需确认一次，之后用顶部开关随时打开或关闭。');
    $('#ageY').textContent = seen ? '打开 18+ 开关' : '我已满 18 岁，打开';
    openDialog($('#ageBox'), $('#ageY'));
  }

  $('#adultSw').onclick = () => {
    if (st.adult) { ageSet(false, 1); toast('已关闭成人向内容'); }
    else if (LS('ageOk')) { ageSet(true, 1); toast('已显示成人向内容'); }
    else ageAsk();
  };
  $('#ageY').onclick = () => {
    closeDialog($('#ageBox'));
    ageSet(true, !ageCb);
    toast('已显示成人向内容，可用顶部开关关闭');
    const cb = ageCb;
    ageCb = null;
    cb?.();
  };
  $('#ageN').onclick = $('#ageBox').onclick = e => {
    if (e.target.id !== 'ageN' && e.target.id !== 'ageBox') return;
    closeDialog($('#ageBox'));
    ageCb = null;
  };
  document.addEventListener('click', e => {
    if (!e.target.closest('[data-age]')) return;
    if (st.adult) { ageSet(false); toast('已隐藏成人内容'); }
    else ageAsk();
  });
  $('#privO').onclick = e => { e.preventDefault(); openDialog($('#privBox')); };
  $('#privBox').onclick = e => {
    if (e.target.id === 'privBox' || e.target.closest('[data-close]')) closeDialog($('#privBox'));
  };

  addEventListener('hashchange', () => {
    const id = location.hash.split('/')[1];
    if (!$('#lb').hidden) {
      if (id) return;
      lbPushed = false;
      lbClose();
      setTimeout(route, closing ? 280 : 0);
      return;
    }
    route();
  });

  // ---------- Bootstrap & Public API ----------
  if (innerWidth <= 760) $('#q').placeholder = '搜索';
  setLang(st.lang, true);
  ageSet(st.adult);
  hdr();
  needKeys([...st.sel]).then(() => tray()).catch(() => {});
  st.tab = '';
  if (Q0 && !location.hash) {
    history.replaceState(null, '', `?q=${encodeURIComponent(Q0)}#${TABS.includes(QT) ? QT : 'costume'}`);
  }
  route();
  if (!desktop && 'serviceWorker' in navigator && location.protocol !== 'file:') {
    addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
  }

  window.APP = {
    ok,
    counterpartType,
    poseCounterpart,
    addLocalItem,
    refreshMaterials,
    cardFace,
    studioSlot,
    suggestions: SUGG,
    sendToStudio,
    ageAsk,
    toggleFav,
    toggleSel,
    lbOpen,
    lbClose,
    lbResetView,
    sketchOpen: opts => window.PoseWearSketch?.open(opts),
    sketchClose: () => window.PoseWearSketch?.close(),
    cmdOpen: () => window.PoseWearCommandPalette?.open(),
    cmdClose: () => window.PoseWearCommandPalette?.close(),
    openItem: (type, id) => setTab(type, id),
    applySearch,
    isTrayOpen: () => trayOpen,
    openTray,
    closeTray,
    setLang,
    trapTab,
    preload,
    previewItem,
    apply,
    li: () => li,
    load,
    ITEMS,
    DATA,
    thumb,
    big,
    ath,
    title,
    prompt,
    toast,
    esc,
    setTab: navigate,
    st,
    copy,
    copyBrief,
    needKeys,
  };
})();
