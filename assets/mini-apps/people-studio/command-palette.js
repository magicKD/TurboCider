'use strict';
(() => {
  // Global Command Palette (⌘K / Ctrl+K) backed by PoseWearCatalog.
  const C = window.PoseWearCatalog;
  if (!C) return;
  const { $, esc, COLLECTIONS, UTILITY_TABS, PFX, FDEF, TNAME } = C;

  let app = null;
  let prevFocus = null;
  let activeIndex = 0;
  let visibleItems = [];

  function buildCatalog() {
    const list = [
      {
        g: '快捷操作',
        id: 'act:sketch',
        label: '开始速写定时练习（60s）',
        kw: 'sketch gesture timer croquis 速写 计时 练习 クロッキー',
        hint: '60s',
        run: () => app.sketchOpen({ dur: 60 }),
      },
      {
        g: '快捷操作',
        id: 'act:shuffle',
        label: '工作台 · 随机一组参考搭配',
        kw: 'studio shuffle random 工作台 灵感 随机 シャッフル スタジオ',
        run: async () => {
          app.setTab('studio');
          await new Promise(r => setTimeout(r, 80));
          $('#wbShuffle')?.click();
        },
      },
      {
        g: '快捷操作',
        id: 'act:dock',
        label: app.isTrayOpen() ? '收起悬浮创作台' : '展开悬浮创作台',
        kw: 'dock tray workspace 悬浮 创作台 パネル',
        run: () => (app.isTrayOpen() ? app.closeTray() : app.openTray()),
      },
      {
        g: '快捷操作',
        id: 'act:theme',
        label: '切换深浅外观',
        kw: 'theme dark light 外观 深色 浅色 ダーク ライト',
        hint: 'T',
        run: () => window.PoseWearTheme.toggle(),
      },
      {
        g: '快捷操作',
        id: 'act:lang',
        label: '切换素材描述语言（中 / 英）',
        kw: 'language prompt en zh 语言 英文 中文 言語',
        hint: 'L',
        run: () => app.setLang(app.st.lang === 'zh' ? 'en' : 'zh'),
      },
      {
        g: '快捷操作',
        id: 'act:cfg',
        label: '打开 API 与本机引擎设置',
        kw: 'api key comfyui sd.cpp settings 设置 配置 ローカル',
        run: () => window.WB?.cfgOpen?.(),
      },
    ];

    if (app.st.tab === 'studio' && !$('#wbExportBoard')?.hidden) {
      list.splice(1, 0, {
        g: '快捷操作',
        id: 'act:export',
        label: '导出参考画板拼图 PNG',
        kw: 'export board moodboard png 导出 拼图 書き出し',
        run: () => $('#wbExportBoard')?.click(),
      });
    }

    for (const col of COLLECTIONS) {
      const count = $(`#libraryMenu [data-tab="${col.id}"] [data-stat]`)?.textContent || '';
      list.push({
        g: '跳转栏目',
        id: 'tab:' + col.id,
        tab: col.id,
        label: col.title,
        kw: `${col.keywords} ${col.title} ${col.label}`,
        hint: count ? String(count) : '',
        run: () => app.setTab(col.id),
      });
    }

    for (const tab of UTILITY_TABS) {
      const count = tab.id === 'fav'
        ? app.st.fav.size
        : tab.id === 'recent'
          ? app.st.rec.length
          : tab.id === 'history'
            ? ($('#nHist')?.textContent || '')
            : '';
      list.push({
        g: '跳转栏目',
        id: 'tab:' + tab.id,
        tab: tab.id,
        label: tab.cmdLabel,
        kw: `${tab.keywords} ${tab.cmdLabel}`,
        hint: count ? String(count) : '',
        run: () => app.setTab(tab.id),
      });
    }

    return list;
  }

  function render() {
    const rawQuery = ($('#cmdQ').value || '').trim();
    const q = rawQuery.toLowerCase();
    const all = buildCatalog();
    const tr = window.PWLocale ? s => window.PWLocale.t(s) : s => s;

    visibleItems = q
      ? all.filter(item => `${item.label} ${tr(item.label)} ${item.kw}`.toLowerCase().includes(q))
      : all;

    if (q && FDEF[app.st.tab]) {
      visibleItems.unshift({
        g: '当前图库搜索',
        id: 'search',
        label: `搜索「${rawQuery}」`,
        kw: q,
        hint: TNAME[app.st.tab],
        run: () => app.applySearch(rawQuery),
      });
    }

    activeIndex = Math.min(activeIndex, Math.max(0, visibleItems.length - 1));
    const listEl = $('#cmdList');
    if (!visibleItems.length) {
      listEl.innerHTML = '<div class="cmd-empty">没有找到匹配的命令或栏目</div>';
      return;
    }

    let html = '';
    let lastGroup = '';
    visibleItems.forEach((item, index) => {
      if (item.g !== lastGroup) {
        html += `<div class="cmd-g">${esc(item.g)}</div>`;
        lastGroup = item.g;
      }
      const dot = item.tab && PFX[item.tab]
        ? `<i class="cmd-dot" data-tab="${item.tab}" aria-hidden="true"></i>`
        : '';
      const selected = index === activeIndex;
      const hint = item.hint ? `<small>${esc(item.hint)}</small>` : '';
      html += `<button type="button" class="cmd-it ${selected ? 'on' : ''}" role="option" aria-selected="${selected}" data-ci="${index}">${dot}<span>${esc(item.label)}</span>${hint}</button>`;
    });

    listEl.innerHTML = html;
    listEl.querySelector('.cmd-it.on')?.scrollIntoView({ block: 'nearest' });
  }

  function open() {
    const box = $('#cmdPal');
    if (!box || !box.hidden || !app) return;
    if (window.PoseWearSketch?.isOpen()) window.PoseWearSketch.close();
    prevFocus = document.activeElement;
    box.hidden = false;
    $('#cmdQ').value = '';
    activeIndex = 0;
    render();
    $('#cmdQ').focus({ preventScroll: true });
  }

  function close() {
    const box = $('#cmdPal');
    if (!box || box.hidden) return;
    box.hidden = true;
    if (prevFocus?.isConnected) prevFocus.focus({ preventScroll: true });
    prevFocus = null;
  }

  function execute(index) {
    const item = visibleItems[index];
    if (!item) return;
    close();
    item.run();
  }

  function isOpen() {
    const box = $('#cmdPal');
    return !!(box && !box.hidden);
  }

  function handleKey(e) {
    if (!isOpen()) return false;
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (visibleItems.length) {
        activeIndex = (activeIndex + 1) % visibleItems.length;
        render();
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (visibleItems.length) {
        activeIndex = (activeIndex - 1 + visibleItems.length) % visibleItems.length;
        render();
      }
    } else if (e.key === 'Enter') {
      e.preventDefault();
      execute(activeIndex);
    } else if (e.key === 'Tab') {
      app.trapTab(e, $('#cmdPal'));
    }
    return true;
  }

  function init(appApi) {
    app = appApi;
    $('#qKbd')?.addEventListener('click', open);
    $('#cmdQ')?.addEventListener('input', () => {
      activeIndex = 0;
      render();
    });
    $('#cmdList')?.addEventListener('click', e => {
      const btn = e.target.closest('[data-ci]');
      if (btn) execute(+btn.dataset.ci);
    });
    $('#cmdPal')?.addEventListener('click', e => {
      if (e.target.id === 'cmdPal') close();
    });
  }

  window.PoseWearCommandPalette = { init, open, close, isOpen, handleKey };
  if (window.APP) init(window.APP);
})();
