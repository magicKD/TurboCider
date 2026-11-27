'use strict';
(() => {
  // Fullscreen Gesture Drawing Practice Studio (#sketch).
  // Reuses the currently filtered gallery pool with configurable timer, count,
  // gentle Web Audio chime, mirror/grid/grayscale/depth-flip study tools, and
  // an end-of-session clickable recap contact sheet.
  const C = window.PoseWearCatalog;
  if (!C) return;
  const { $, $$, esc, shuffle } = C;

  const state = {
    open: false,
    dur: 60,
    total: 10,
    pool: [],
    deck: [],
    i: 0,
    seen: [],
    rem: 60,
    last: 0,
    paused: false,
    sound: false,
    flip: false,
    grid: 0,
    gray: false,
    pair: false,
    timer: null,
    actx: null,
    beeped: 0,
    prevFocus: null,
  };

  let app = null;

  function chime(freq = 660, duration = 0.12) {
    if (!state.sound) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      state.actx ||= new AudioCtx();
      if (state.actx.state === 'suspended') state.actx.resume();
      const osc = state.actx.createOscillator();
      const gain = state.actx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, state.actx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.08, state.actx.currentTime + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, state.actx.currentTime + duration);
      osc.connect(gain);
      gain.connect(state.actx.destination);
      osc.start();
      osc.stop(state.actx.currentTime + duration + 0.02);
    } catch {}
  }

  function formatTime(sec) {
    const s = Math.max(0, Math.ceil(sec));
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }

  function sessionCap() {
    return state.total ? Math.min(state.total, state.deck.length) : 0;
  }

  function renderTools() {
    const it = state.deck[state.i];
    const frame = $('#skFrame');
    const gridEl = $('#skGrid');
    if (frame) {
      frame.classList.toggle('flipped', state.flip);
      frame.classList.toggle('gray', state.gray);
    }
    if (gridEl) gridEl.dataset.mode = ['', 'thirds', 'cross'][state.grid] || '';

    const setActive = (key, on) => {
      const btn = $(`#sketch [data-skt="${key}"]`);
      if (!btn) return;
      btn.classList.toggle('on', !!on);
      btn.setAttribute('aria-pressed', String(!!on));
    };
    setActive('flip', state.flip);
    setActive('grid', state.grid > 0);
    setActive('gray', state.gray);
    setActive('pair', state.pair);

    const favBtn = $('#sketch [data-skt="fav"]');
    const isFav = !!(it && app.st.fav.has(it.k));
    if (favBtn) {
      favBtn.classList.toggle('on', isFav);
      favBtn.setAttribute('aria-pressed', String(isFav));
      favBtn.textContent = isFav ? '♥ 已收藏' : '♡ 收藏';
    }

    const pairBtn = $('#sketch [data-skt="pair"]');
    const mate = it && app.poseCounterpart(it);
    if (pairBtn) {
      pairBtn.hidden = !mate;
      if (mate) {
        const shownType = state.pair ? mate.item.type : it.type;
        pairBtn.textContent = shownType === 'pose' ? '深度图' : '线稿';
      }
    }

    const grayBtn = $('#sketch [data-skt="gray"]');
    const shown = state.pair && mate ? mate.item : it;
    if (grayBtn) grayBtn.hidden = !shown || shown.type === 'pose' || shown.type === 'depth';

    const soundBtn = $('#skSound');
    if (soundBtn) {
      soundBtn.classList.toggle('on', state.sound);
      soundBtn.setAttribute('aria-pressed', String(state.sound));
    }
  }

  function showCard() {
    const it = state.deck[state.i];
    if (!it) return;
    if (!state.seen.includes(it)) state.seen.push(it);
    const mate = app.poseCounterpart(it);
    const shown = state.pair && mate ? mate.item : it;
    const src = state.pair && mate
      ? (mate.item.angles[mate.angle].i || mate.item.angles[mate.angle].t)
      : app.big(it);

    const img = $('#skImg');
    const stage = $('#skStage');
    const frame = $('#skFrame');
    $('#skDone').hidden = true;
    frame.hidden = false;
    stage.dataset.imgType = shown.type;
    img.referrerPolicy = shown.ref || '';
    img.src = src;
    img.alt = app.title(shown);

    const cap = sessionCap();
    $('#skStep').textContent = `${state.i + 1} / ${cap || '∞'}`;
    const rawAng = it.angles?.[it.ai]?.zh;
    const locAng = rawAng ? (window.PWLocale ? window.PWLocale.t(rawAng) : rawAng) : '';
    $('#skName').textContent = app.title(it) + (locAng ? ' · ' + locAng : '');

    state.rem = state.dur;
    state.beeped = 0;
    state.last = performance.now();
    $('#skTime').textContent = formatTime(state.rem);
    $('#skProg').style.transform = 'scaleX(1)';

    const nextItem = state.deck[state.i + 1];
    if (nextItem) app.preload(app.big(nextItem));

    const pairType = app.counterpartType(it);
    if (pairType && !app.DATA[pairType]) {
      app.load(pairType).then(() => state.open && renderTools(), () => {});
    }
    renderTools();
  }

  function finish() {
    clearInterval(state.timer);
    state.timer = null;
    state.paused = true;
    $('#skFrame').hidden = true;
    $('#skDone').hidden = false;
    $('#skProg').style.transform = 'scaleX(0)';
    const unit = state.dur >= 60 ? `${Math.round(state.dur / 60)} 分钟` : `${state.dur} 秒`;
    $('#skSummary').textContent = `已完成 ${state.seen.length} 张参考练习（每张 ${unit}）`;
    $('#skRecap').innerHTML = state.seen
      .map(it => `<button type="button" data-sk-open="${esc(it.k)}" title="${esc(app.title(it))}"><img src="${esc(app.thumb(it))}" alt="${esc(app.title(it))}"${it.ref ? ` referrerpolicy="${esc(it.ref)}"` : ''}></button>`)
      .join('');
    chime(784, 0.25);
  }

  function step(delta) {
    const next = state.i + delta;
    if (next < 0) return;
    if (state.total && next >= Math.min(state.total, state.deck.length)) {
      finish();
      return;
    }
    if (next >= state.deck.length) {
      const base = state.pool.length ? state.pool : app.st.list.filter(app.ok);
      if (!base.length) {
        finish();
        return;
      }
      state.deck = state.deck.concat(shuffle(base.slice()));
    }
    state.i = next;
    state.pair = false;
    showCard();
    if (delta > 0) chime(659, 0.14);
  }

  function tick() {
    if (!state.open || state.paused || document.hidden) return;
    const now = performance.now();
    const dt = Math.min(0.5, (now - state.last) / 1000);
    state.last = now;
    state.rem = Math.max(0, state.rem - dt);
    $('#skTime').textContent = formatTime(state.rem);
    $('#skProg').style.transform = `scaleX(${Math.max(0, state.rem / state.dur)})`;
    const ceil = Math.ceil(state.rem);
    if (ceil <= 3 && ceil >= 1 && state.beeped !== ceil) {
      state.beeped = ceil;
      chime(523, 0.07);
    }
    if (state.rem <= 0) step(1);
  }

  function togglePause(force) {
    state.paused = force === undefined ? !state.paused : !!force;
    state.last = performance.now();
    $('#skPause').textContent = state.paused ? '继续' : '暂停';
  }

  async function open(opts = {}) {
    if (!app) return;
    if (opts.dur) state.dur = opts.dur;
    if (opts.total !== undefined) state.total = opts.total;
    let pool = app.st.list.filter(app.ok);
    if (pool.length < 2) {
      if (!app.DATA.pose) await app.load('pose');
      pool = (app.DATA.pose || []).filter(app.ok);
    }
    if (!pool.length) return app.toast('当前没有可用于速写的素材');

    state.prevFocus = document.activeElement;
    state.pool = pool.slice();
    state.deck = shuffle(pool.slice());
    state.i = 0;
    state.seen = [];
    state.flip = false;
    state.grid = 0;
    state.gray = false;
    state.pair = false;
    state.open = true;
    state.paused = false;

    $('#sketch').hidden = false;
    document.body.style.overflow = 'hidden';
    $$('#skDur button').forEach(b => {
      const on = +b.dataset.sd === state.dur;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    });
    $$('#skTotal button').forEach(b => {
      const on = +b.dataset.sc === state.total;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    });
    $('#skPause').textContent = '暂停';

    showCard();
    clearInterval(state.timer);
    state.timer = setInterval(tick, 200);
    $('#skPause').focus({ preventScroll: true });
  }

  function close() {
    if (!state.open) return;
    state.open = false;
    clearInterval(state.timer);
    state.timer = null;
    $('#sketch').hidden = true;
    $('#skImg').removeAttribute('src');
    document.body.style.overflow = '';
    if (state.prevFocus?.isConnected) state.prevFocus.focus({ preventScroll: true });
    state.prevFocus = null;
  }

  function handleKey(e) {
    if (!state.open) return false;
    const k = (e.key || '').toLowerCase();
    if (e.key === 'Escape') { e.preventDefault(); close(); }
    else if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); togglePause(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
    else if (k === 'h') { state.flip = !state.flip; renderTools(); }
    else if (k === 'g') { state.grid = (state.grid + 1) % 3; renderTools(); }
    else if (k === 'b') { state.gray = !state.gray; renderTools(); }
    else if (k === 'v') {
      const it = state.deck[state.i];
      if (it && app.poseCounterpart(it)) {
        state.pair = !state.pair;
        showCard();
      }
    } else if (k === 'f') {
      const it = state.deck[state.i];
      if (it) {
        app.toggleFav(it.k);
        renderTools();
      }
    } else if (e.key === 'Tab') {
      app.trapTab(e, $('#sketch'));
    }
    return true;
  }

  function init(appApi) {
    app = appApi;
    document.addEventListener('visibilitychange', () => {
      if (state.open && !document.hidden) state.last = performance.now();
    });

    $('#sketchBtn')?.addEventListener('click', () => open());
    $('#skClose')?.addEventListener('click', close);
    $('#skExit')?.addEventListener('click', close);
    $('#skAgain')?.addEventListener('click', () => open());
    $('#skPause')?.addEventListener('click', () => togglePause());
    $('#skPrev')?.addEventListener('click', () => step(-1));
    $('#skNext')?.addEventListener('click', () => step(1));
    $('#skSound')?.addEventListener('click', () => {
      state.sound = !state.sound;
      renderTools();
      if (state.sound) chime(659, 0.1);
    });

    $('#skDur')?.addEventListener('click', e => {
      const btn = e.target.closest('[data-sd]');
      if (!btn) return;
      state.dur = +btn.dataset.sd;
      state.rem = state.dur;
      state.last = performance.now();
      state.beeped = 0;
      $$('#skDur button').forEach(x => {
        const on = x === btn;
        x.classList.toggle('on', on);
        x.setAttribute('aria-pressed', String(on));
      });
      if ($('#skDone').hidden) {
        $('#skTime').textContent = formatTime(state.rem);
        $('#skProg').style.transform = 'scaleX(1)';
      }
    });

    $('#skTotal')?.addEventListener('click', e => {
      const btn = e.target.closest('[data-sc]');
      if (!btn) return;
      state.total = +btn.dataset.sc;
      $$('#skTotal button').forEach(x => {
        const on = x === btn;
        x.classList.toggle('on', on);
        x.setAttribute('aria-pressed', String(on));
      });
      if ($('#skDone').hidden) $('#skStep').textContent = `${state.i + 1} / ${sessionCap() || '∞'}`;
    });

    $('#sketch')?.addEventListener('click', e => {
      const tool = e.target.closest('[data-skt]');
      if (tool) {
        const action = tool.dataset.skt;
        const it = state.deck[state.i];
        if (action === 'flip') state.flip = !state.flip;
        else if (action === 'grid') state.grid = (state.grid + 1) % 3;
        else if (action === 'gray') state.gray = !state.gray;
        else if (action === 'pair' && it && app.poseCounterpart(it)) {
          state.pair = !state.pair;
          showCard();
          return;
        } else if (action === 'fav' && it) {
          app.toggleFav(it.k);
        }
        renderTools();
        return;
      }
      const recap = e.target.closest('[data-sk-open]');
      if (recap) {
        const it = app.ITEMS[recap.dataset.skOpen];
        close();
        if (it) app.openItem(it.type, it.id);
      }
    });
  }

  window.PoseWearSketch = {
    init,
    open,
    close,
    isOpen: () => state.open,
    handleKey,
  };
  if (window.APP) init(window.APP);
})();
