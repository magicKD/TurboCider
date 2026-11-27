'use strict';
(() => {
  const A = window.APP;
  const C = window.PoseWearCatalog;
  if (!A || !C) return;
  const {
    $, $$, esc, readLS: LS, writeLS: SV,
    BOARD_LABELS: boardLabels,
    GARMENT_TYPES: garmentTypes,
    OTHER_TYPES: otherTypes,
    MATERIAL_ROLES,
  } = C;
  const toast = A.toast;
  const desktop = window.PeopleStudioNative;

  // ---------- Image Generation Providers ----------
  const { providers, generateImage, testConnection, nice } = window.PoseWearImageProviders;
  const PROV = { ...providers };
  const localEngine = document.documentElement.dataset.localEngine;
  const localPreset = document.documentElement.dataset.localPreset === '1';

  $('#cfLocalSetup').hidden = !!localEngine;
  if (localEngine) {
    const name = localEngine === 'comfyui' ? 'ComfyUI' : 'sd.cpp';
    PROV.local = { name, base: '', model: name, models: [] };
    const label = document.createElement('label');
    label.innerHTML =
      `<input type="radio" name="prov" value="local"><span>` +
      `<i class="lg local-lg"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="12" rx="2.5"/><path d="M8 20h8m-4-4v4"/></svg></i>` +
      `<b>${name}</b><small>本机引擎</small></span>`;
    $('#cfg .prov').append(label);
  }

  const configured = (settings = cfg) =>
    !!desktop || (settings.prov === 'local'
      ? !!localEngine && (!localPreset || !!settings.checkpoint)
      : !!settings.key);

  const defaultConfig = {
    prov: localEngine && !localPreset ? 'local' : 'openai',
    base: '',
    key: '',
    model: '',
    field: 'image[]',
    proto: 'edits',
    proxy: false,
  };
  const storedConfig = LS('wbCfg');
  const cfg = { ...defaultConfig, ...(Object.hasOwn(PROV, storedConfig?.prov) ? storedConfig : {}) };

  // ---------- Presets & Studio State ----------
  const CHARS = [
    ['asia', '东亚女性', 'an adult East Asian woman in her mid-20s, long black hair, natural makeup, slim build', '一位二十多岁的成年东亚女性，黑色长发，自然淡妆，身材纤细'],
    ['west', '欧美女性', 'an adult Caucasian woman in her late 20s, wavy light-brown hair, natural makeup, athletic build', '一位二十多岁的成年欧美女性，浅棕色波浪长发，自然妆容，匀称身材'],
    ['man', '男性模特', 'an adult East Asian man in his late 20s, short black hair, clean-shaven, tall and lean', '一位二十多岁的成年东亚男性，黑色短发，干净面容，高瘦身材'],
    ['anime', '动漫角色', 'an adult anime-style girl, long silver hair, large expressive eyes, adult proportions, clean cel-shaded illustration', '一位成年动漫风格女性角色，银色长发，大眼睛，成人身材比例，干净的赛璐璐上色插画'],
    ['custom', '自定义', '', ''],
  ];

  const STYLES = [
    ['keep', '照片质感', 'photorealistic, matching the original photo quality', '写实摄影质感，与原照片画质一致'],
    ['studio', '影棚大片', 'clean high-end fashion studio, soft key light, seamless light-grey backdrop, editorial lookbook photo', '高端时尚影棚，柔和主光，浅灰无缝背景，杂志画册质感'],
    ['street', '城市街拍', 'candid city street-style photo, natural daylight, shallow depth of field', '城市街拍抓拍，自然日光，浅景深'],
    ['ecom', '电商白底', 'e-commerce product-model photo, pure white background, even lighting, full body', '电商模特图，纯白背景，均匀布光，全身构图'],
    ['film', '胶片氛围', '35mm film photograph, warm tones, subtle grain, golden hour light', '35mm 胶片摄影，暖色调，轻微颗粒，黄金时刻光线'],
    ['anime', '动漫插画', 'high-quality anime illustration, clean line art, soft cel shading', '高质量动漫插画，干净线稿，柔和赛璐璐上色'],
  ];

  const ANG_EN = {
    正面: 'front view',
    斜侧: 'three-quarter view',
    侧面: 'side profile',
    俯拍: 'high-angle shot looking down',
    仰拍: 'low-angle shot looking up',
    背面: 'back view',
    下から: 'low angle',
  };

  const preferredMode = LS('wbMode');
  let autoMode = !preferredMode;

  const W = {
    mode: preferredMode || 'gen',
    persons: [],
    garments: [],
    poses: [],
    exprs: [],
    extraRefs: [],
    char: LS('wbChar') || 'asia',
    charTxt: LS('wbCharTxt') || '',
    style: LS('wbStyle') || 'keep',
    bm: LS('wbBm') || 'each',
    pl: LS('wbPl') || 'en',
    keepBg: LS('wbKeepBg') ?? true,
    busy: null,
    dirty: false,
    results: [],
    ri: 0,
    cmp: true,
    view: 'board',
  };

  const g0 = LS('wbSel') || {};
  if (desktop) {
    $('#wbTwoStage').checked = !!LS('wbTwoStage');
    $('#wbTwoStage').onchange = event => SV('wbTwoStage', event.target.checked);
  }
  const savedDraft = LS('wbDraft');
  if (savedDraft) {
    $('#wbExtra').value = savedDraft.extra;
    $('#wbN').value = savedDraft.count;
    $('#wbAR').value = savedDraft.ratio;
    W.dirty = savedDraft.prompt != null;
    if (W.dirty) $('#wbPrompt').value = savedDraft.prompt;
  }

  function saveDraft() {
    SV('wbDraft', {
      prompt: W.dirty ? $('#wbPrompt').value : null,
      extra: $('#wbExtra').value,
      count: $('#wbN').value,
      ratio: $('#wbAR').value,
    });
  }

  // ---------- Helpers ----------
  const modelOf = (settings = cfg) =>
    settings.prov === 'local' && localPreset
      ? settings.checkpoint
      : (settings.model || '').trim() || PROV[settings.prov].model;

  const provLabel = () => desktop ? desktop.model :
    configured() ? [PROV[cfg.prov].name, cfg.prov !== 'local' && modelOf()].filter(Boolean).join(' · ') : '';

  function configError(settings) {
    if (settings.prov === 'local') return localPreset && !settings.checkpoint ? '请先测试连接并选择本机模型' : '';
    if (settings.prov === 'custom' && !settings.base) return '自定义中转需要填写 Base URL';
    if (!settings.base) return '';
    const allowHttp = settings.prov === 'custom';
    try {
      const u = new URL(settings.base);
      if ((u.protocol === 'https:' || (allowHttp && u.protocol === 'http:')) && !u.username && !u.password) return '';
    } catch {}
    return allowHttp
      ? 'Base URL 需填写有效的 HTTP/HTTPS 地址，请勿在地址中包含账号或密码。'
      : 'Base URL 需填写有效的 HTTPS 地址，请勿在地址中包含账号或密码。';
  }

  function loadImg(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(Error('图片读取失败'));
      img.src = src;
    });
  }

  async function toJpeg(src, max = 1536) {
    const img = await loadImg(src);
    let w = img.naturalWidth;
    let h = img.naturalHeight;
    const scale = Math.min(1, max / Math.max(w, h));
    w = Math.round(w * scale);
    h = Math.round(h * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);
    return new Promise(resolve => canvas.toBlob(blob => resolve({ blob, w, h }), 'image/jpeg', 0.92));
  }

  const libSrc = ref => {
    const it = A.ITEMS[ref.k];
    return it.angles ? it.angles[ref.ai ?? it.ai].i : it.i;
  };

  async function assetBlob(ref, signal) {
    if (ref.blob) return ref.blob;
    if (desktop && ref.tcMediaID) return desktop.asBlob(desktop.imageURL(ref.tcMediaID), signal);
    const src = ref.kind === 'portrait' ? A.ITEMS[ref.k].m : libSrc(ref);
    const resp = await fetch(src, { signal, referrerPolicy: A.ITEMS[ref.k].ref || '' });
    if (!resp.ok) throw Error(`素材下载失败（${resp.status}）`);
    const url = URL.createObjectURL(await resp.blob());
    try {
      return (await toJpeg(url, ref.kind === 'portrait' ? 1536 : 1280)).blob;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  const isGen = () => W.mode === 'gen';
  const hasPerson = () => W.persons.length > 0;
  const allSelectedCount = () => W.persons.length + W.garments.length + W.poses.length + W.exprs.length + W.extraRefs.length;

  // ---------- Prompt Composition ----------
  // Line art and depth maps represent the same pose only when sharing the same camera angle.
  function poseGroups(poses) {
    const groups = [];
    for (const ref of poses) {
      const item = A.ITEMS[ref.k];
      const mate = A.poseCounterpart(item, ref.ai ?? item.ai);
      const partner = mate && groups.find(group => {
        const first = group[0];
        return group.length === 1 && first.k === mate.item.k && (first.ai ?? mate.item.ai) === mate.angle;
      });
      if (partner) partner.push(ref);
      else groups.push([ref]);
    }
    return groups;
  }

  function poseDesc(ref, en) {
    const it = A.ITEMS[ref.k];
    if (!it) return '';
    const angle = it.angles[ref.ai ?? it.ai];
    const isTwoPerson = /2人|双人/.test(it.line);
    if (en) {
      return `${it.en} (${it.lineEn}, ${ANG_EN[angle.zh] || angle.en || angle.zh})` +
        (isTwoPerson ? ' — a two-person pose: match the number of people and their relative positions' : '');
    }
    return `${it.zh}（${it.line}，${angle.zh}）` +
      (isTwoPerson ? '——双人姿势，人数与相对位置需与参考图一致' : '');
  }

  function poseInstructions(groups, poseNumbers, en, sheet) {
    const roleOf = ref => {
      const depth = A.ITEMS[ref.k].type === 'depth';
      const kind = en ? (depth ? 'depth map' : 'pose sketch') : (depth ? '深度图' : '姿势线稿');
      return en ? `${poseNumbers.get(ref)} (${kind})` : `${poseNumbers.get(ref)}（${kind}）`;
    };
    const details = groups.map((group, idx) => {
      const refs = group.map(roleOf).join(en ? ' + ' : '＋');
      const desc = poseDesc(group[0], en);
      const pairing = group.length === 2
        ? (en ? ' together show the same pose and camera angle: ' : '对应同一动作和机位：')
        : (en ? ': ' : '：');
      return (sheet ? `(${idx + 1}) ` : '') + refs + pairing + desc;
    });
    const prefix = en
      ? (sheet ? `Pose: draw ${groups.length} panels in one image, one per pose, with the same subject and outfit, even spacing and no text. ` : 'Pose: ')
      : (sheet ? `姿势：在同一张图中画 ${groups.length} 格，每格对应一个姿势，人物与服装保持一致，均匀留白，不写文字。` : '姿势：');
    const types = new Set(groups.flat().map(ref => A.ITEMS[ref.k].type));
    const guides = [];
    if (types.has('pose')) {
      guides.push(en ? 'Use the sketch for body posture, weight and limb placement. Omit its guide lines.' : '按线稿还原身体姿态、重心和四肢位置，省略辅助线。');
    }
    if (types.has('depth')) {
      guides.push(en ? 'Use the depth map for near/far relationships and occlusion. Render natural colors and lighting.' : '按深度图还原肢体前后关系和遮挡，成图使用自然色彩与光照。');
    }
    return prefix + details.join(en ? '; ' : '；') + (en ? '. ' : '。') + guides.join(en ? ' ' : '');
  }

  function compose(poseSet, state = W) {
    const en = state.pl === 'en';
    const P = state.persons.length;
    const G = state.garments.length;
    const ps = poseSet || state.poses;
    const groups = poseGroups(ps);
    const np = groups.length;
    const sheet = np > 1 && state.bm === 'sheet';
    const style = STYLES.find(s => s[0] === state.style) || STYLES[0];
    const ch = CHARS.find(c => c[0] === state.char) || CHARS[0];
    const extra = state.extra ?? $('#wbExtra').value.trim();
    const garmentItems = state.garments.map(g => g.kind === 'lib' && A.ITEMS[g.k]).filter(Boolean);

    let seq = 0;
    const rangeLabel = count => {
      if (!count) return null;
      const start = seq + 1;
      const end = seq + count;
      seq = end;
      if (start === end) return en ? `image ${start}` : `第 ${start} 张`;
      return en ? `images ${start}–${end}` : `第 ${start}–${end} 张`;
    };

    const E = state.exprs.length;
    const exprItem = E ? A.ITEMS[state.exprs[0].k] : null;
    const rP = rangeLabel(P);
    const rG = rangeLabel(G);
    const poseNumbers = new Map(ps.map(ref => [ref, rangeLabel(1)]));
    const rE = rangeLabel(E);

    const materials = state.extraRefs.map(ref => {
      const it = A.ITEMS[ref.k];
      const role = MATERIAL_ROLES[it.type][en ? 1 : 0];
      const desc = A.prompt({ ...it, ai: ref.ai ?? it.ai }, state.pl);
      return en ? `${role} (${rangeLabel(1)}): ${desc}` : `${role}（${rangeLabel(1)}）：${desc}`;
    });

    if (!(P || G || np || E)) {
      if (state.mode === 'gen' && state.char === 'custom' && state.charTxt.trim()) materials.unshift(state.charTxt.trim());
      if (!materials.length) return extra;
      if (state.dirtyStyle) materials.push(style[en ? 2 : 3]);
      if (extra) materials.push(extra);
      return materials.join('\n');
    }

    const tryon = P && G && state.mode === 'tryon';
    const keepBackground = tryon && state.keepBg && !state.extraRefs.some(ref => ['scene', 'light'].includes(A.ITEMS[ref.k].type));
    const hasHairRef = state.extraRefs.some(ref => A.ITEMS[ref.k].type === 'hair');
    const lines = [];

    if (en) {
      lines.push(
        P
          ? `Fashion ${tryon ? 'virtual try-on' : 'image generation'} using reference images. ${rP} ${P > 1 ? 'show' : 'shows'} the person (same individual)${G ? `; ${rG} ${G > 1 ? 'show' : 'shows'} the outfit${G > 1 ? ' pieces (top and bottom)' : ''} on an invisible mannequin` : ''}.`
          : `Fashion image generation from reference images.${G ? ` ${rG} ${G > 1 ? 'show' : 'shows'} the outfit on an invisible mannequin.` : ''}`
      );
      lines.push(
        P
          ? `Subject: the person from the photo. Preserve their face, identity, ${hasHairRef ? '' : 'hairstyle, '}skin tone and body proportions exactly; they must stay recognisably the same adult.`
          : `Subject: ${(state.char === 'custom' ? state.charTxt : ch[2]) || ch[2] || 'an adult model'}. Always an adult with adult proportions.`
      );
      if (G) {
        const garmentText = garmentItems.length
          ? ': ' + garmentItems.map(x => x.en.replace(/^[^,]*outfit,\s*/, '')).join('; ')
          : '';
        lines.push(`Outfit: dress the subject in the complete outfit from ${rG}, reproducing every garment faithfully — cut, color, fabric, pattern, layering, shoes and accessories${garmentText}. Replace the original clothing entirely. Never draw the mannequin or a blank body.`);
      }
      if (np) lines.push(poseInstructions(groups, poseNumbers, true, sheet));
      else if (P) lines.push('Pose: keep the original pose, framing and camera angle.');
      if (E && exprItem) {
        lines.push(`Expression: ${rE} shows the target expression (${exprItem.en}). Transfer the expression only.${P ? ' Preserve the facial features and identity from the person photo.' : ''}`);
      }
      lines.push(...materials);
      lines.push(`Look: ${keepBackground ? 'keep the original background and lighting of the photo; ' : ''}${style[2]}.${G ? ' Natural fabric drape with realistic folds and shadows.' : ''} High detail. No text, no watermark, no borders.`);
      if (extra) lines.push('Additional: ' + extra);
    } else {
      lines.push(
        P
          ? `时尚${tryon ? '虚拟试穿' : '生图'}，使用参考图。${rP}是人物照片（同一人）${G ? `；${rG}是放在隐形人台上的服装${G > 1 ? '（上装与下装）' : ''}` : ''}。`
          : `根据参考图进行时尚生图。${G ? `${rG}是放在隐形人台上的服装。` : ''}`
      );
      lines.push(
        P
          ? `人物：照片中的人。严格保留其面部、身份特征、${hasHairRef ? '' : '发型、'}肤色与身材比例，必须一眼认出是同一位成年人。`
          : `人物：${(state.char === 'custom' ? state.charTxt : ch[3]) || ch[3] || '一位成年模特'}。始终为成年人比例。`
      );
      if (G) {
        const garmentText = garmentItems.length
          ? '：' + garmentItems.map(x => x.zh.replace(/[。.]$/, '')).join('；')
          : '';
        lines.push(`服装：为人物换上${rG}中的整套服装，准确还原版型、颜色、面料、图案、叠穿、鞋袜与配饰${garmentText}。完全替换原有衣物，不要画出人台或空白身体。`);
      }
      if (np) lines.push(poseInstructions(groups, poseNumbers, false, sheet));
      else if (P) lines.push('姿势：保持原照片的姿势、构图和拍摄角度。');
      if (E && exprItem) {
        lines.push(`表情：${rE} 展示目标表情（${exprItem.zh}）。只迁移神态。${P ? '保留人物照片中的五官、脸型和身份。' : ''}`);
      }
      lines.push(...materials);
      lines.push(`画面：${keepBackground ? '保留原照片的背景与光线；' : ''}${style[3]}。${G ? '衣物垂坠自然，褶皱阴影真实。' : ''}高细节。不要文字、水印或边框。`);
      if (extra) lines.push('补充：' + extra);
    }
    return lines.join('\n');
  }

  function reprompt(force) {
    if (force || !W.dirty) {
      $('#wbPrompt').value = compose();
      W.dirty = false;
    }
  }

  // ---------- Board & Slot Rendering ----------
  const typeButton = (type, attr) => `<button type="button" ${attr}="${type}">${boardLabels[type]}</button>`;
  $('#wbOtherTypes').innerHTML = otherTypes.map(type => typeButton(type, 'data-pick')).join('');

  const thumbnail = (ref, it) => ref.url || (it.angles ? A.ath(it.angles[ref.ai ?? it.ai].t) : A.thumb(it));

  function pickRow(ref, slot, index) {
    const it = A.ITEMS[ref.k];
    const type = it?.type || slot;
    const name = it ? A.title(it) : ref.name || '自定义图片';
    const angle = it?.angles?.[ref.ai ?? it.ai];
    const sub = it ? `<span>${boardLabels[type]}</span> · ${esc(it.no || it.id)}` : '本地上传';
    const views = angle
      ? `<select data-ang="${index}" data-slot="${slot}" aria-label="角度">${it.angles.map((a, i) => `<option value="${i}" ${a === angle ? 'selected' : ''}>${esc(a.zh)}</option>`).join('')}</select>`
      : '';
    return `<div class="pick ${slot}" data-type="${type}"><img src="${esc(thumbnail(ref, it))}" alt=""${it?.ref ? ` referrerpolicy="${esc(it.ref)}"` : ''}><div><b>${esc(name)}</b><small>${sub}</small></div>${views}<button type="button" class="rmx" data-rm="${slot}" data-i="${index}" aria-label="移除" title="移除">×</button></div>`;
  }

  function renderOther() {
    const refs = [...W.exprs, ...W.extraRefs];
    $('#wbOtherPreview').innerHTML = refs
      .slice(0, 3)
      .map(ref => {
        const it = A.ITEMS[ref.k];
        return `<img data-type="${it.type}" src="${esc(thumbnail(ref, it))}" alt="" title="${esc(A.title(it))}"${it.ref ? ` referrerpolicy="${esc(it.ref)}"` : ''}>`;
      })
      .join('');
    const count = $('#wbOtherCount');
    count.hidden = !refs.length;
    count.textContent = refs.length || '';
    $('#wbEx').innerHTML = W.exprs.map((ref, i) => pickRow(ref, 'expr', i)).join('');
    $('#wbOtherRefs').innerHTML = W.extraRefs.map((ref, i) => pickRow(ref, 'other', i)).join('');
  }

  let exampleRefs = [];

  function syncCanvasView() {
    const hasOut = W.results.length > 0;
    const sw = $('#wbCanvasSwitch');
    if (sw) {
      sw.hidden = !hasOut;
      sw.querySelectorAll('[data-cv]').forEach(b => {
        const on = b.dataset.cv === W.view;
        b.classList.toggle('on', on);
        b.setAttribute('aria-pressed', String(on));
      });
    }
    const showResult = hasOut && W.view === 'result';
    $('#wbPh0').hidden = showResult;
    $('#wbCmp').hidden = !showResult;
    $('#wbRes').hidden = !showResult || W.results.length < 2;
    $('#wbOutAct').hidden = !showResult;
    const sent = $('#wbSentD');
    if (sent) sent.hidden = !showResult || !W.results[W.ri]?.prompt;
    const exportBtn = $('#wbExportBoard');
    if (exportBtn) exportBtn.hidden = showResult || !allSelectedCount();
  }

  function chosenBoardRefs() {
    return [
      ...W.persons.map(ref => ['portrait', ref]),
      ...W.garments.map(ref => ['costume', ref]),
      ...W.poses.map(ref => ['pose', ref]),
      ...W.exprs.map(ref => ['expr', ref]),
      ...W.extraRefs.map(ref => [A.ITEMS[ref.k]?.type || 'other', ref]),
    ];
  }

  function paintBoard() {
    const hasOut = W.results.length > 0;
    const output = hasOut && W.view === 'result';
    const chosen = chosenBoardRefs();
    const example = !chosen.length;
    syncCanvasView();
    $('#wbCanvasTitle').textContent = output ? '生成结果' : example && exampleRefs.length ? '搭配示例' : '参考画板';
    const ratio = $('#wbAR').value;
    const result = W.results[W.ri];
    const raster = result?.width && result.height ? `${result.width}×${result.height}` : '';
    const referenceSize = result?.referencePixels ? `参考图 ${result.referencePixels.toLocaleString()} 像素` : '';
    $('#wbCanvasRatio').textContent = output ? [raster, referenceSize].filter(Boolean).join(' · ') : (ratio === 'auto' ? '' : ratio);
    if (output) return;

    const refs = example ? exampleRefs.map(it => [it.type, libItem(it)]) : chosen;
    const cards = $('#wbBoardCards');
    cards.classList.toggle('sample', example);
    cards.classList.toggle('many', refs.length > 3);
    cards.innerHTML = refs
      .map(([type, ref], idx) => {
        const item = A.ITEMS[ref.k];
        const source = thumbnail(ref, item);
        const material = item?.type || type;
        const label = item ? A.title(item) : ref.name || boardLabels[type];
        const angle = !example && item?.angles?.[ref.ai ?? item.ai]?.zh;
        const idxBadge = example ? '' : `<i class="board-idx" aria-hidden="true">${idx + 1}</i>`;
        const angBadge = angle ? `<em class="board-ang">${esc(angle)}</em>` : '';
        return `<button class="board-card" data-pick="${material}" data-type="${material}"${item ? ` data-pk-key="${esc(item.k)}"` : ''} title="${esc(label)}">${idxBadge}<img src="${esc(source)}" alt="${esc(label)}"${item?.ref ? ` referrerpolicy="${esc(item.ref)}"` : ''}>${angBadge}<span>${boardLabels[material]}</span></button>`;
      })
      .join('');

    const showActions = example && refs.length > 0;
    if ($('#wbBoardActions')) $('#wbBoardActions').hidden = !showActions;
    $('#wbExample').hidden = !showActions;
    if ($('#wbShuffle')) $('#wbShuffle').hidden = !showActions;
  }

  async function exportBoardPng(opts = {}) {
    const chosen = chosenBoardRefs();
    if (!chosen.length || !window.PoseWearBoardExport) return null;
    const btn = $('#wbExportBoard');
    if (btn) btn.disabled = true;
    try {
      const { blob, extSkipped, now } = await window.PoseWearBoardExport.renderPng(chosen, {
        app: A,
        aspectRatio: $('#wbAR').value,
        promptText: $('#wbPrompt')?.value || '',
      });
      if (!blob) throw Error('PNG 导出失败');
      if (opts.download !== false && window.X?.dl) {
        window.X.dl(blob, `posewear-board-${stamp(now)}.png`);
        toast(extSkipped ? '已导出参考拼图 PNG（外链图已标出）' : '已导出参考拼图 PNG');
      }
      return blob;
    } catch {
      toast('导出参考拼图失败，请重试');
      return null;
    } finally {
      if (btn) btn.disabled = false;
    }
  }
  $('#wbExportBoard')?.addEventListener('click', () => exportBoardPng());

  async function loadExample(randomize = false) {
    if (!randomize && exampleRefs.length) return;
    const types = ['portrait', 'costume', 'pose'];
    const groups = await Promise.allSettled(types.map(A.load));
    const prevKeys = new Set(exampleRefs.map(it => it.k));
    exampleRefs = groups.flatMap(result => {
      if (result.status !== 'fulfilled') return [];
      const pool = result.value.filter(it => !it.a && (!desktop || it.demo === true));
      if (!pool.length) return [];
      if (!randomize) return [pool[0]];
      const fresh = pool.filter(it => !prevKeys.has(it.k));
      const pickFrom = fresh.length ? fresh : pool;
      return [pickFrom[(Math.random() * pickFrom.length) | 0]];
    });
    paintBoard();
    if (randomize && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
      $$('#wbBoardCards .board-card').forEach((card, i) =>
        card.animate([{ opacity: 0, transform: 'translateY(14px) scale(.94)' }, { opacity: 1 }], {
          duration: 360,
          delay: i * 55,
          easing: 'cubic-bezier(.16,1,.3,1)',
        })
      );
    }
  }

  if (desktop) $('#wbShuffle').hidden = true;
  else $('#wbShuffle')?.addEventListener('click', () => loadExample(true));
  $('#wbCanvasSwitch')?.addEventListener('click', e => {
    const b = e.target.closest('[data-cv]');
    if (!b || !W.results.length) return;
    W.view = b.dataset.cv;
    paintBoard();
    if (W.view === 'board' && !allSelectedCount()) loadExample();
  });
  $('#wbClearAll')?.addEventListener('click', () => {
    if (!allSelectedCount()) return;
    const snap = {
      persons: W.persons.slice(),
      garments: W.garments.slice(),
      poses: W.poses.slice(),
      exprs: W.exprs.slice(),
      extraRefs: W.extraRefs.slice(),
      dirty: W.dirty,
      prompt: $('#wbPrompt').value,
      view: W.view,
    };
    W.persons = [];
    W.garments = [];
    W.poses = [];
    W.exprs = [];
    W.extraRefs = [];
    W.dirty = false;
    W.view = 'board';
    saveDraft();
    render();
    loadExample();
    toast('已清空素材', () => {
      W.persons = snap.persons;
      W.garments = snap.garments;
      W.poses = snap.poses;
      W.exprs = snap.exprs;
      W.extraRefs = snap.extraRefs;
      W.dirty = snap.dirty;
      W.view = snap.view;
      render();
      if (snap.dirty) $('#wbPrompt').value = snap.prompt;
      saveDraft();
    });
  });
  $('#wbExample').onclick = async () => {
    const button = $('#wbExample');
    button.disabled = true;
    try {
      await loadExample();
      await setSel({
        portrait: exampleRefs.filter(it => it.type === 'portrait'),
        garment: exampleRefs.filter(it => it.type === 'costume'),
        pose: exampleRefs.filter(it => it.type === 'pose'),
      });
    } catch {
      toast('素材未能载入，请重试');
    } finally {
      button.disabled = false;
    }
  };

  function updateStart() {
    const ready =
      allSelectedCount() ||
      $('#wbExtra').value.trim() ||
      (W.dirty && $('#wbPrompt').value.trim()) ||
      (isGen() && W.char === 'custom' && W.charTxt.trim());
    const need = ready ? [] : ['选择素材或输入提示词'];
    const button = $('#wbGo');
    if (!W.busy) {
      button.disabled = !!need.length;
      button.title = need.length ? '请先' + need.join('、') : '';
    }
    return need;
  }

  function setMode(mode) {
    autoMode = false;
    W.mode = mode;
    SV('wbMode', mode);
    if (mode === 'gen' && W.style === 'keep') W.style = 'studio';
    if (mode === 'tryon' && W.style === 'studio' && !W.dirtyStyle) W.style = 'keep';
  }

  const nJobs = () => (W.bm === 'each' ? Math.max(1, poseGroups(W.poses).length) : 1) * (+$('#wbN').value || 1);

  function render() {
    if (autoMode) W.mode = hasPerson() && W.garments.length ? 'tryon' : 'gen';
    const G = isGen();
    $$('[data-mode]').forEach(b => {
      b.classList.toggle('on', b.dataset.mode === W.mode);
      b.setAttribute('aria-pressed', String(b.dataset.mode === W.mode));
    });
    $('#wbPersons').innerHTML = W.persons
      .map((p, i) => `<div class="pp"><img src="${esc(p.url)}" alt="人物照片 ${i + 1}">${p.w && p.h ? `<small>${p.w}×${p.h}</small>` : ''}<button class="x" data-rmp="${i}" aria-label="移除">×</button></div>`)
      .join('');
    $('#wbPortraitPick').hidden = !!W.persons.length;
    $('#wbDrop').hidden = W.persons.length >= 3;
    $('#wbChar').hidden = !G || hasPerson();
    $('#wbChars').innerHTML = CHARS.map(c => `<button class="chip ${W.char === c[0] ? 'on' : ''}" data-char="${c[0]}">${c[1]}</button>`).join('');
    $('#wbCharTxt').hidden = W.char !== 'custom';
    $('#wbCharTxt').value = W.charTxt;
    $('#wbG').innerHTML = W.garments.length
      ? W.garments.map((g, i) => pickRow(g, 'costume', i)).join('')
      : '<button class="empty0" data-pick="costume"><span><b>选择服装</b></span></button>';
    $('#wbPo').innerHTML = W.poses.length
      ? W.poses.map((p, i) => pickRow(p, 'pose', i)).join('')
      : '<button class="empty0" data-pick="pose"><span><b>选择姿势</b></span></button>';
    renderOther();

    if ($('#wbClearAll')) $('#wbClearAll').hidden = !allSelectedCount();
    $('#wbBatch').hidden = poseGroups(W.poses).length < 2;
    $$('[data-bm]').forEach(b => b.classList.toggle('on', b.dataset.bm === W.bm));
    $('#wbAllAng').hidden = W.poses.length !== 1 || !A.ITEMS[(W.poses[0] || {}).k];
    $('#wbStyles').innerHTML = STYLES.map(s => `<button class="chip ${W.style === s[0] ? 'on' : ''}" data-style="${s[0]}">${s[1]}</button>`).join('');
    $('#wbKeepBg').checked = W.keepBg;
    $('#wbKeepBg').closest('label').hidden = G || !hasPerson();
    $$('[data-pl]').forEach(b => b.classList.toggle('on', b.dataset.pl === W.pl));

    const ok = configured();
    $('#wbDot').className = 'dot' + (ok ? ' on' : '');
    $('#wbProv').textContent = provLabel() || '未配置 API';
    $('#wbDot2').className = 'dot' + (ok ? ' on' : '');
    $('#wbProv2').textContent = desktop ? desktop.model : ok ? PROV[cfg.prov].name : '未配置 API';

    const sizes = [W.persons.length, W.garments.length, W.poses.length];
    for (const step of $$('.primary-materials .step')) {
      const size = sizes[Number(step.dataset.s) - 1];
      step.classList.toggle('done', size > 0);
      step.querySelector('.pick-more').hidden = !size || (step.dataset.s === '1' && size >= 3);
    }

    $('#wbTwoStage').closest('label').hidden = !desktop || !W.persons.length || !W.garments.length || !W.poses.length;
    const jobs = nJobs();
    if (!W.busy) $('#wbGo').textContent = '生成';
    $('#wbEst').textContent = `将生成 ${jobs} 张`;
    updateStart();
    paintBoard();
    SV('wbSel', {
      r: W.persons.filter(p => p.k).map(p => p.k),
      g: W.garments.filter(g => g.kind === 'lib').map(g => (g.ai == null ? g.k : [g.k, g.ai])),
      p: W.poses.map(p => [p.k, p.ai]),
      e: W.exprs.map(e => [e.k, e.ai]),
      x: W.extraRefs.map(x => [x.k, x.ai]),
    });
    if (desktop) {
      const snapshot = group => group.filter(ref => ref.tcMediaID && !ref.k)
        .map(ref => ({ tcMediaID: ref.tcMediaID, w: ref.w, h: ref.h, name: ref.name }));
      SV('wbLocalDraft', { persons: snapshot(W.persons), garments: snapshot(W.garments) });
    }
    reprompt();
  }

  // ---------- Uploads & Slot Events ----------
  const libPortrait = ref => ({ kind: 'portrait', k: ref.k, url: A.ITEMS[ref.k].m });

  function addPortrait(it) {
    if (W.persons.length >= 3) return toast('最多 3 张人物照片');
    W.persons.push(libPortrait(it));
    render();
    toast(W.mode === 'gen' ? '已设为脸部参考（生成时保持这张脸）' : '已设为人物照片');
  }

  async function addPersons(fileList) {
    const files = [...fileList].filter(f => /^image\//.test(f.type));
    if (!files.length) return toast('请选择图片文件');
    for (const f of files) {
      if (W.persons.length >= 3) {
        toast('最多 3 张人物照片');
        break;
      }
      try {
        if (desktop) {
          const media = await desktop.importImage(f, f.name);
          W.persons.push({ tcMediaID: media.id, name: media.name, url: media.url, w: media.width, h: media.height });
        } else {
          const url = URL.createObjectURL(f);
          try {
            const res = await toJpeg(url, 1536);
            W.persons.push({ blob: res.blob, url: URL.createObjectURL(res.blob), w: res.w, h: res.h });
          } finally { URL.revokeObjectURL(url); }
        }
      } catch (error) {
        toast(desktop ? error.message : '无法读取这张图片');
      }
    }
    render();
  }

  async function addGarmentFiles(fileList) {
    const files = [...fileList].filter(f => /^image\//.test(f.type));
    if (!files.length) return toast('请选择图片文件');
    for (const file of files) {
      if (W.garments.length >= 2) {
        toast('最多 2 张服装参考图');
        break;
      }
      try {
        if (desktop) {
          const media = await desktop.importImage(file, file.name);
          W.garments.push({ kind: 'file', name: media.name, tcMediaID: media.id, url: media.url });
        } else {
          const url = URL.createObjectURL(file);
          try {
            const result = await toJpeg(url, 1280);
            W.garments.push(fileItem(result.blob, file.name));
          } finally { URL.revokeObjectURL(url); }
        }
      } catch (error) {
        toast(desktop ? error.message : '无法读取服装图');
      }
    }
    render();
  }

  async function addLocalMaterials(fileList) {
    if (!desktop) return;
    const type = $('#wbLocalRole').value;
    const slot = C.studioSlot(type);
    const limit = { portrait: 3, garment: 2, pose: 6, expr: 1, extra: 10 }[slot];
    for (const file of [...fileList]) {
      try {
        const media = await desktop.importImage(file, file.name);
        const k = A.addLocalItem(type, media);
        const group = { portrait: W.persons, garment: W.garments, pose: W.poses, expr: W.exprs, extra: W.extraRefs }[slot];
        if (group.length < limit && allSelectedCount() < 10) {
          group.push(slot === 'portrait' ? libPortrait({ k }) : libItem({ k }));
          render();
        } else {
          toast('已加入本机图库。请移除当前工作台参考图后选择。');
        }
      } catch (error) { toast(error.message); }
    }
  }

  $('#wbLocalFile')?.addEventListener('change', event => {
    void addLocalMaterials([...event.target.files]);
    event.target.value = '';
  });

  const bindDrop = (el, fn) => {
    if (!el) return;
    ['dragenter', 'dragover'].forEach(evt => el.addEventListener(evt, e => { e.preventDefault(); el.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(evt => el.addEventListener(evt, e => { e.preventDefault(); el.classList.remove('over'); }));
    el.addEventListener('drop', e => fn(e.dataTransfer.files));
  };
  bindDrop($('.person-step'), addPersons);
  bindDrop($('.garment-step'), addGarmentFiles);

  $('#wbFile').onchange = e => { addPersons(e.target.files); e.target.value = ''; };
  $('#wbGFile').onchange = e => { addGarmentFiles(e.target.files); e.target.value = ''; };
  for (const label of $$('.upload-link')) {
    label.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        label.querySelector('input').click();
      }
    });
  }
  document.addEventListener('paste', e => {
    if (document.body.classList.contains('studio') && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
      const files = [...e.clipboardData.files].filter(f => /^image/.test(f.type));
      if (files.length) addPersons(files);
    }
  });

  $('#studio').addEventListener('click', e => {
    const t = e.target;
    let b;
    if ((b = t.closest('[data-rm]'))) {
      const refs = { pose: W.poses, expr: W.exprs, costume: W.garments, other: W.extraRefs }[b.dataset.rm];
      const index = Number(b.dataset.i);
      const group = b.closest('.step,.material-extras');
      const type = b.closest('.pick').dataset.type;
      const [ref] = refs.splice(index, 1);
      if (ref.blob) URL.revokeObjectURL(ref.url);
      render();
      const remaining = group.querySelectorAll(`[data-rm="${b.dataset.rm}"]`);
      const next = remaining[Math.min(index, remaining.length - 1)] ||
        group.querySelector(`[data-pick="${type}"]:not([hidden])`) ||
        group.querySelector('.empty0');
      next?.focus({ preventScroll: true });
    }
    if ((b = t.closest('[data-rmp]'))) {
      const i = Number(b.dataset.rmp);
      const [person] = W.persons.splice(i, 1);
      if (person.blob) URL.revokeObjectURL(person.url);
      render();
      ($(`#wbPersons [data-rmp="${Math.min(i, W.persons.length - 1)}"]`) || $('#wbPortraitPick')).focus({ preventScroll: true });
    }
    if ((b = t.closest('[data-pick]'))) picker(b.dataset.pick, b);
    if ((b = t.closest('[data-pl]'))) { W.pl = b.dataset.pl; SV('wbPl', W.pl); reprompt(true); saveDraft(); render(); }
    if ((b = t.closest('[data-mode]'))) { setMode(b.dataset.mode); render(); }
    if ((b = t.closest('[data-char]'))) { W.char = b.dataset.char; SV('wbChar', W.char); render(); }
    if ((b = t.closest('[data-style]'))) { W.style = b.dataset.style; W.dirtyStyle = 1; SV('wbStyle', W.style); render(); }
    if ((b = t.closest('[data-bm]'))) { W.bm = b.dataset.bm; SV('wbBm', W.bm); render(); }
  });

  $('#studio').addEventListener('change', e => {
    const sel = e.target.closest('[data-ang]');
    if (sel) {
      const slot = sel.dataset.slot;
      const index = sel.dataset.ang;
      const refs = { pose: W.poses, expr: W.exprs, costume: W.garments, other: W.extraRefs }[slot];
      refs[Number(index)].ai = Number(sel.value);
      render();
      $(`#studio [data-slot="${slot}"][data-ang="${index}"]`).focus({ preventScroll: true });
    } else if (e.target.id === 'wbN' || e.target.id === 'wbAR') {
      saveDraft();
      render();
    } else if (e.target.id === 'wbKeepBg') {
      W.keepBg = e.target.checked;
      SV('wbKeepBg', W.keepBg);
      render();
    }
  });

  $('#wbCharTxt').oninput = e => { W.charTxt = e.target.value; SV('wbCharTxt', W.charTxt); reprompt(); updateStart(); };
  $('#wbAllAng').onclick = () => {
    const p = W.poses[0];
    const it = A.ITEMS[p.k];
    W.poses = it.angles.slice(0, 6).map((_, i) => ({ kind: 'lib', k: p.k, ai: i }));
    toast(`已展开 ${W.poses.length} 个角度`);
    render();
  };
  $('#wbPrompt').oninput = () => { W.dirty = true; saveDraft(); updateStart(); };
  $('#wbExtra').oninput = () => { reprompt(); saveDraft(); updateStart(); };
  $('#wbReP').onclick = () => { reprompt(true); saveDraft(); updateStart(); toast('已按当前选择重新生成提示词'); };

  function copyStudioBrief(text, inputs, lang) {
    if (!$('#copyRefs').checked) return A.copy(text, '提示词');
    const selected = [
      ...(inputs.persons || []),
      ...(inputs.garments || []),
      ...(inputs.poses || []),
      ...(inputs.exprs || []),
      ...(inputs.extraRefs || []),
    ];
    const known = selected.filter(r => r.k);
    const copyCurrent = () => {
      const items = selected.flatMap(r => (r.k ? (A.ITEMS[r.k] ? [{ ...A.ITEMS[r.k], ai: r.ai ?? 0 }] : []) : [null]));
      return A.copyBrief(text, items, lang, '提示词');
    };
    if (known.every(r => A.ITEMS[r.k])) return copyCurrent();
    return A.needKeys(known.map(r => r.k)).then(copyCurrent).catch(() => toast('素材未能载入，请重试'));
  }
  $('#wbCopyP').onclick = () => copyStudioBrief($('#wbPrompt').value, W, W.pl);

  // ---------- Material Picker Modal ----------
  let pkType = 'costume';
  let pkSink = null;
  let pkItems = null;
  let pkLastId = '';
  let pkTimer;
  let pkFav = false;
  let pkCat = '';
  let pkMatches = [];
  let pkRendered = 0;

  const poseType = type => type === 'pose' || type === 'depth';
  const pkItemCat = it => it.cat || it.sec || it.era || it.emo || '';

  function pickerRefs() {
    if (pkSink) return pkSink.keys().map(k => A.ITEMS[k]).filter(A.ok);
    if (poseType(pkType)) return W.poses;
    if (garmentTypes.includes(pkType)) return W.garments;
    if (pkType === 'portrait') return W.persons;
    return pkType === 'expr' ? W.exprs : W.extraRefs;
  }

  function picker(type, opener = document.activeElement) {
    const initialKey = opener?.dataset.pkKey;
    pkLastId = A.ITEMS[initialKey]?.id || '';
    $('#pkQ').value = '';
    pkFav = false;
    pkCat = '';
    openDialog($('#pk'), $('#pkQ'), opener);
    showPickerType(type, null, initialKey);
  }

  function pickerPoseTitle() {
    if (pkSink) return '选择姿势';
    return W.poses.length ? `姿势参考图（${W.poses.length}/6 张）` : '选择姿势参考图（最多 6 张）';
  }

  function pickerAnchor() {
    const grid = $('#pkG');
    const view = grid.getBoundingClientRect();
    const visible = [];
    for (const card of grid.querySelectorAll('button[data-k]')) {
      const rect = card.getBoundingClientRect();
      if (rect.top >= view.bottom) break;
      if (rect.bottom <= view.top) continue;
      const anchor = { item: A.ITEMS[card.dataset.k], angle: +card.dataset.ai, offset: rect.top - view.top };
      if (anchor.item.id === pkLastId) visible.unshift(anchor);
      else visible.push(anchor);
    }
    return { visible, scroll: grid.scrollTop };
  }

  function drawPkFilters() {
    const bar = $('#pkBar');
    const favBtn = $('#pkFavOnly');
    const catsEl = $('#pkCats');
    if (!bar || !pkItems) return;
    const visible = pkItems.filter(A.ok);
    const favCount = visible.filter(it => A.st.fav.has(it.k)).length;
    if (!favCount) pkFav = false;
    favBtn.hidden = !favCount;
    favBtn.classList.toggle('on', pkFav);
    favBtn.setAttribute('aria-pressed', String(pkFav));

    const counts = new Map();
    for (const it of visible) {
      if (pkFav && !A.st.fav.has(it.k)) continue;
      const cat = pkItemCat(it);
      if (cat) counts.set(cat, (counts.get(cat) || 0) + 1);
    }
    if (pkCat && !counts.has(pkCat)) pkCat = '';
    const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    if (entries.length >= 2) {
      catsEl.innerHTML =
        `<button type="button" class="chip ${!pkCat ? 'on' : ''}" data-pk-cat="" aria-pressed="${!pkCat}">全部</button>` +
        entries.map(([cat, n]) => `<button type="button" class="chip ${pkCat === cat ? 'on' : ''}" data-pk-cat="${esc(cat)}" aria-pressed="${pkCat === cat}">${esc(cat)}<i>${n}</i></button>`).join('');
      catsEl.hidden = false;
    } else {
      catsEl.innerHTML = '';
      catsEl.hidden = true;
    }
    bar.hidden = favBtn.hidden && catsEl.hidden;
  }

  async function showPickerType(type, anchor, initialKey) {
    clearTimeout(pkTimer);
    const prevPose = poseType(pkType) && poseType(type);
    pkType = type;
    pkItems = null;
    if (!prevPose) { pkFav = false; pkCat = ''; }

    const poses = poseType(type);
    $('#pkPoseTypes').hidden = !poses;
    for (const tab of $$('#pkPoseTypes button')) {
      const on = tab.dataset.pkType === type;
      tab.classList.toggle('on', on);
      tab.setAttribute('aria-pressed', on);
    }

    const group = garmentTypes.includes(type) ? garmentTypes : otherTypes.includes(type) ? otherTypes : null;
    const tabs = $('#pkMaterialTypes');
    tabs.hidden = !group;
    if (group) {
      if (tabs.dataset.group !== group[0]) {
        tabs.innerHTML = group.map(kind => typeButton(kind, 'data-pk-material')).join('');
        tabs.dataset.group = group[0];
      }
      for (const tab of tabs.children) {
        const active = tab.dataset.pkMaterial === type;
        tab.classList.toggle('on', active);
        tab.setAttribute('aria-pressed', String(active));
      }
    }

    if ($('#pkBar')) $('#pkBar').hidden = true;
    $('#pkDone').hidden = false;
    $('#pkT').textContent = poses ? pickerPoseTitle() : group === otherTypes ? '其他素材' : group === garmentTypes ? '选择服装' : '选择人物';
    const grid = $('#pkG');
    if (!A.DATA[type]) grid.innerHTML = '<div class="skel"></div>'.repeat(12);

    try {
      const items = await A.load(type);
      if (pkType !== type || $('#pk').hidden) return;
      pkItems = items.filter(it => !it.previewOnly);
      if (type === 'depth' && A.DATA.pose) {
        const end = A.DATA.pose.length;
        const rank = it => (A.counterpartType(it) ? (A.ITEMS['p:' + it.id]?.ord ?? end) : end);
        pkItems.sort((a, b) => rank(a) - rank(b));
      }
      drawPkFilters();
      drawPk(initialKey);
      if (initialKey) grid.querySelector(`[data-k="${CSS.escape(initialKey)}"]`)?.scrollIntoView({ block: 'nearest' });
      if (anchor) {
        for (const { item, angle, offset } of anchor.visible) {
          const mate = A.poseCounterpart(item, angle);
          const card = mate && grid.querySelector(`[data-k="${CSS.escape(mate.item.k)}"]`);
          if (!card) continue;
          grid.scrollTop += card.getBoundingClientRect().top - grid.getBoundingClientRect().top - offset;
          return;
        }
        grid.scrollTop = anchor.scroll;
      }
    } catch {
      if (pkType === type && !$('#pk').hidden) grid.textContent = '素材加载失败，请关闭后重试';
    }
  }

  function pickerPoseAngles(refs) {
    const angles = new Map();
    for (const ref of refs) {
      const item = A.ITEMS[ref.k];
      if (!A.ok(item)) continue;
      const index = ref.ai ?? item.ai;
      if (item.type === pkType) { angles.set(item.k, index); continue; }
      const mate = A.poseCounterpart(item, index);
      if (mate && !angles.has(mate.item.k)) angles.set(mate.item.k, mate.angle);
    }
    return angles;
  }

  function drawPk(initialKey) {
    if (!pkItems) return;
    const words = $('#pkQ').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    pkMatches = pkItems.filter(it => {
      if (!A.ok(it)) return false;
      if (pkFav && !A.st.fav.has(it.k)) return false;
      if (pkCat && pkItemCat(it) !== pkCat) return false;
      if (!words.length) return true;
      const text = [
        it.id, it.no, it.ti, it.zh, it.en, it.ja,
        it.line, it.lineEn, it.lineJa, it.cat, it.catEn, it.catJa,
        it.kw, it.sec, it.len, it.st,
        ...(it.angles || []).flatMap(a => [a.zh, a.en, a.ja]),
      ].join(' ').toLowerCase();
      return words.every(word => text.includes(word));
    });

    const grid = $('#pkG');
    grid.replaceChildren();
    pkRendered = 0;
    if (pkMatches.length) {
      const count = poseType(pkType)
        ? pkMatches.length
        : Math.max(150, initialKey ? 1 + pkMatches.findIndex(it => it.k === initialKey) : 0);
      appendPickerItems(count);
    } else {
      grid.innerHTML = '<p class="fine">没有匹配的结果</p>';
    }
    grid.scrollTop = 0;
  }

  function appendPickerItems(count) {
    const refs = pickerRefs();
    const selected = new Set(refs.map(ref => ref.k));
    const poses = poseType(pkType);
    const angles = poses ? pickerPoseAngles(refs) : null;
    const items = pkMatches.slice(pkRendered, pkRendered + count);
    const markup = items
      .map(it => {
        const on = selected.has(it.k);
        const label = esc(A.title(it));
        const angle = poses ? (angles.get(it.k) ?? it.ai) : null;
        const src = poses ? A.ath(it.angles[angle].t) : A.thumb(it);
        return `<button type="button" data-k="${esc(it.k)}"${poses ? ` data-ai="${angle}"` : ''} aria-pressed="${on}" class="${pkType} ${on ? 'on' : ''}" title="${label}"><img loading="lazy" src="${esc(src)}"${it.ref ? ` referrerpolicy="${esc(it.ref)}"` : ''} alt=""><span>${label}</span></button>`;
      })
      .join('');
    $('#pkG').insertAdjacentHTML('beforeend', markup);
    pkRendered += items.length;
  }

  $('#pkG').addEventListener('scroll', e => {
    const grid = e.currentTarget;
    if (pkItems && pkRendered < pkMatches.length && grid.scrollHeight - grid.scrollTop - grid.clientHeight < grid.clientHeight) {
      appendPickerItems(150);
    }
  }, { passive: true });
  $('#pkQ').oninput = () => {
    pkLastId = '';
    clearTimeout(pkTimer);
    pkTimer = setTimeout(drawPk, 150);
  };

  function pkClose() {
    clearTimeout(pkTimer);
    pkItems = null;
    const dialog = $('#pk');
    const opener = dialogOpeners.get(dialog);
    if (!pkSink) render();
    pkSink = null;
    closeDialog(dialog);
    if (opener?.dataset.pick && !opener.isConnected) {
      const type = opener.dataset.pick === 'depth' ? 'pose' : opener.dataset.pick;
      const key = opener.dataset.pkKey;
      const board = key && $('#wbBoardCards').querySelector(`[data-pk-key="${CSS.escape(key)}"]`);
      const slot = [...$$(`#studio .step [data-pick="${CSS.escape(type)}"]`)].find(el => el.getClientRects().length);
      (board || slot || $('#wbOther summary')).focus({ preventScroll: true });
    }
  }
  $('#pkX').onclick = pkClose;
  $('#pkDone').onclick = pkClose;
  $('#pk').onclick = e => {
    if (e.target.id === 'pk') return pkClose();
    if (e.target.closest('#pkFavOnly')) { pkFav = !pkFav; drawPkFilters(); drawPk(); return; }
    const catBtn = e.target.closest('[data-pk-cat]');
    if (catBtn) { pkCat = catBtn.dataset.pkCat; drawPkFilters(); drawPk(); return; }
    const tab = e.target.closest('[data-pk-type]');
    if (tab) { if (tab.dataset.pkType !== pkType) showPickerType(tab.dataset.pkType, pickerAnchor()); return; }
    const material = e.target.closest('[data-pk-material]');
    if (material) { if (material.dataset.pkMaterial !== pkType) showPickerType(material.dataset.pkMaterial); return; }

    const card = e.target.closest('#pkG [data-k]');
    if (!card) return;
    const key = card.dataset.k;
    const item = A.ITEMS[key];
    const angle = poseType(pkType) ? +card.dataset.ai : undefined;
    let on;
    if (pkSink) {
      on = pkSink.toggle(key, angle);
    } else if (poseType(pkType)) {
      on = !W.poses.some(ref => ref.k === key);
      if (!on) W.poses = W.poses.filter(ref => ref.k !== key);
      else if (W.poses.length >= 6) return toast('最多选择 6 张姿势参考图');
      else W.poses.push({ kind: 'lib', k: key, ai: angle });
      $('#pkT').textContent = pickerPoseTitle();
    } else if (pkType === 'portrait') {
      const index = W.persons.findIndex(ref => ref.k === key);
      on = index < 0;
      if (!on) W.persons.splice(index, 1);
      else if (W.persons.length < 3) W.persons.push(libPortrait(item));
      else return toast('最多 3 张人物照片');
    } else {
      const refs = pickerRefs();
      const index = refs.findIndex(ref => ref.k === key);
      on = index < 0;
      if (!on) {
        refs.splice(index, 1);
      } else if (pkType === 'expr') {
        refs.splice(0, refs.length, libItem(item));
        for (const prior of $('#pkG').querySelectorAll('[aria-pressed="true"]')) {
          prior.classList.remove('on');
          prior.setAttribute('aria-pressed', 'false');
        }
      } else if (garmentTypes.includes(pkType) && refs.length >= 2) {
        return toast('最多 2 张服装参考图');
      } else {
        refs.push(libItem(item));
      }
    }
    card.classList.toggle('on', on);
    card.setAttribute('aria-pressed', on);
    pkLastId = item.id;
  };

  // ---------- Neutral Wording Rewriter (reduces false safety flags without altering outfit/pose) ----------
  const NEUTRAL = [
    [/(?:胸部|臀部|腰臀|身材)(?:曲线|线条)?(?:突出|丰满|饱满|凸显)/g, ''],
    [/凸显(?:身材|曲线|胸部|臀部)/g, ''],
    [/火辣|撩人|诱惑|魅惑|勾人|性感/g, '时尚'],
    [/紧身/g, '修身'],
    [/露肩/g, '一字肩'],
    [/露背/g, '开背设计'],
    [/露脐/g, '短款'],
    [/低胸/g, 'V 领'],
    [/透视/g, '薄纱拼接'],
    [/比基尼|泳装|泳衣/g, '泳池边时装'],
    [/情趣内衣|内衣/g, '内搭'],
    [/裸露|裸体|半裸/g, ''],
    [/\b(?:emphasi[sz]ing|accentuating|showing off|highlighting) (?:her |his |the )?(?:curves|figure|body|bust|chest|hips|cleavage)\b/gi, ''],
    [/\b(?:curvy|voluptuous|busty)\b/gi, ''],
    [/\b(?:sexy|seductive|sultry|alluring|provocative|hot)\b/gi, 'stylish'],
    [/\b(?:skin-?tight|bodycon)\b/gi, 'fitted'],
    [/\btight\b/gi, 'fitted'],
    [/\boff[- ]the[- ]shoulder\b/gi, 'off-shoulder neckline'],
    [/\bbackless\b/gi, 'open-back design'],
    [/\b(?:bikini|swimsuit|swimwear)\b/gi, 'swimwear fashion catalog piece'],
    [/\b(?:lingerie|underwear|bra and panties)\b/gi, 'loungewear catalog set'],
    [/\b(?:nude|naked|topless|cleavage|bust|busty)\b/gi, ''],
    [/\bsee-?through\b/gi, 'sheer-panel'],
    [/\blow-?cut\b/gi, 'V-neck'],
  ];

  const FRAME = {
    en: 'Professional e-commerce fashion try-on photo for a clothing catalog. The subject is an adult, fully clothed in exactly the reference garments (same design, color and cut, nothing added or removed), keeping the same identity, in a natural, relaxed, non-suggestive pose, in a tasteful studio or lifestyle setting.',
    zh: '服装电商的专业试穿照片，用于商品目录。人物是成年人，完整穿着参考图中的服装（款式、颜色、剪裁完全一致，不增不减），保持同一人的长相，姿势自然放松，场景为简洁的影棚或生活场景。',
  };

  const neutralOn = () => LS('wbNeutral') !== false;
  $('#wbNeutral').checked = neutralOn();
  $('#wbNeutral').onchange = e => {
    SV('wbNeutral', e.target.checked);
    toast(e.target.checked ? '已开启减少误拦截' : '已关闭减少误拦截，将按原提示词发送');
  };

  const isAdultJob = (garments, set) =>
    [...garments, ...set].some(g => g.kind === 'lib' && A.ITEMS[g.k] && A.ITEMS[g.k].a);

  function neutralize(text, lang) {
    let out = text;
    for (const [pattern, replacement] of NEUTRAL) {
      out = out.replace(pattern, replacement ? `\x01${replacement}\x01` : '\x02');
    }
    out = out
      .replace(/\b(with|and|in)\s+\x02\s+(?:and|with)\b/gi, '$1')
      .replace(/\x01\x02*\x01/g, '、')
      .replace(/\x01\s+\x01/g, ' ')
      .replace(/[\x01]/g, '')
      .replace(/\s*\x02\s*/g, ' ')
      .replace(/(^|[，,、；;：:。.\n])\s*(?:and|with|,)?\s*(?=[，,、；;：:。.\n]|$)/g, '$1')
      .replace(/([，、；])[，、；]+/g, '$1')
      .replace(/([,;])\s*[,;]+/g, '$1')
      .replace(/[ \t]{2,}/g, ' ')
      .replace(/\s+([,.;，。；、])/g, '$1')
      .replace(/^[，,、]\s*|[，,、]\s*$/gm, '')
      .replace(/[,，、]\s*([.。;；])/g, '$1');
    const isZh = /[\u4e00-\u9fa5]/.test(out.slice(0, 40)) && lang !== 'en';
    return FRAME[isZh ? 'zh' : 'en'] + '\n' + out;
  }

  // ---------- Provider Settings Dialog ----------
  let checkpoints = [];
  let connectionCheck = null;

  function cancelConnectionCheck() {
    if (connectionCheck) connectionCheck.abort();
    connectionCheck = null;
  }

  function showCheckpoints(selected, files) {
    const field = $('#cfCheckpoint');
    field.replaceChildren(new Option('选择模型', ''), ...files.map(file => new Option(file, file)));
    field.value = files.includes(selected) ? selected : '';
  }

  function cfgFill(settings) {
    cancelConnectionCheck();
    const preset = settings.prov === 'local' && localPreset;
    $('#cfCheckpointW').hidden = !preset;
    if (preset) {
      const files = settings.checkpoint && !checkpoints.includes(settings.checkpoint)
        ? [settings.checkpoint, ...checkpoints]
        : checkpoints;
      showCheckpoints(settings.checkpoint, files);
    }
    $('#cfg').classList.toggle('local-engine', settings.prov === 'local');
    $$('[name=prov]').forEach(r => { r.checked = r.value === settings.prov; });
    const P = PROV[settings.prov];
    $('#cfBase').value = settings.base;
    $('#cfBase').placeholder = P.base || 'https://你的中转地址/v1';
    $('#cfKey').value = settings.key;
    $('#cfKey').placeholder = settings.prov === 'gemini' ? 'AIza…' : 'sk-… / 方舟 API Key';
    $('#cfModel').value = settings.model;
    $('#cfModel').placeholder = P.model;
    $('#cfModels').innerHTML = P.models.map(m => `<option value="${m}">`).join('');
    $('#cfField').value = settings.field;
    $('#cfProto').value = settings.proto;
    $('#cfProtoW').hidden = settings.prov !== 'custom';
    $('#cfFieldW').hidden = !(settings.prov === 'custom' && settings.proto === 'edits');
    $('#cfProxy').checked = settings.proxy;
    $('#cfHelp').innerHTML =
      ({
        local: preset ? '使用一张参考图。测试连接后选择模型。' : '使用本机已启动的引擎。',
        openai: '在 <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener">platform.openai.com/api-keys</a> 创建密钥；gpt-image-1 需要已验证的组织。',
        gemini: '在 <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Google AI Studio</a> 免费获取 API Key（AIza 开头）。',
        seedream: '在 <a href="https://console.volcengine.com/ark" target="_blank" rel="noopener">火山方舟控制台</a> 开通 Seedream 模型并创建 API Key；会自动走本站代理。',
        custom: '直连时，浏览器将密钥和图片发给填写的地址；启用本站代理后经本站转发。http/https 与任意端口均可。',
      })[settings.prov] || '';
    $('#cfMsg').textContent = '';
    $('#cfMsg').className = 'fine';
  }

  function cfgRead() {
    const prov = $('[name=prov]:checked').value;
    const settings = {
      prov,
      base: $('#cfBase').value.trim(),
      key: $('#cfKey').value.trim(),
      model: $('#cfModel').value.trim(),
      field: $('#cfField').value,
      proto: $('#cfProto').value,
      proxy: $('#cfProxy').checked,
    };
    if (prov === 'local' && localPreset) settings.checkpoint = $('#cfCheckpoint').value;
    return settings;
  }

  const dialogOpeners = new WeakMap();
  function openDialog(dialog, focusEl, opener = document.activeElement) {
    dialogOpeners.set(dialog, opener);
    dialog.hidden = false;
    document.body.style.overflow = 'hidden';
    focusEl.focus({ preventScroll: true });
  }
  function closeDialog(dialog) {
    dialog.hidden = true;
    document.body.style.overflow = '';
    const opener = dialogOpeners.get(dialog);
    dialogOpeners.delete(dialog);
    if (opener?.isConnected) opener.focus({ preventScroll: true });
  }

  function cfgOpen() {
    if (desktop) { void desktop.settingsDialog(); return; }
    $('#cfKey').type = 'password';
    $('#cfEye').textContent = '显示';
    cfgFill(cfg);
    openDialog($('#cfg'), $('[name=prov]:checked'));
  }
  const cfgClose = () => {
    cancelConnectionCheck();
    closeDialog($('#cfg'));
  };

  $('#wbCfg').onclick = cfgOpen;
  $('#wbCfg2').onclick = cfgOpen;
  $('#cfgX').onclick = cfgClose;
  $('#cfg').addEventListener('click', e => { if (e.target.id === 'cfg') cfgClose(); });
  $$('[name=prov]').forEach(r => {
    r.onchange = () => {
      const draft = cfgRead();
      if (draft.prov === cfg.prov) cfgFill(cfg);
      else cfgFill({ ...draft, base: '', key: '', model: '', proxy: draft.prov === 'seedream' });
      if (draft.prov === 'seedream') msg('火山方舟接口不允许浏览器跨域，已开启本站代理');
    };
  });
  $('#cfProto').onchange = () => { $('#cfFieldW').hidden = $('#cfProto').value !== 'edits'; };
  $('#cfEye').onclick = () => {
    const k = $('#cfKey');
    k.type = k.type === 'password' ? 'text' : 'password';
    $('#cfEye').textContent = k.type === 'password' ? '显示' : '隐藏';
  };
  $('#cfSave').onclick = () => {
    const next = cfgRead();
    const error = configError(next);
    if (error) return msg(error, 1);
    try {
      localStorage.setItem('wbCfg', JSON.stringify(next));
    } catch {
      return msg('无法保存设置，请检查浏览器存储权限', 1);
    }
    Object.assign(cfg, next);
    render();
    cfgClose();
    toast('API 设置已保存在本机');
  };
  $('#cfClr').onclick = () => {
    cancelConnectionCheck();
    cfg.key = '';
    $('#cfKey').value = '';
    render();
    try {
      localStorage.setItem('wbCfg', JSON.stringify(cfg));
      msg('已从本机清除密钥');
    } catch {
      msg('当前页面已清除密钥，浏览器存储未能更新', 1);
    }
  };

  function msg(text, bad) {
    const m = $('#cfMsg');
    m.textContent = text;
    m.className = 'fine ' + (bad ? 'bad' : 'good');
  }
  $('#cfg').addEventListener('input', () => {
    cancelConnectionCheck();
    $('#cfMsg').textContent = '';
  });

  $('#cfTest').onclick = async () => {
    const settings = cfgRead();
    cancelConnectionCheck();
    if (settings.prov !== 'local') {
      if (!settings.key) return msg('请先填写 API Key', 1);
      const invalid = configError(settings);
      if (invalid) return msg(invalid, 1);
    }
    const check = new AbortController();
    connectionCheck = check;
    msg('正在连接…');
    try {
      if (settings.prov !== 'local') {
        const result = await testConnection(settings, modelOf(settings), check.signal);
        if (!check.signal.aborted) msg(result);
        return;
      }
      const reply = await fetch('/api/local/status', { headers: { 'x-posewear-local': '1' }, signal: check.signal });
      if (localPreset && reply.status === 503) throw Error('无法连接本机 ComfyUI。请先启动服务，再测试连接。');
      if (!reply.ok) throw Error(await reply.text());
      const status = await reply.json();
      if (check.signal.aborted) return;
      if (localPreset) {
        checkpoints = status.checkpoints;
        showCheckpoints($('#cfCheckpoint').value, checkpoints);
        if (!checkpoints.length) return msg('ComfyUI 中还没有可用的 checkpoint', 1);
      }
      msg('本机引擎已连接');
    } catch (error) {
      if (!check.signal.aborted) msg(settings.prov === 'local' ? error.message : nice(error), 1);
    } finally {
      if (connectionCheck === check) connectionCheck = null;
    }
  };

  const p2 = n => String(n).padStart(2, '0');
  const stamp = d => `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-${p2(d.getHours())}${p2(d.getMinutes())}`;

  // ---------- Generation Pipeline ----------
  let tick;
  let done = 0;
  let total = 1;

  const stage = text => { $('#wbStage1').textContent = text; };

  function busy(on) {
    const panel = $('#wbBusy');
    const go = $('#wbGo');
    const progress = $('#wbProg');
    panel.hidden = !on;
    go.classList.toggle('run', on);
    go.disabled = on;
    clearInterval(tick);
    if (!on) { render(); return; }
    go.textContent = '生成中…';
    const started = Date.now();
    const update = () => {
      progress.max = total;
      if (total > 1) progress.value = done;
      else progress.removeAttribute('value');
      $('#wbTime').textContent = Math.floor((Date.now() - started) / 1000) + ' 秒' + (total > 1 ? ` · 已完成 ${done} / ${total}` : '');
    };
    update();
    tick = setInterval(update, 1000);
  }

  let retryBatch = null;

  function jobBatch(opts) {
    const state = opts.selection ? selectionState(opts.selection, opts.lang) : { ...W };
    const mode = state.persons.length && state.garments.length ? state.mode : 'gen';
    const persons = state.persons.map(p => ({ ...p }));
    const garments = state.garments.map(g => ({ ...g }));
    const exprs = state.exprs.map(e => ({ ...e }));
    const poses = state.poses.map(p => ({ ...p }));
    const groups = poseGroups(poses);
    const split = !opts.quick && groups.length > 1 && state.bm === 'each';
    const sets = split ? groups : [poses];
    const count = Number(opts.n || $('#wbN').value) || 1;
    const custom = opts.prompt ?? (!opts.selection && W.dirty ? $('#wbPrompt').value.trim() : null);
    const ar = opts.ar || $('#wbAR').value;
    const splitStages = !!desktop && !opts.quick && $('#wbTwoStage').checked &&
      !!persons.length && !!garments.length && !!poses.length;

    const jobs = [];
    for (const set of sets) {
      const original = custom || compose(set, state);
      const soft = garments.length > 0 && $('#wbNeutral').checked && !isAdultJob(garments, [...set, ...exprs, ...state.extraRefs]);
      const prompt = soft ? neutralize(original, state.pl) : original;
      const label = poseGroups(set)
        .map(([ref]) => {
          const item = A.ITEMS[ref.k];
          const view = item.angles[ref.ai ?? item.ai];
          return `${item.zh} · ${view?.zh || '原图'}`;
        })
        .join(' / ');
      let stages;
      if (splitStages && !custom) {
        const pose = compose(set, { ...state, mode: 'gen', garments: [] });
        const garment = compose([], { ...state, mode: 'tryon', persons: [persons[0]], exprs: [],
          extraRefs: [], keepBg: true, extra: '' });
        stages = { pose, garment: soft ? neutralize(garment, state.pl) : garment };
      }
      for (let i = 0; i < count; i++) jobs.push({ set, prompt, original, soft, label, stages });
    }
    return {
      jobs,
      persons,
      garments,
      exprs,
      extraRefs: state.extraRefs.map(ref => ({ ...ref })),
      mode,
      ar,
      lang: state.pl,
      bm: state.bm,
      customStagedPrompt: splitStages && !!custom,
    };
  }

  function recordInputs(batch, job, imgs) {
    const ref = item => ({ k: item.k, ai: item.ai });
    return {
      persons: batch.persons.map((p, i) => ({ k: p.k, w: p.w, h: p.h, ...(i ? { blob: imgs[i] } : {}) })),
      garments: batch.garments.map(g => (g.kind === 'file' ? { name: g.name, blob: g.blob } : ref(g))),
      poses: job.set.map(ref),
      exprs: batch.exprs.map(ref),
      extraRefs: batch.extraRefs.map(ref),
      ar: batch.ar,
      lang: batch.lang,
      bm: batch.bm,
    };
  }

  async function run(opts = {}) {
    const outcome = { got: [], errs: [], warns: [], unsaved: 0, cancelled: false };
    if (W.busy) { outcome.errs.push(Error('已有生成任务正在进行')); return outcome; }
    if (!opts.quick && !opts.retry) {
      const need = updateStart();
      if (need.length) return toast('请先' + need.join('、'));
    }
    const batch = opts.retry || jobBatch(opts);
    if (batch.customStagedPrompt) {
      const error = Error('分两步编辑使用分别编号的自动提示词。请恢复自动提示词，或关闭分两步编辑后提交自定义指令。');
      outcome.errs.push(error);
      errBox(error, '', true);
      return outcome;
    }
    if (!batch.jobs[0].original.trim()) { outcome.errs.push(Error('选择素材或输入提示词')); return outcome; }
    if (!configured()) { toast('请先配置你的 API'); cfgOpen(); outcome.nokey = true; return outcome; }
    const invalid = desktop ? null : configError(cfg);
    if (invalid) { outcome.errs.push(Error(invalid)); cfgOpen(); msg(invalid, 1); return outcome; }

    const settings = { ...cfg };
    const controller = new AbortController();
    const signal = controller.signal;
    W.busy = controller;
    if (!opts.quick) retryBatch = null;
    $('#wbErr').hidden = true;
    $('#wbWarn').hidden = true;
    done = 0;
    total = batch.jobs.length;
    if (!opts.quick) busy(true);
    stage('正在准备素材…');

    const warnings = new Set();
    const failed = [];
    const retained = opts.retry?.retained || [];
    let nativeJobs;
    const blobs = new Map();
    const imageFor = item => {
      if (!blobs.has(item)) blobs.set(item, assetBlob(item, signal));
      return blobs.get(item);
    };

    let next = 0;
    const worker = async () => {
      while (next < batch.jobs.length && !signal.aborted) {
        const idx = next++;
        const job = batch.jobs[idx];
        try {
          const items = [...batch.persons, ...batch.garments, ...job.set, ...batch.exprs, ...batch.extraRefs];
          const imgs = desktop ? [] : await Promise.all(items.map(imageFor));
          signal.throwIfAborted();
          stage(total > 1 ? `模型正在绘制… 共 ${total} 张` : '模型正在绘制…');
          W.lastSent = job.prompt;

          const request = new AbortController();
          const cancel = () => request.abort();
          signal.addEventListener('abort', cancel, { once: true });
          let timedOut = false;
          let result;
          const timer = desktop ? null : setTimeout(() => { timedOut = true; request.abort(); }, window.__TMO || 180000);
          try {
            result = desktop ? await desktop.wait(nativeJobs[idx], request.signal, job => {
              stage(job.state === 'queued' ? '等待本机 GPU…' :
                (job.phase === 'denoise' ? `本机生成 · ${job.completed} / ${job.total} 步` : '本机模型正在处理图片…'));
            }) : await generateImage({
              images: imgs,
              prompt: job.prompt,
              signal: request.signal,
              ratio: batch.ar,
              settings,
              model: modelOf(settings),
              checkpoint: localPreset ? settings.checkpoint : null,
            });
            request.signal.throwIfAborted();
          } catch (error) {
            if (timedOut && !signal.aborted) {
              const e = Error('timeout');
              e.zh = '等了 3 分钟还没有返回。服务商可能正忙，稍后点「重试」。';
              throw e;
            }
            throw error;
          } finally {
            clearTimeout(timer);
            signal.removeEventListener('abort', cancel);
          }

          if (result.warn) warnings.add(result.warn);
          const rec = desktop ? desktop.record(result) : {
            t: Date.now() + Math.random(),
            prov: settings.prov,
            model: modelOf(settings),
            prompt: job.prompt,
            orig: job.soft ? job.original : null,
            neutral: job.soft,
            mode: batch.mode,
            label: job.label,
            before: batch.persons.length ? imgs[0] : null,
            after: result.blob,
            inputs: recordInputs(batch, job, imgs),
          };
          outcome.got.push(rec);
          if (desktop) desktop.remember(rec);
          else { try { await save(rec); } catch { outcome.unsaved++; } }
          if (!opts.quick) {
            W.results = [...retained, ...outcome.got];
            show(W.results.length - 1);
          }
        } catch (error) {
          if (signal.aborted) break;
          error.sent = job.prompt;
          outcome.errs.push(error);
          failed.push(job);
        } finally {
          done++;
          if (opts.onTick) opts.onTick(done, total);
        }
      }
    };

    try {
      if (desktop) nativeJobs = await desktop.submit(batch, A.ITEMS, signal);
      await Promise.all(Array.from({ length: Math.min(2, batch.jobs.length) }, worker));
      outcome.cancelled = signal.aborted;
      if (outcome.unsaved) warnings.add('生成完成，部分图片未能存入生成记录，请先下载。');
      if (outcome.got.length) { hist(); window.HIS?.refresh(); }
      if (signal.aborted) {
        toast('已取消生成');
      } else if (outcome.got.length) {
        toast(outcome.unsaved ? '记录保存失败，请先下载' : outcome.got.length > 1 ? `完成 ${outcome.got.length} 张` : '生成完成');
      }
      if (!opts.quick && failed.length) {
        retryBatch = { ...batch, jobs: failed, retained: [...retained, ...outcome.got] };
        const error = outcome.errs.find(e => e.safe) || outcome.errs[0];
        errBox(error, retained.length || outcome.got.length ? `${failed.length} 张生成失败` : '', batch.garments.length > 0);
      }
      outcome.warns = [...warnings];
      if (!opts.quick && warnings.size) {
        $('#wbWarn').hidden = false;
        $('#wbWarn').textContent = outcome.warns.join(' ');
      }
    } catch (error) {
      outcome.cancelled = signal.aborted;
      if (!signal.aborted) {
        outcome.errs.push(error);
        if (!opts.quick) { retryBatch = batch; errBox(error, '', batch.garments.length > 0); }
      }
    } finally {
      W.busy = null;
      if (!opts.quick) busy(false);
    }
    return outcome;
  }

  function errBox(e, head, hasGarment) {
    const el = $('#wbErr');
    el.hidden = false;
    el.classList.toggle('safe', !!e.safe);
    if (e.safe) {
      const safety = e.safe;
      const where = safety.where === 'in' ? '提交的参考图' : safety.where === 'out' ? '生成结果' : '本次请求';
      const reason = safety.cats.map(cat => esc(window.PWLocale.t(cat))).join(' · ');
      const edit = hasGarment
        ? '<button class="sm ghost" data-ef="cos">换一套服装</button>'
        : '<button class="sm ghost" data-ef="prompt">编辑提示词</button>';
      const detail =
        e.raw +
        (safety.rid && !e.raw.includes(safety.rid) ? '\nrequest id: ' + safety.rid : '') +
        (e.sent ? `\n\n${window.PWLocale.t('实际发送的提示词')}\n${e.sent}` : '');
      el.innerHTML =
        `<b>${head ? esc(head) + ' · ' : ''}<span>被服务商的安全审核拦下了</span></b>` +
        `<p>${esc(PROV[cfg.prov].name)} · <span>${where}</span>${reason ? ' · ' + reason : ''}</p>` +
        `<div class="row"><button class="sm" data-ef="prov">换服务商重试</button>${edit}<button class="sm ghost" id="wbRetry">重试</button></div>` +
        `${hasGarment && !neutralOn() ? '<button class="sm" data-ef="neutral">打开减少误拦截后重试</button>' : ''}` +
        `<details class="raw"><summary>查看原始信息</summary><pre>${esc(detail)}</pre><button class="sm ghost" data-ef="copy">复制</button></details>`;
      return;
    }
    el.innerHTML =
      `<b>${head || '生成失败'}</b>` +
      `<p>${esc(nice(e)).replace(/\n/g, '<br>')}</p>` +
      `<div class="row"><button class="sm" id="wbRetry">重试</button><button class="sm ghost" data-open-cfg>检查 API 设置</button></div>`;
  }

  $('#wbStage').addEventListener('click', e => {
    const f = e.target.closest('[data-ef]');
    if (f) {
      const action = f.dataset.ef;
      if (action === 'prov') {
        cfgOpen();
        const p = $('#cfg .prov');
        p.scrollIntoView({ block: 'start' });
        p.classList.remove('hl');
        void p.offsetWidth;
        p.classList.add('hl');
        const other = $('[name=prov]:not(:checked)');
        other?.focus();
      } else if (action === 'cos') {
        picker('costume');
      } else if (action === 'prompt') {
        $('#wbPrompt').focus();
      } else if (action === 'neutral') {
        SV('wbNeutral', true);
        $('#wbNeutral').checked = true;
        run();
      } else if (action === 'copy') {
        A.copy(f.previousElementSibling.textContent);
      }
      return;
    }
    if (e.target.id === 'wbRetry' && retryBatch) run({ retry: retryBatch });
    if (e.target.closest('[data-open-cfg]')) cfgOpen();
  });

  $('#wbGo').onclick = () => run();
  $('#wbAgain').onclick = () => run();
  $('#wbStop').onclick = () => W.busy?.abort();

  // ---------- Result Display & Comparison ----------
  const ru = new Map();
  const U = blob => {
    if (typeof blob === 'string') return blob;
    if (!ru.has(blob)) ru.set(blob, URL.createObjectURL(blob));
    return ru.get(blob);
  };

  function setX(x) {
    $('#wbCmp').style.setProperty('--x', x + '%');
    $('#wbRange').value = x;
  }
  $('#wbRange').oninput = e => setX(e.target.value);

  function show(i, meta) {
    const r = W.results[i];
    if (!r) return;
    W.view = 'result';
    let details = $('#wbSentD');
    if (!details) {
      details = document.createElement('details');
      details.id = 'wbSentD';
      details.className = 'raw sent';
      $('#wbOutAct').after(details);
    }
    details.hidden = !r.prompt;
    details.innerHTML = r.prompt
      ? `<summary>查看原始信息${r.neutral ? ' · 已减少误拦截' : ''}</summary><pre>${esc((r.orig ? '原提示词：\n' + r.orig + '\n\n实际发送：\n' : '实际发送：\n') + r.prompt)}</pre>`
      : '';

    W.ri = i;
    const afterUrl = U(r.after);
    const hasBefore = !!r.before;
    $('#wbAfter').src = afterUrl;
    if (hasBefore) $('#wbBefore').src = U(r.before);
    $('#wbCmp').classList.toggle('solo', !hasBefore || !W.cmp);
    $('#wbCmpT').hidden = !hasBefore;
    $('#wbCmpT').textContent = W.cmp ? '关闭对比' : '前后对比';
    paintBoard();
    $('#wbErr').hidden = true;
    setX(50);
    $('#wbZip').hidden = W.results.length < 2;

    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
      $('#wbAfter').animate(
        [{ opacity: 0, transform: 'scale(1.03)', filter: 'blur(8px)' }, { opacity: 1, transform: 'none', filter: 'none' }],
        { duration: 620, easing: 'cubic-bezier(.16,1,.3,1)' }
      );
      if (hasBefore && W.cmp && !W._wig) {
        W._wig = 1;
        const t0 = performance.now();
        const wiggle = t => {
          const p = (t - t0) / 1400;
          if (p >= 1) return setX(50);
          setX(50 + Math.sin(p * Math.PI * 2) * 12 * (1 - p));
          requestAnimationFrame(wiggle);
        };
        setTimeout(() => requestAnimationFrame(wiggle), 500);
      }
    }

    $('#wbMeta').textContent = meta || ((PROV[r.prov] || { name: r.prov }).name + ' · ' + r.model + (r.label ? ' · ' + r.label : ''));
    $('#wbRes').innerHTML = W.results
      .map((x, j) => `<button data-ri="${j}" class="${j === i ? 'on' : ''}"><img src="${U(x.after)}" alt="结果 ${j + 1}"></button>`)
      .join('');
    const rect = $('#wbStage').getBoundingClientRect();
    if (rect.bottom < 0 || rect.top > innerHeight) {
      $('#wbStage').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'nearest' });
    }
  }

  $('#wbRes').onclick = e => { const b = e.target.closest('[data-ri]'); if (b) show(+b.dataset.ri); };
  $('#wbCmpT').onclick = () => { W.cmp = !W.cmp; show(W.ri); };
  $('#wbZip').onclick = async () => {
    if (!window.makeZip) return;
    try {
      const ts = stamp(new Date());
      const files = await Promise.all(W.results.map(async (r, i) => ({
        name: `posewear-${r.mode === 'gen' ? 'gen' : 'tryon'}-${ts}-${String(i + 1).padStart(2, '0')}.png`,
        blob: desktop ? await desktop.asBlob(r.after) : r.after,
      })));
      await window.X.dl(await makeZip(files), `posewear-tryon-${ts}.zip`);
      toast(`已打包 ${files.length} 张`);
    } catch (error) { toast(error.message); }
  };
  $('#cfExp').onclick = () => {
    const notice = '导出时包含 API Key 吗？\n确定 = 包含，取消 = 不含密钥';
    const withKey = confirm(window.PWLocale.t(notice));
    const copyCfg = cfgRead();
    if (!withKey) delete copyCfg.key;
    window.X.dl(
      new Blob([JSON.stringify({ app: '衣姿 PoseWear', type: 'api-config', cfg: copyCfg }, null, 2)], { type: 'application/json' }),
      '衣姿-API设置.json'
    );
    msg('已导出设置' + (withKey ? '（含密钥，请妥善保管）' : '（不含密钥）'));
  };
  $('#cfImp').onclick = () => $('#cfImpF').click();
  $('#cfImpF').onchange = async e => {
    try {
      const parsed = JSON.parse(await e.target.files[0].text());
      const incoming = parsed.cfg || parsed;
      if (!Object.hasOwn(PROV, incoming.prov) || ['base', 'key', 'model', 'checkpoint'].some(k => incoming[k] != null && typeof incoming[k] !== 'string')) {
        throw Error();
      }
      cfgFill({ ...cfg, ...incoming, key: incoming.key || '' });
      msg('已读入设置，点击保存后生效');
    } catch {
      msg('导入失败：文件格式不正确', 1);
    }
    e.target.value = '';
  };
  $('#wbDl').onclick = () => {
    const r = W.results[W.ri];
    if (!r) return;
    if (desktop && r.tcOutput) {
      void window.TurboCider.media.save(r.tcOutput).catch(error => toast(error.message));
      return;
    }
    const link = document.createElement('a');
    link.href = U(r.after);
    link.download = `posewear-${r.mode === 'gen' ? 'gen' : 'tryon'}-${stamp(new Date(r.t))}-${String(W.ri + 1).padStart(2, '0')}.png`;
    link.click();
  };
  $('#wbUse').onclick = async () => {
    const r = W.results[W.ri];
    if (!r) return;
    if (desktop && r.tcOutput) {
      try {
        const image = await window.TurboCider.media.get(r.tcOutput);
        W.persons = [{ tcMediaID: image.id, url: image.url, w: image.width, h: image.height }];
      } catch (error) { toast(error.message); return; }
    } else {
      const x = await toJpeg(U(r.after), 1536);
      W.persons = [{ blob: x.blob, url: URL.createObjectURL(x.blob), w: x.w, h: x.h }];
    }
    render();
    toast('已将结果设为新的人物照片');
  };

  // ---------- IndexedDB Local History ----------
  let dbPromise;
  const openDB = () =>
    (dbPromise ||= new Promise((resolve, reject) => {
      const req = indexedDB.open('lk-studio', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('h', { keyPath: 't' });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    }));

  const tx = async (mode, fn) => {
    if (desktop) return desktop.transaction(mode, fn);
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction('h', mode);
      const store = t.objectStore('h');
      const query = fn(store);
      t.oncomplete = () => resolve(query && query.result);
      t.onerror = () => reject(t.error);
    });
  };

  async function save(rec) {
    await tx('readwrite', store => {
      store.put(rec);
      const keys = store.getAllKeys();
      keys.onsuccess = () => keys.result.slice(0, -200).forEach(key => store.delete(key));
    });
  }

  let histUrls = [];
  async function hist() {
    let list;
    try {
      list = await tx('readonly', s => s.getAll());
    } catch {
      $('#wbHist').textContent = '无法读取生成记录，请检查浏览器存储权限';
      return;
    }
    list.sort((a, b) => b.t - a.t);
    histUrls.forEach(URL.revokeObjectURL);
    histUrls = [];
    const recent = list.slice(0, 30);
    $('#wbHist').innerHTML = recent.length
      ? recent
          .map(r => {
            const url = typeof r.after === 'string' ? r.after : URL.createObjectURL(r.after);
            if (typeof r.after !== 'string') histUrls.push(url);
            const titleText = new Date(r.t).toLocaleString('zh-CN') + (r.label ? ' · ' + r.label : '');
            const timeText = new Date(r.t).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
            return `<button data-t="${r.t}" title="${esc(titleText)}"><img src="${url}" alt="历史结果"><span>${r.mode === 'gen' ? '生图' : '试穿'} · ${esc(timeText)}</span><i data-del="${r.t}" aria-label="删除">×</i></button>`;
          })
          .join('') +
        (list.length > 30 ? `<button class="hist-more" data-go="history" title="查看全部 ${list.length} 条记录"><b>全部</b><small>${list.length}条 ↗</small></button>` : '')
      : '';
    W.hist = list;
  }

  $('#wbHist').onclick = async e => {
    const delBtn = e.target.closest('[data-del]');
    if (delBtn) {
      e.stopPropagation();
      const rec = (W.hist || []).find(x => x.t == delBtn.dataset.del);
      await tx('readwrite', s => s.delete(+delBtn.dataset.del));
      await hist();
      if (rec) {
        toast('已删除 1 条记录', async () => {
          await tx('readwrite', s => s.put(rec));
          hist();
        });
      }
      return;
    }
    const card = e.target.closest('[data-t]');
    if (!card) return;
    const rec = W.hist.find(x => x.t == card.dataset.t);
    if (rec) {
      W.results = [rec];
      show(0, `${(PROV[rec.prov] || { name: rec.prov }).name} · ${rec.model} · ${new Date(rec.t).toLocaleString('zh-CN')}`);
      $('#wbStage').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'nearest' });
    }
  };
  $('#wbHClr').onclick = async () => {
    const all = W.hist || [];
    if (!all.length) return toast('还没有生成记录');
    await tx('readwrite', s => s.clear());
    await hist();
    toast(`已清空 ${all.length} 条记录`, async () => {
      await tx('readwrite', s => { all.forEach(r => s.put(r)); });
      hist();
    });
  };

  // ---------- Public Studio Integration API ----------
  const libItem = ref => ({ kind: 'lib', k: ref.k, ai: ref.ai ?? A.ITEMS[ref.k]?.ai });
  const fileItem = (blob, name) => ({ kind: 'file', blob, name, url: URL.createObjectURL(blob) });

  function groupLibraryRefs(refs) {
    const groups = { portrait: [], garment: [], pose: [], expr: [], extra: [] };
    for (const ref of refs) {
      const item = A.ITEMS[ref.k];
      if (item && !item.previewOnly) groups[A.studioSlot(item.type)].push(libItem(ref));
    }
    return groups;
  }

  let initPromise;
  function init() {
    if (!initPromise) {
      initPromise = (async () => {
        const refs = [...(g0.r || []), ...(g0.g || []), ...(g0.p || []), ...(g0.e || []), ...(g0.x || [])]
          .map(saved => (Array.isArray(saved) ? { k: saved[0], ai: saved[1] } : { k: saved }));
        await A.needKeys(refs.map(ref => ref.k));
        const groups = groupLibraryRefs(refs);
        if (!W.persons.length) W.persons = groups.portrait.slice(0, 3).map(libPortrait);
        if (!W.garments.length) W.garments = groups.garment.slice(0, 2);
        if (desktop) {
          const local = LS('wbLocalDraft');
          const restore = async ref => {
            try { return { ...ref, kind: 'file', url: (await window.TurboCider.media.get(ref.tcMediaID)).url }; }
            catch { return null; }
          };
          for (const group of ['persons', 'garments']) {
            const locals = (await Promise.all((local?.[group] || []).map(restore))).filter(Boolean);
            W[group] = [...W[group], ...locals].slice(0, group === 'persons' ? 3 : 2);
          }
        }
        if (!W.poses.length) W.poses = groups.pose.slice(0, 6);
        if (!W.exprs.length) W.exprs = groups.expr.slice(0, 1);
        if (!W.extraRefs.length) W.extraRefs = groups.extra;
        hist();
      })().catch(error => {
        initPromise = null;
        throw error;
      });
    }
    return initPromise;
  }

  function selectionState(sel, lang) {
    const groups = groupLibraryRefs([
      ...(sel.portrait || []),
      ...(sel.garment || []),
      ...(sel.pose || []),
      ...(sel.expr || []),
      ...(sel.extra || []),
    ]);
    const persons = groups.portrait.slice(0, 1).map(libPortrait);
    const garments = groups.garment.slice(0, 2);
    return {
      ...W,
      extra: '',
      charTxt: Object.values(groups).some(items => items.length) ? W.charTxt : '',
      pl: lang ? (lang === 'zh' ? 'zh' : 'en') : W.pl,
      persons,
      garments,
      poses: groups.pose.slice(0, 6),
      exprs: groups.expr.slice(0, 1),
      extraRefs: groups.extra,
      mode: persons.length && garments.length ? 'tryon' : 'gen',
      bm: 'sheet',
    };
  }

  let importVersion = 0;
  async function setSel(sel, extraText, draft) {
    const version = ++importVersion;
    await init();
    if (version !== importVersion) return;
    const state = selectionState(sel);
    W.extraRefs = state.extraRefs;
    W.garments = state.garments;
    W.poses = state.poses;
    W.exprs = state.exprs;
    if (state.persons.length || draft) W.persons = state.persons;
    else W.persons = W.persons.filter(p => !p.k);
    if (draft) {
      setMode(state.mode);
      W.pl = draft.lang;
      W.bm = state.bm;
      SV('wbPl', W.pl);
      SV('wbBm', W.bm);
      $('#wbN').value = draft.count;
      $('#wbAR').value = draft.ratio;
      $('#wbExtra').value = extraText || '';
    } else {
      appendExtra(extraText);
    }
    reprompt(true);
    if (draft?.prompt) {
      $('#wbPrompt').value = draft.prompt;
      W.dirty = true;
    }
    saveDraft();
    render();
    return $('#wbPrompt').value;
  }

  function appendExtra(text) {
    if (!text) return;
    const input = $('#wbExtra');
    if (!input.value.includes(text)) {
      input.value = [input.value.trim(), text].filter(Boolean).join('\n');
      saveDraft();
    }
  }

  async function addSelection(sel, extraText) {
    await init();
    const refs = groupLibraryRefs([
      ...(sel.portrait || []),
      ...(sel.garment || []),
      ...(sel.pose || []),
      ...(sel.expr || []),
      ...(sel.extra || []),
    ]);
    if (refs.garment.length) W.garments = refs.garment.slice(0, 2);
    if (refs.expr.length) W.exprs = refs.expr.slice(0, 1);
    for (const it of refs.pose) {
      if (W.poses.some(p => p.k === it.k)) continue;
      if (W.poses.length >= 6) W.poses.shift();
      W.poses.push(libItem(it));
    }
    for (const it of refs.portrait) {
      if (W.persons.length >= 3) break;
      if (!W.persons.some(p => p.k === it.k)) await addPortrait(it);
    }
    for (const it of refs.extra) {
      if (!W.extraRefs.some(r => r.k === it.k && r.ai === it.ai)) W.extraRefs.push(it);
    }
    appendExtra(extraText);
    render();
  }

  async function pick(it) {
    if (it.previewOnly) return toast('这张素材支持原站预览。');
    await init();
    if (it.type === 'pose' || it.type === 'depth') {
      if (!W.poses.some(p => p.k === it.k)) {
        if (W.poses.length >= 6) W.poses.shift();
        W.poses.push(libItem(it));
      }
      toast(`已加入姿势参考（${W.poses.length} 个）`);
    } else if (it.type === 'portrait') {
      await addPortrait(it);
    } else if (it.type === 'expr') {
      W.exprs = [libItem(it)];
      toast('已加入表情参考（只取神态）');
    } else if (garmentTypes.includes(it.type)) {
      W.garments = [libItem(it)];
      toast('已选为试穿服装');
    } else {
      if (!W.extraRefs.some(r => r.k === it.k && r.ai === it.ai)) W.extraRefs.push(libItem(it));
      toast('已加入工作台');
    }
    A.setTab('studio');
  }

  async function restoreRecord(r) {
    await init();
    let inputs = r.inputs;
    const problems = [];
    if (desktop && inputs) {
      inputs = { ...inputs };
      // Keep the exact previously used image available when the online catalog is closed.
      for (const name of ['garments', 'poses', 'exprs', 'extraRefs']) {
        inputs[name] = await Promise.all((inputs[name] || []).map(async ref => {
          if (!ref.sourceCollection || !ref.mediaID) return ref;
          try {
            const image = await window.TurboCider.media.get(ref.mediaID);
            const k = A.addLocalItem(ref.sourceCollection, { ...image, name: ref.sourceTitle || image.name });
            return { ...ref, k, ai: 0 };
          } catch { return ref; }
        }));
      }
    }
    const garments = inputs?.garments || [];
    const savedRefs = [...garments, ...(inputs?.poses || []), ...(inputs?.exprs || []), ...(inputs?.extraRefs || [])].filter(ref => ref.k);
    const personKeys = (inputs?.persons || []).filter(ref => ref.k).map(ref => ref.k);
    try { await A.needKeys([...personKeys, ...savedRefs.map(ref => ref.k)]); } catch { problems.push('部分素材库未能加载'); }
    if (savedRefs.some(ref => !A.ITEMS[ref.k])) problems.push('部分参考图需重选');

    const groups = groupLibraryRefs(savedRefs);
    W.garments = garments.flatMap(g =>
      desktop && g.mediaID && !g.k
        ? [{ kind: 'file', tcMediaID: g.mediaID, name: g.name, url: desktop.imageURL(g.mediaID) }]
        : g.blob
        ? [fileItem(g.blob, g.name)]
        : A.ITEMS[g.k] && A.studioSlot(A.ITEMS[g.k].type) === 'garment'
          ? [libItem(g)]
          : []
    );
    W.poses = groups.pose;
    W.exprs = groups.expr;
    W.extraRefs = groups.extra;
    W.persons = groups.portrait.map(libPortrait);

    const savedPeople = inputs?.persons || (r.before ? [{}] : []);
    for (const [i, p] of savedPeople.entries()) {
      if (desktop && p.mediaID) {
        try {
          const image = await window.TurboCider.media.get(p.mediaID);
          W.persons.push({ tcMediaID: image.id, url: image.url, w: image.width, h: image.height, name: image.name });
        } catch { problems.push('部分人物照片需重选'); }
        continue;
      }
      const blob = i ? p.blob : r.before;
      if (!blob) { problems.push('部分人物照片需重选'); continue; }
      let w = p.w;
      let h = p.h;
      if (!w || !h) {
        const image = await createImageBitmap(blob).catch(() => null);
        if (image) { w = image.width; h = image.height; image.close(); }
      }
      W.persons.push({ blob, k: p.k, w: w || '?', h: h || '?', url: URL.createObjectURL(blob) });
    }

    autoMode = false;
    W.mode = r.mode || 'tryon';
    W.pl = inputs?.lang || W.pl;
    W.bm = inputs?.bm || 'each';
    $('#wbAR').value = inputs?.ar || 'auto';
    $('#wbN').value = '1';
    $('#wbExtra').value = '';
    $('#wbNeutral').checked = !!r.neutral;
    $('#wbPrompt').value = r.orig || r.prompt || '';
    W.dirty = !!$('#wbPrompt').value;
    SV('wbMode', W.mode);
    SV('wbPl', W.pl);
    SV('wbBm', W.bm);
    saveDraft();
    if (!inputs) problems.push('旧记录的服装和姿势需重选');
    if (!desktop && (cfg.prov !== r.prov || modelOf() !== r.model)) problems.push('请核对当前服务商及模型');
    W.view = 'board';
    render();
    return problems;
  }

  window.WB = {
    copyStudioBrief,
    async show() {
      try {
        await init();
        render();
        if (!W.results.length) loadExample();
      } catch {
        toast('试穿台选材加载失败，请重新打开后再试');
      }
    },
    pick,
    restoreRecord,
    _W: W,
    _cfg: cfg,
    db: {
      tx,
      save,
      hist: () => tx('readonly', s => s.getAll()),
      del: t => tx('readwrite', s => s.delete(t)),
      clear: () => tx('readwrite', s => s.clear()),
      put: r => tx('readwrite', s => s.put(r)),
      refreshHist: hist,
    },
    quick: opts => run({ ...opts, quick: true }),
    cfgOpen,
    exportBoardPng,
    nice,
    busy: () => !!W.busy,
    stop: () => W.busy?.abort(),
    configured,
    provLabel,
    setSel,
    addSelection,
    selectionPrompt: (sel, lang) => compose(undefined, selectionState(sel, lang)),
    openPicker: (type, sink) => { pkSink = sink; picker(type); },
  };

  if (document.body.classList.contains('studio')) WB.show();
  if (location.hash === '#studio') {
    const resetScroll = () => scrollTo(0, 0);
    requestAnimationFrame(resetScroll);
    addEventListener('load', () => setTimeout(resetScroll, 0), { once: true });
  }

  addEventListener('keydown', e => {
    const dialog = !$('#pk').hidden ? $('#pk') : !$('#cfg').hidden ? $('#cfg') : null;
    if (!dialog) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopImmediatePropagation();
      if (dialog.id === 'pk') pkClose();
      else cfgClose();
      return;
    }
    if (e.key !== 'Tab') return;
    const nodes = [...dialog.querySelectorAll('button,input,select,textarea,a[href]')].filter(el => !el.disabled && el.getClientRects().length);
    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
  }, true);
})();
