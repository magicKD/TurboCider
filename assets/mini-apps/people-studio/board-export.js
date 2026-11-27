'use strict';
(() => {
  // Pure client-side HTML5 Canvas compositor for exporting the Studio reference board
  // as a crisp 2x editorial moodboard PNG matching the active light/dark theme.
  const C = window.PoseWearCatalog;
  if (!C) return;
  const { ACCENT, BOARD_LABELS } = C;

  function roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function clipText(ctx, str, maxW) {
    const s = String(str || '').replace(/\s+/g, ' ').trim();
    if (!s || ctx.measureText(s).width <= maxW) return s;
    let lo = 0;
    let hi = s.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ctx.measureText(s.slice(0, mid) + '…').width <= maxW) lo = mid;
      else hi = mid - 1;
    }
    return s.slice(0, lo) + '…';
  }

  function loadImage(ref, item) {
    return new Promise(resolve => {
      const img = new Image();
      let src = '';
      if (ref.blob || (ref.url && !item)) {
        src = ref.url;
      } else if (item && !item.ext) {
        src = item.angles
          ? (item.angles[ref.ai ?? item.ai]?.i || item.angles[ref.ai ?? item.ai]?.t)
          : (item.m || item.i || item.t);
      } else if (item?.ext) {
        img.crossOrigin = 'anonymous';
        if (item.ref) img.referrerPolicy = item.ref;
        src = item.m || item.t;
      }
      if (!src) return resolve(null);
      const done = ok => resolve(ok && img.naturalWidth ? img : null);
      img.onload = () => done(true);
      img.onerror = () => done(false);
      img.src = src;
    });
  }

  async function renderPng(chosen, { app, aspectRatio = 'auto', promptText = '' } = {}) {
    if (!chosen.length) return { blob: null, extSkipped: 0 };
    const t = s => (window.PWLocale ? window.PWLocale.t(s) : s);

    const entries = await Promise.all(
      chosen.map(async ([type, ref], idx) => {
        const item = app.ITEMS[ref.k];
        const material = item?.type || type;
        const title = item ? app.title(item) : (ref.name || t('本地上传'));
        const rawAngle = item?.angles?.[ref.ai ?? item.ai]?.zh || '';
        const img = await loadImage(ref, item);
        return {
          idx: idx + 1,
          material,
          cat: t(BOARD_LABELS[material] || material),
          title,
          angle: rawAngle ? t(rawAngle) : '',
          img,
          ext: !!item?.ext,
        };
      })
    );

    const dark = document.documentElement.dataset.theme === 'dark';
    const n = entries.length;
    const cols = n <= 3 ? n : n === 4 ? 2 : n <= 6 ? 3 : 4;
    const rows = Math.ceil(n / cols);
    const cardW = 360;
    const padImg = 12;
    const imgW = cardW - padImg * 2;
    const imgH = 448;
    const capH = 68;
    const cardH = padImg + imgH + capH;
    const gap = 28;
    const padX = 56;
    const headH = 96;
    const footH = 68;
    const width = Math.max(840, padX * 2 + cols * cardW + (cols - 1) * gap);
    const height = headH + rows * cardH + (rows - 1) * gap + footH;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const sans = '-apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans SC","Helvetica Neue",Arial,sans-serif';

    // Surface + subtle studio dot-grid
    ctx.fillStyle = dark ? '#141416' : '#f7f7f8';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.065)';
    for (let y = 24; y < height; y += 24) {
      for (let x = 24; x < width; x += 24) {
        ctx.beginPath();
        ctx.arc(x, y, 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Header brand mark + title + metadata
    const ac = dark ? '#ff3b57' : '#ff2442';
    const ink = dark ? '#f2f2f2' : '#111111';
    const mu = dark ? '#9a9a9e' : '#666666';
    const cardBg = dark ? '#1e1e22' : '#ffffff';
    const lineCol = dark ? '#323236' : '#e4e4e7';

    ctx.fillStyle = ac;
    ctx.beginPath();
    ctx.arc(padX + 14, 35, 4.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = ac;
    ctx.lineWidth = 3.2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(padX + 14, 42);
    ctx.lineTo(padX + 3, 55);
    ctx.lineTo(padX + 25, 55);
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = ink;
    ctx.font = `700 22px ${sans}`;
    ctx.textBaseline = 'middle';
    ctx.fillText('PoseWear', padX + 36, 46);
    const brandW = ctx.measureText('PoseWear').width;
    ctx.fillStyle = mu;
    ctx.font = `500 16px ${sans}`;
    ctx.fillText('·  ' + t('参考画板'), padX + 48 + brandW, 47);

    const now = new Date();
    const dateStamp = `${now.getFullYear()}.${String(now.getMonth() + 1).padStart(2, '0')}.${String(now.getDate()).padStart(2, '0')}`;
    const unit = document.documentElement.lang === 'en' && n === 1 ? 'image' : t('张');
    const metaParts = [`${n} ${unit}`, aspectRatio !== 'auto' ? aspectRatio : '', dateStamp].filter(Boolean).join('  ·  ');
    ctx.textAlign = 'right';
    ctx.font = `500 14px ${sans}`;
    ctx.fillStyle = mu;
    ctx.fillText(metaParts, width - padX, 47);
    ctx.textAlign = 'left';

    // Cards
    let extSkipped = 0;
    entries.forEach((entry, i) => {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const rowCount = row === rows - 1 ? n - row * cols : cols;
      const rowW = rowCount * cardW + (rowCount - 1) * gap;
      const startX = (width - rowW) / 2;
      const x = startX + col * (cardW + gap);
      const y = headH + row * (cardH + gap);
      const edge = ACCENT[entry.material] || lineCol;

      ctx.save();
      ctx.shadowColor = dark ? 'rgba(0,0,0,0.45)' : 'rgba(0,0,0,0.08)';
      ctx.shadowBlur = 22;
      ctx.shadowOffsetY = 8;
      roundRectPath(ctx, x, y, cardW, cardH, 18);
      ctx.fillStyle = cardBg;
      ctx.fill();
      ctx.restore();

      roundRectPath(ctx, x, y, cardW, cardH, 18);
      ctx.strokeStyle = edge + '77';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const ix = x + padImg;
      const iy = y + padImg;
      ctx.save();
      roundRectPath(ctx, ix, iy, imgW, imgH, 12);
      ctx.clip();
      ctx.fillStyle = entry.material === 'depth'
        ? '#141414'
        : entry.material === 'pose'
          ? '#f4f4f4'
          : (dark ? '#26262a' : '#f3f3f5');
      ctx.fillRect(ix, iy, imgW, imgH);

      if (entry.img) {
        const pad = entry.material === 'pose' ? 16 : 0;
        const aw = imgW - pad * 2;
        const ah = imgH - pad * 2;
        const scale = Math.min(aw / entry.img.naturalWidth, ah / entry.img.naturalHeight);
        const dw = entry.img.naturalWidth * scale;
        const dh = entry.img.naturalHeight * scale;
        if (entry.material === 'pose') ctx.globalCompositeOperation = 'multiply';
        ctx.drawImage(entry.img, ix + pad + (aw - dw) / 2, iy + pad + (ah - dh) / 2, dw, dh);
        ctx.globalCompositeOperation = 'source-over';
      } else {
        if (entry.ext) extSkipped++;
        ctx.fillStyle = mu;
        ctx.font = `600 15px ${sans}`;
        ctx.textAlign = 'center';
        ctx.fillText(entry.cat, ix + imgW / 2, iy + imgH / 2 - 16);
        ctx.font = `400 13px ${sans}`;
        ctx.fillText(clipText(ctx, entry.title, imgW - 36), ix + imgW / 2, iy + imgH / 2 + 8);
        if (entry.ext) {
          ctx.font = `400 12px ${sans}`;
          ctx.fillText(t('在原站查看'), ix + imgW / 2, iy + imgH / 2 + 32);
        }
        ctx.textAlign = 'left';
      }
      ctx.restore();

      // Send-order badge
      roundRectPath(ctx, ix + 10, iy + 10, 28, 28, 8);
      ctx.fillStyle = 'rgba(17,17,17,0.82)';
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = `700 14px ${sans}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(entry.idx), ix + 24, iy + 24);
      ctx.textAlign = 'left';

      // Camera angle pill
      if (entry.angle) {
        ctx.font = `600 12px ${sans}`;
        const pillW = ctx.measureText(entry.angle).width + 18;
        roundRectPath(ctx, ix + imgW - pillW - 10, iy + imgH - 32, pillW, 22, 11);
        ctx.fillStyle = 'rgba(17,17,17,0.78)';
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.fillText(entry.angle, ix + imgW - pillW / 2 - 10, iy + imgH - 21);
        ctx.textAlign = 'left';
      }

      // Caption footer
      const cy = iy + imgH + 22;
      ctx.fillStyle = edge;
      ctx.beginPath();
      ctx.arc(x + 20, cy, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = mu;
      ctx.font = `600 13px ${sans}`;
      ctx.fillText(entry.cat, x + 32, cy);
      ctx.fillStyle = ink;
      ctx.font = `600 15px ${sans}`;
      ctx.fillText(clipText(ctx, entry.title, cardW - 36), x + 18, cy + 24);
    });

    // Footer colophon
    const fy = height - footH + 24;
    ctx.strokeStyle = lineCol;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padX, fy - 14);
    ctx.lineTo(width - padX, fy - 14);
    ctx.stroke();

    const promptSnippet = clipText(ctx, promptText, width - padX * 2 - 190);
    ctx.fillStyle = mu;
    ctx.font = `400 13px ${sans}`;
    if (promptSnippet) ctx.fillText(promptSnippet, padX, fy + 12);
    ctx.textAlign = 'right';
    ctx.font = `600 13px ${sans}`;
    ctx.fillText('posewear.pages.dev', width - padX, fy + 12);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    return { blob, extSkipped, now };
  }

  window.PoseWearBoardExport = { renderPng };
})();
