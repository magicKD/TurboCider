'use strict';
(() => {
  // Real-time color palette extractor for color reference images in the lightbox.
  // Samples the image into a 48x48 offscreen canvas with center-subject bias and
  // saturation weighting so garment/subject tones outrank neutral studio backdrops.
  function extract(img) {
    if (!img || !img.complete || !img.naturalWidth) return [];
    try {
      const size = 48;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, size, size);
      const data = ctx.getImageData(0, 0, size, size).data;

      const bucketsMap = new Map();
      for (let p = 0; p < size * size; p++) {
        const i = p * 4;
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];
        if (a < 200) continue;
        if (r > 242 && g > 242 && b > 242) continue;

        const x = p % size;
        const y = (p / size) | 0;
        const centerWeight = (x >= 7 && x <= 40 && y >= 4 && y <= 44) ? 1.8 : 0.25;
        const qr = Math.min(240, Math.round(r / 16) * 16);
        const qg = Math.min(240, Math.round(g / 16) * 16);
        const qb = Math.min(240, Math.round(b / 16) * 16);
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const sat = max ? (max - min) / max : 0;
        const weight = (0.35 + sat * 4.5) * centerWeight;
        const key = (qr << 16) | (qg << 8) | qb;

        const cur = bucketsMap.get(key);
        if (cur) {
          cur.w += weight;
          cur.r += r * weight;
          cur.g += g * weight;
          cur.b += b * weight;
        } else {
          bucketsMap.set(key, { w: weight, r: r * weight, g: g * weight, b: b * weight });
        }
      }

      const ranked = [...bucketsMap.values()]
        .sort((a, b) => b.w - a.w)
        .map(x => [Math.round(x.r / x.w), Math.round(x.g / x.w), Math.round(x.b / x.w)]);

      const chosen = [];
      for (const rgb of ranked) {
        if (chosen.every(o => Math.hypot(o[0] - rgb[0], o[1] - rgb[1], o[2] - rgb[2]) >= 44)) {
          chosen.push(rgb);
          if (chosen.length >= 5) break;
        }
      }

      return chosen.map(([r, g, b]) =>
        '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('').toUpperCase()
      );
    } catch {
      return [];
    }
  }

  function render(box, img, item, isActive) {
    if (!box || !item || item.type === 'pose' || item.type === 'depth' || item.ext) {
      if (box) box.hidden = true;
      return;
    }
    const paint = () => {
      if (!box.isConnected || !isActive() || img.classList.contains('lo')) return;
      const colors = extract(img);
      if (colors.length < 2) {
        box.hidden = true;
        return;
      }
      const chips = colors
        .map(hex => `<button type="button" class="pal-sw" data-hex="${hex}" style="--sw:${hex}" title="复制 ${hex}"><i></i><span>${hex}</span></button>`)
        .join('');
      box.innerHTML = `<div class="lb-pal-h"><small>提取配色</small><small class="mu">点击复制</small></div><div class="lb-pal-sw">${chips}</div>`;
      box.hidden = false;
    };

    if (img.complete && img.naturalWidth && !img.classList.contains('lo')) paint();
    else img.addEventListener('load', paint, { once: true });
  }

  window.PoseWearPalette = { extract, render };
})();
