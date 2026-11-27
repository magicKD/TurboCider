'use strict';
(() => {
  // Single source of truth for PoseWear's 12 reference collections,
  // filter metadata, studio slot routing, accent tokens, and shared DOM/storage helpers.
  const COLLECTIONS = [
    {
      id: 'pose',
      prefix: 'p:',
      label: '姿势',
      title: '姿势库',
      file: 'poses.json',
      statKey: 'pose',
      slot: 'pose',
      accent: '#849dbb',
      filters: [['line', '系列'], ['cat', '分类'], ['ang', '角度']],
      suggestions: ['站姿', '坐姿', '回头', '双手抱头', '可爱', '躺卧', '比心', '跪立'],
      keywords: 'pose sketch line 姿态 ポーズ',
    },
    {
      id: 'depth',
      prefix: 'd:',
      label: '深度姿势',
      title: '深度姿势',
      file: 'depths.json',
      statKey: 'depth',
      slot: 'pose',
      accent: '#9689ae',
      filters: [['line', '系列'], ['cat', '分类'], ['ang', '角度']],
      suggestions: ['站姿', '坐姿', '上半身', '全身', '俯拍', '侧面', '可爱', '跪立'],
      keywords: 'depth 3d map 深度',
    },
    {
      id: 'costume',
      prefix: 'c:',
      label: '服装',
      title: '服装库',
      file: 'costumes.json',
      statKey: 'cos',
      slot: 'garment',
      accent: '#be929a',
      filters: [['cat', '分类'], ['col', '颜色'], ['item', '单品']],
      suggestions: ['连衣裙', '黑色', '泳装', '针织', '蕾丝', '冬季', '牛仔', '高跟鞋'],
      keywords: 'costume outfit dress 服装 穿搭 衣装',
    },
    {
      id: 'anime',
      prefix: 'a:',
      label: '动画服装',
      title: '动画服装库',
      file: 'anime.json',
      statKey: 'ani',
      slot: 'garment',
      accent: '#b08fad',
      filters: [['cat', '分类'], ['col', '颜色'], ['item', '单品']],
      suggestions: ['连衣裙', '黑色', '泳装', '针织', '蕾丝', '冬季', '牛仔', '制服'],
      keywords: 'anime costume outfit 动画 二次元 アニメ衣装',
    },
    {
      id: 'portrait',
      prefix: 'r:',
      label: '人像',
      title: '人像库',
      file: 'portraits.json',
      statKey: 'por',
      slot: 'portrait',
      accent: '#c49a82',
      filters: [['sec', '类型'], ['len', '发长'], ['st', '发型']],
      suggestions: ['短发', '波波', '狼尾', '刘海', '卷', '奶茶', '黑发', '正脸'],
      keywords: 'portrait face model 人像 人物',
    },
    {
      id: 'expr',
      prefix: 'e:',
      label: '表情',
      title: '表情库',
      file: 'expressions.json',
      statKey: 'exp',
      slot: 'expr',
      accent: '#a2778b',
      filters: [['emo', '情绪'], ['int', '强度']],
      suggestions: ['开心', '大笑', '愤怒', '害羞', '搞怪', '严肃'],
      keywords: 'expression emotion 表情',
    },
    {
      id: 'hair',
      prefix: 'h:',
      label: '发型',
      title: '发型库',
      file: 'hair.json',
      statKey: 'hair',
      slot: 'extra',
      roleEn: 'Hairstyle',
      accent: '#968071',
      filters: [['cat', '分类']],
      suggestions: ['长发', '双马尾', '刘海', '发饰', '瞳色'],
      keywords: 'hairstyle hair 发型 髪型',
    },
    {
      id: 'dyn',
      prefix: 'y:',
      label: '历代服饰',
      title: '历代服饰',
      file: 'dynasties.json',
      statKey: 'dyn',
      slot: 'garment',
      accent: '#ab7870',
      filters: [['era', '朝代'], ['ang', '角度']],
      suggestions: ['唐', '宋', '明', '曲裾', '齐胸'],
      keywords: 'dynasty historical hanfu 历代 汉服 時代衣装',
    },
    {
      id: 'scene',
      prefix: 's:',
      label: '场景',
      title: '场景库',
      file: 'scenes.json',
      statKey: 'scene',
      slot: 'extra',
      roleEn: 'Scene',
      accent: '#88a48a',
      filters: [['cat', '分类'], ['ang', '机位']],
      suggestions: ['客厅', '庭院', '街道', '海岸'],
      keywords: 'scene background 场景 背景',
    },
    {
      id: 'light',
      prefix: 'l:',
      label: '光影',
      title: '光影库',
      file: 'lights.json',
      statKey: 'light',
      slot: 'extra',
      roleEn: 'Lighting',
      accent: '#a2b6bf',
      filters: [['cat', '分类'], ['ang', '机位']],
      suggestions: ['伦勃朗', '窗光', '黄昏', '夜景'],
      keywords: 'lighting light 光影 布光 照明',
    },
    {
      id: 'prop',
      prefix: 't:',
      label: '道具',
      title: '道具库',
      file: 'props.json',
      statKey: 'prop',
      slot: 'extra',
      roleEn: 'Prop',
      accent: '#739e9c',
      filters: [['cat', '分类'], ['ang', '视角']],
      suggestions: ['吉他', '油纸伞', '灯笼', '相机'],
      keywords: 'prop item 道具 小道具',
    },
    {
      id: 'furn',
      prefix: 'f:',
      label: '家具',
      title: '家具库',
      file: 'furniture.json',
      statKey: 'furn',
      slot: 'extra',
      roleEn: 'Furniture',
      accent: '#95928e',
      filters: [['cat', '分类'], ['ang', '视角']],
      suggestions: ['餐椅', '沙发', '造型台', '床榻'],
      keywords: 'furniture 家具',
    },
  ];

  const UTILITY_TABS = [
    { id: 'studio', title: '工作台', cmdLabel: '创作工作台', keywords: 'studio tryon generate 工作台 试穿 生图 スタジオ' },
    { id: 'fav', title: '我的收藏', cmdLabel: '我的收藏', keywords: 'favorite saved 收藏 お気に入り' },
    { id: 'recent', title: '最近浏览', cmdLabel: '最近浏览', keywords: 'recent history 最近 履歴' },
    { id: 'history', title: '生成记录', cmdLabel: '生成记录', keywords: 'generation history 记录 生成履歴' },
  ];

  const TABS = [...COLLECTIONS.map(c => c.id), ...UTILITY_TABS.map(t => t.id)];
  const TNAME = Object.fromEntries([
    ...COLLECTIONS.map(c => [c.id, c.title]),
    ...UTILITY_TABS.map(t => [t.id, t.title]),
  ]);
  const PFX = Object.fromEntries(COLLECTIONS.map(c => [c.id, c.prefix]));
  const KT = Object.fromEntries(COLLECTIONS.map(c => [c.prefix[0], c.id]));
  const LABELS = Object.fromEntries(COLLECTIONS.map(c => [c.id, c.label]));
  const BOARD_LABELS = { ...LABELS, portrait: '人物' };
  const ACCENT = Object.fromEntries(COLLECTIONS.map(c => [c.id, c.accent]));
  const FDEF = {
    ...Object.fromEntries(COLLECTIONS.map(c => [c.id, c.filters])),
    fav: [['type', '类型']],
    recent: [['type', '类型']],
  };
  const SUGG = {
    ...Object.fromEntries(COLLECTIONS.map(c => [c.id, c.suggestions])),
    fav: [],
    recent: [],
  };
  const STUDIO_SLOT = Object.fromEntries(
    COLLECTIONS.filter(c => c.slot !== 'extra').map(c => [c.id, c.slot])
  );
  const GARMENT_TYPES = COLLECTIONS.filter(c => c.slot === 'garment').map(c => c.id);
  const OTHER_TYPES = COLLECTIONS.filter(c => c.slot === 'expr' || c.slot === 'extra').map(c => c.id);
  const MATERIAL_ROLES = Object.fromEntries(
    COLLECTIONS.filter(c => c.slot === 'extra').map(c => [c.id, [c.label, c.roleEn]])
  );

  const FLAB = {
    sec: '类型',
    len: '发长',
    st: '发型',
    line: '系列',
    cat: '分类',
    ang: '角度',
    col: '颜色',
    item: '单品',
    type: '类型',
    emo: '情绪',
    int: '强度',
    era: '朝代',
  };

  const COLS = {
    黑: 'black', 白: 'white', 红: 'red', 粉: 'pink', 蓝: 'blue', 藏青: 'navy',
    棕: 'brown', 米: 'beige', 象牙白: 'ivory', 灰: 'grey', 绿: 'green', 黄: 'yellow',
    紫: 'purple', 金: 'gold', 银: 'silver', 卡其: 'khaki', 酒红: 'burgundy', 橙: 'orange',
    奶油: 'cream', 薄荷: 'mint', 薰衣草: 'lavender', 橄榄: 'olive',
  };

  const COL_SWATCH = {
    黑: '#1c1c1e', 白: '#ffffff', 红: '#d92d46', 粉: '#f2a5b8', 蓝: '#3b7dd8', 藏青: '#223354',
    棕: '#80553a', 米: '#d8cab6', 象牙白: '#f5f1e6', 灰: '#8e8e93', 绿: '#478a5d', 黄: '#d8be6b',
    紫: '#8664b8', 金: '#b89e68', 银: '#c0c4ca', 卡其: '#b5a27e', 酒红: '#7c2237', 橙: '#df7a42',
    奶油: '#f3ebd8', 薄荷: '#8fd3be', 薰衣草: '#b8a5d6', 橄榄: '#687444',
  };

  const ITEMK = {
    连衣裙: 'dress', 半身裙: 'skirt', 短裤: 'shorts', 长裤: 'trousers', 牛仔裤: 'jeans',
    大衣: 'coat', 夹克: 'jacket', 开衫: 'cardigan', 衬衫: 'blouse', 毛衣: 'sweater',
    吊带: 'camisole', 文胸: 'bra', 泳衣: 'swimsuit', 靴子: 'boots', 高跟鞋: 'high heels',
    运动鞋: 'sneakers', 凉鞋: 'sandals', 连裤袜: 'tights', 长筒袜: 'stockings',
  };

  // Shared lightweight utilities (zero external dependencies)
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => root.querySelectorAll(sel);
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const readLS = key => { if (window.PeopleStudioNative) return PeopleStudioNative.get(key); try { return JSON.parse(localStorage[key] || 'null'); } catch { return null; } };
  const writeLS = (key, value) => { if (window.PeopleStudioNative) return PeopleStudioNative.set(key, value); try { localStorage[key] = JSON.stringify(value); } catch {} };
  const shuffle = list => {
    for (let i = list.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      [list[i], list[j]] = [list[j], list[i]];
    }
    return list;
  };
  const TL = item => LABELS[item.type];
  const studioSlot = type => STUDIO_SLOT[type] || 'extra';

  window.PoseWearCatalog = {
    COLLECTIONS,
    UTILITY_TABS,
    TABS,
    TNAME,
    PFX,
    KT,
    LABELS,
    BOARD_LABELS,
    ACCENT,
    FDEF,
    FLAB,
    SUGG,
    STUDIO_SLOT,
    GARMENT_TYPES,
    OTHER_TYPES,
    MATERIAL_ROLES,
    COLS,
    COL_SWATCH,
    ITEMK,
    TL,
    studioSlot,
    $,
    $$,
    esc,
    readLS,
    writeLS,
    shuffle,
  };
})();
