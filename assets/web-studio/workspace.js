'use strict';
(() => {
  const { $, terminal, states, operations, phases, api, node, icon, button, link, toast, action, createID, preference,
    upload, getSession } = TC;
  const base = '/api/studio/';
  const studio = (path, body) => api(base + path, body);
  const host = (name, payload = {}) => studio('host', { action: name, ...payload });
  const labels = {
    create: ['创作', '从文字、参考图或首尾帧开始。'], upscale: ['超分', '提升图片清晰度，或恢复视频细节。']
  };
  const presets = [['正方形', 512, 512], ['竖图', 352, 704], ['横图', 704, 352],
    ['长竖图', 384, 768], ['宽横图', 768, 384]];
  const amberPrompt = '暖橘色摄影棚里，琥珀色竖纹玻璃圆盘斜倚在浅杏色石台上，前方是一颗透明玻璃球。午后硬光从左上方穿过玻璃，在台面投下条纹与长阴影。低机位静物摄影，轻微胶片颗粒。';
  const imagePrep = [['original', '恢复原图'], ['automatic', '自动缩小'], ['fit512', '适合 512×512'],
    ['portrait512', '适合 512×768'], ['landscape512', '适合 768×512']];
  const mediaOperations = new Set(['image.edit', 'image.transform', 'image.upscale', 'video.image',
    'video.keyframes', 'video.reference', 'video.upscale']);
  const controls = {
    residency: [['', '模型默认'], ['resident', '常驻内存'], ['component_staged', '分阶段加载'], ['streamed', 'Streaming']],
    execution: [['gpu', 'GPU'], ['auto', '自动加速'], ['gpu_ane', 'GPU ＋ ANE']],
    reference: [['budget', '共享 512² 像素预算'], ['fast', '快速 512 编码'], ['standard', '标准编码']],
    ltxBackend: [['auto', '自动'], ['c_metal', 'C / Metal'], ['cpp_mlx', 'C++ / MLX']],
    ltxSpeed: [['quality', '画质优先'], ['sol', 'Sol 近似'], ['fast_approx', '快速近似']]
  };
  let catalog, apps = [], jobs = [], library = { uploads: [], outputs: [] }, devices;
  let drafts = {}, page = 'create', templateID, activeDraft, form, previewID, previewKey;
  let jobFilter = 'all', libraryFilter = 'all', jobScope = 'own', libraryScope = 'own', jobKey;
  let timer, pollBusy = false, busyToken, catalogPromise, catalogWorkspace, scopeVersion = 0;

  function setChildren(target, children) { target.replaceChildren(...children.filter(Boolean)); }
  function message(text) { return node('p', text, 'empty'); }
  function emptyState(symbol, title, detail, control) {
    const box = node('div', null, 'empty-state');
    const mark = node('span', null, 'empty-mark'); mark.append(icon(symbol));
    box.append(mark, node('strong', title));
    if (detail) box.append(node('p', detail));
    if (control) box.append(control);
    return box;
  }
  function picture(url, description = '') {
    const img = node('img'); img.src = url; img.alt = description; img.decoding = 'async';
    return img;
  }
  function actions(...items) {
    const row = node('div', null, 'actions');
    setChildren(row, items);
    return row;
  }
  function heading(title, detail, ...items) {
    const row = node('div', null, 'page-heading');
    const copy = node('div');
    copy.append(node('h1', title));
    if (detail) copy.append(node('p', detail));
    row.append(copy);
    if (items.length) row.append(actions(...items));
    return row;
  }
  function buildFileInput(title, accept, onSelect, multiple = false) {
    const label = node('label', title, 'file-button');
    const input = node('input');
    input.type = 'file'; input.accept = accept; input.multiple = multiple;
    input.setAttribute('aria-label', title);
    input.addEventListener('change', () => {
      const files = [...input.files]; input.value = '';
      if (files.length) Promise.resolve(onSelect(files)).catch(error => toast(error.message, true));
    });
    label.prepend(icon('upload'));
    label.append(input);
    return label;
  }
  function select(name, entries, current, change) {
    const input = node('select');
    input.name = name;
    for (const [value, label, disabled] of entries) {
      const item = node('option', label);
      item.value = value; item.disabled = !!disabled; input.append(item);
    }
    if (current !== undefined && [...input.options].some(item => item.value === String(current))) input.value = current;
    if (change) input.addEventListener('change', change);
    return input;
  }
  function field(label, input, full = false, hint) {
    const wrapper = node('label', label, full ? 'field full' : 'field');
    wrapper.append(input);
    if (hint) wrapper.append(node('small', hint));
    return wrapper;
  }
  function fieldSelect(label, name, entries, current, full = false, change, hint) {
    return field(label, select(name, entries, current, change), full, hint);
  }
  function fieldNumber(label, name, value, min, max, step = 1) {
    const input = node('input');
    input.type = 'number'; input.name = name; input.value = value ?? ''; input.min = min; input.max = max; input.step = step;
    return field(label, input);
  }
  function check(label, name, state) {
    const wrapper = node('label', null, 'check');
    const input = node('input');
    input.type = 'checkbox'; input.name = name; input.checked = !!state;
    wrapper.append(input, node('span', label));
    return wrapper;
  }
  function setTabs(target, options, current, use) {
    const focused = target.contains(document.activeElement) ? document.activeElement.dataset.filter : null;
    target.setAttribute('role', 'group');
    setChildren(target, options.map(([id, title, inactive, count]) => {
      const item = button(title, () => use(id));
      item.dataset.filter = id;
      item.setAttribute('aria-pressed', id === current);
      item.disabled = !!inactive;
      if (count !== undefined) item.append(node('small', String(count), 'count'));
      return item;
    }));
    if (focused) [...target.children].find(item => item.dataset.filter === focused)?.focus({ preventScroll: true });
  }
  function selectedTemplate() { return catalog?.templates.find(entry => entry.id === templateID); }
  function models() {
    if (!catalog) return [];
    if (page === 'template') return catalog.models.filter(model => model.id === 'qwen-image-2.1');
    const upside = page === 'upscale';
    return catalog.models.filter(model => model.operations.some(operation => operation.endsWith('.upscale') === upside));
  }
  function modelOperations(model) {
    if (!model) return [];
    return model.operations.filter(value => value.endsWith('.upscale') === (page === 'upscale'));
  }
  function key() { return page === 'template' ? 'template/' + templateID : page; }
  function saveDraft() {
    const workspace = getSession()?.workspace;
    if (workspace) preference('turbocider.drafts.' + workspace, drafts);
  }
  function draftDefaults(model, installation, operation) {
    return { model: model?.id || '', installation: installation?.id || '', operation: operation || modelOperations(model)[0] || 'image.generate',
      prompt: page === 'template' ? selectedTemplate()?.instruction || '' : '',
      refs: [], roles: {}, params: { width: model?.width || 512, height: model?.height || 512, steps: installation?.steps || model?.steps || 1,
        frames: model?.frames || 1, fps: model?.fps || 1, audio: model?.defaultAudio || false, dynamicText: true,
        qwenReference: installation?.turbo ? 'budget' : 'standard', expansion: 1.5 } };
  }
  function bindDraft() {
    const options = models();
    const existing = drafts[key()];
    const model = options.find(item => item.id === existing?.model)
      || options.find(item => item.id === catalog.preferred && item.installations.length)
      || options.find(item => item.operations.includes('image.generate') && item.installations.length)
      || options.find(item => item.installations.length) || options[0];
    const install = model?.installations.find(item => item.id === existing?.installation) || model?.installations[0];
    const draft = existing || draftDefaults(model, install);
    draft.model = model?.id ?? ''; draft.installation = install?.id ?? '';
    if (!modelOperations(model).includes(draft.operation) && page !== 'template') draft.operation = modelOperations(model)[0] || '';
    drafts[key()] = activeDraft = draft;
    return [model, install];
  }
  function switchModel(model, installation) {
    if (!model) return;
    const next = draftDefaults(model, installation || model.installations[0]);
    next.prompt = activeDraft.prompt;
    if (modelOperations(model).includes(activeDraft.operation)) next.operation = activeDraft.operation;
    const input = next.operation === 'video.upscale' ? 'video' : 'image';
    next.refs = activeDraft.refs.filter(asset => asset.kind === input);
    next.roles = activeDraft.roles;
    drafts[key()] = next;
    previewID = ''; saveDraft(); renderComposer();
  }
  function updateForm() {
    if (!form || !form.isConnected || form.dataset.draft !== key()) return;
    for (const input of form.elements) {
      if (!input.name || input.disabled || !['INPUT', 'SELECT', 'TEXTAREA'].includes(input.tagName) || input.type === 'file') continue;
      if (input.name === 'prompt') { activeDraft.prompt = input.value; continue; }
      if (input.name.startsWith('_')) continue;
      const value = input.type === 'checkbox' ? input.checked : input.type === 'number' ?
        (input.value === '' ? undefined : Number(input.value)) : input.value || undefined;
      activeDraft.params[input.name] = value;
    }
    saveDraft();
  }
  function maxReferences(model, operation) {
    if (operation === 'video.keyframes') return 2;
    if (['image.edit', 'video.reference'].includes(operation)) return Math.min(10, model?.maxImages ?? 10);
    return 1;
  }
  function renderComposer() {
    if (!catalog || !['create', 'upscale', 'template'].includes(page)) return;
    if (page === 'template' && !selectedTemplate()) { location.hash = '#apps'; return; }
    const target = $(page);
    const [model, installation] = bindDraft();
    const draft = activeDraft;
    const template = selectedTemplate();
    const isTemplate = page === 'template', upside = page === 'upscale', video = draft.operation?.startsWith('video.');
    const advancedOpen = !!target.querySelector('.advanced[open]');
    const [title, detail] = isTemplate ? [template.name, template.detail] : labels[page];
    const navAction = isTemplate
      ? link('所有应用', '#apps', 'button')
      : button('新建' + (upside ? '超分' : '创作'), () => {
        drafts[key()] = draftDefaults(model, installation); previewID = ''; saveDraft(); renderComposer();
      });
    navAction.prepend(icon(isTemplate ? 'apps' : 'plus'));
    const header = heading(title, detail, navAction);
    const grid = node('div', null, 'composer-grid'), panel = node('section', null, 'card canvas');
    panel.setAttribute('aria-label', '预览');
    panel.dataset.preview = page;
    grid.append(panel);
    form = node('form', null, 'card composer');
    form.dataset.draft = key();
    const setup = node('div', null, 'form-grid model-fields');
    const available = models().map(item => [item.id, item.name + (item.installations.length ? '' : ' · 未安装')]);
    setup.append(fieldSelect('模型', '_model', available, model?.id, true, event => {
      switchModel(models().find(item => item.id === event.target.value));
    }));
    if (model?.installations.length) {
      setup.append(fieldSelect('模型安装', '_installation', model.installations.map(item =>
        [item.id, item.turbo ? '8 步 Turbo · ' + item.name : item.name]), installation?.id, true, event => {
        switchModel(model, model.installations.find(item => item.id === event.target.value));
      }));
    } else setup.append(link('前往模型中心', '#models', 'button'));
    if (!isTemplate) {
      setup.append(fieldSelect('工作流', '_operation', modelOperations(model).map(value => [value, operations[value] || value]),
        draft.operation, true, event => {
          draft.operation = event.target.value;
          previewID = null;
          if (model?.id === 'relayvsr-2x' && draft.operation === 'video.upscale') draft.params.audio = true;
          if (draft.operation !== 'video.upscale') draft.refs = draft.refs.filter(item => item.kind === 'image');
          else draft.refs = draft.refs.filter(item => item.kind === 'video');
          if (draft.operation === 'image.edit' && draft.params.matchAspect === undefined) draft.params.matchAspect = true;
          if (draft.operation === 'video.keyframes') {
            draft.params.strength = 1;
            draft.params.ltxBackend = 'c_metal';
            draft.params.ltxSpeed = 'quality';
            draft.params.streaming = 'off';
          }
          saveDraft(); renderComposer();
        }));
    }
    if (isTemplate || !upside) {
      const input = node('textarea');
      input.name = 'prompt'; input.rows = 3; input.value = draft.prompt; input.maxLength = 6000;
      input.placeholder = isTemplate ? '描述附加要求' : '描述想生成的画面，或希望如何修改参考图…';
      const promptField = field(isTemplate ? '指令' : '提示词', input, true);
      promptField.classList.add('prompt');
      form.append(promptField);
    }
    form.append(setup);
    if (isTemplate) {
      const groups = node('div', null, 'input-zone');
      for (const role of template.roles) {
        groups.append(renderInputs(draft, model, role.title + (role.required ? ' · 必填' : ' · 可选'), 'image', role.id));
      }
      form.append(groups);
    } else if (mediaOperations.has(draft.operation)) {
      form.append(renderInputs(draft, model, upside ? '源素材' : draft.operation === 'video.keyframes' ? '首帧与尾帧' : '参考素材',
        draft.operation === 'video.upscale' ? 'video' : 'image'));
    }
    if (!model?.coreML && model?.id !== 'relayvsr-2x') form.append(renderShape(draft, model, installation, video));
    if (model?.id === 'relayvsr-2x' && video) form.append(check('保留原音轨', 'audio', draft.params.audio ?? true));
    const advanced = node('details', null, 'advanced'); advanced.open = advancedOpen;
    const advancedTitle = node('summary', '高级参数');
    advancedTitle.prepend(icon('settings')); advancedTitle.append(icon('chevron'));
    advanced.append(advancedTitle, renderAdvanced(draft, model, installation, video));
    form.append(advanced);
    const run = node('div', null, 'run-row');
    const submit = node('button', isTemplate ? '运行模板 ↗' : upside ? '提交超分 ↗' : '开始生成 ↗', 'primary');
    submit.type = 'submit'; submit.disabled = !installation || busyToken?.draft === key();
    const resultMessage = node('p', null, 'form-message'); resultMessage.setAttribute('role', 'status'); resultMessage.hidden = true;
    run.append(submit, link('查看队列', '#jobs', 'button'));
    form.append(run, resultMessage);
    form.addEventListener('input', event => {
      updateForm();
      if (['width', 'height', 'matchAspect'].includes(event.target.name)) syncShapes();
    });
    form.addEventListener('submit', event => { event.preventDefault(); submitDraft(event.currentTarget, submit, resultMessage); });
    grid.append(form);
    target.replaceChildren(header, grid);
    syncShapes();
    previewKey = null;
    renderPreview();
  }

  function renderShape(draft, model, installation, video) {
    const parameters = node('div', null, 'form-grid');
    if (page === 'upscale') {
      parameters.append(fieldSelect('输出尺寸', 'seedVR2Size',
        [['', '手动设置'], ['scale2', '2×'], ['scale4', '4×'], ['long512', '长边 512'], ['long1024', '长边 1024']],
        draft.params.seedVR2Size || 'scale2', true));
    } else {
      const shapes = node('div', null, 'shape-options');
      shapes.setAttribute('aria-label', '输出画幅');
      shapes.setAttribute('role', 'group');
      const options = video ? [['正方形', 512, 512], ['竖屏', 384, 640], ['横屏', 640, 384], ['宽屏', 768, 448]] : presets;
      for (const [name, width, height] of options) {
        const choice = button(name, () => {
          draft.params.width = width; draft.params.height = height; draft.params.matchAspect = false;
          form.elements.width.value = width; form.elements.height.value = height;
          if (form.elements.matchAspect) form.elements.matchAspect.checked = false;
          updateForm(); syncShapes();
        });
        choice.dataset.width = width; choice.dataset.height = height;
        const frame = node('span', null, 'shape-glyph');
        frame.style.setProperty('--w', Math.round(16 * Math.min(1, width / height)) + 'px');
        frame.style.setProperty('--h', Math.round(16 * Math.min(1, height / width)) + 'px');
        frame.setAttribute('aria-hidden', 'true'); choice.prepend(frame);
        choice.title = width + ' × ' + height;
        shapes.append(choice);
      }
      const wrapper = node('div', null, 'field full'); wrapper.append(node('strong', '输出画幅'), shapes); parameters.append(wrapper);
    }
    parameters.append(fieldNumber('宽度（px）', 'width', draft.params.width, 64, model?.id === 'qwen-image-2.1' ? 4096 : 2048),
      fieldNumber('高度（px）', 'height', draft.params.height, 64, model?.id === 'qwen-image-2.1' ? 4096 : 2048));
    if (page !== 'upscale' && (mediaOperations.has(draft.operation) || page === 'template')) {
      parameters.append(check('按第一张参考图匹配比例', 'matchAspect', draft.params.matchAspect));
    }
    if (video && page !== 'upscale') {
      parameters.append(fieldNumber('帧数', 'frames', draft.params.frames, 1, 4097),
        fieldNumber('帧率', 'fps', draft.params.fps, 1, 120));
    }
    if (video && model?.audio) parameters.append(check('输出音频', 'audio', draft.params.audio));
    return parameters;
  }

  function syncShapes() {
    if (!form) return;
    const { width, height, matchAspect } = form.elements;
    for (const choice of form.querySelectorAll('.shape-options button')) {
      const selected = !matchAspect?.checked && choice.dataset.width === width?.value && choice.dataset.height === height?.value;
      choice.setAttribute('aria-pressed', String(selected));
    }
  }

  function renderAdvanced(draft, model, installation, video) {
    const grid = node('div', null, 'form-grid');
    const params = draft.params;
    const selectRow = (title, name, options, initial = '', full = false) => fieldSelect(title, name, options, params[name] ?? initial, full);
    if (model?.coreML) {
      grid.append(selectRow('计算设备', 'execution', [['gpu', 'GPU'], ['ane', 'ANE 优先']], 'gpu', true));
      return grid;
    }
    if (page !== 'upscale') {
      const steps = fieldNumber('采样步数', 'steps', params.steps, 1, 50);
      if (installation?.turbo || model?.id === 'ltx-2.5-distilled') steps.lastElementChild.readOnly = true;
      grid.append(steps, fieldNumber('种子（留空为随机）', 'seed', params.seed, 0, 2147483647));
      if (['image.transform', 'video.image', 'video.keyframes'].includes(draft.operation)) {
        grid.append(fieldNumber('输入强度', 'strength', params.strength, 0, 1, .05));
      }
      grid.append(check('动态文本长度', 'dynamicText', params.dynamicText));
    }
    if (model?.id !== 'relayvsr-2x') grid.append(selectRow('权重加载方式', 'residency', controls.residency));
    if (model?.supportsANE) {
      grid.append(selectRow('计算设备', 'execution', controls.execution, 'gpu'),
        selectRow('ANE 分区', 'anePartition', [['', '自动选择'], ...catalog.ane.filter(item => item.model === model.id).map(item => [item.id, item.name])]));
    }
    if (model?.id?.startsWith('flux2-')) grid.append(check('编译 GPU 计算块', 'compileGPU', params.compileGPU));
    if (['flux2-klein-4b', 'z-image-turbo', 'ltx-2.5-distilled'].includes(model?.id)) {
      grid.append(selectRow('公共流式内存档位', 'streaming', [['off', '关闭'], ...[8, 10, 12, 16, 20].map(value => ['tier' + value, value + ' GiB'])], 'off'));
    }
    if (model?.id === 'z-image-turbo') grid.append(selectRow('Z-Image 流式预算', 'zImageBudget',
      [6, 8, 10, 12].map(value => [value, value + ' GiB']), 10));
    if (model?.id === 'qwen-image-2.1') {
      if (draft.operation === 'image.edit' || page === 'template') {
        grid.append(selectRow('参考图编码', 'qwenReference', controls.reference.filter(([value]) => value !== 'budget' || installation?.turbo),
          installation?.turbo ? 'budget' : 'standard', true));
      }
      if (!installation?.turbo && !installation?.convrot) {
        grid.append(selectRow('DiT 缓存', 'qwenCache',
          [['off', '关闭'], ['conservative', '保守'], ['balanced', '均衡'], ['fast', '快速']], 'off'),
          check('增强提示词', 'promptEnhance', params.promptEnhance), check('允许增强编辑提示词', 'promptEnhanceEdit', params.promptEnhanceEdit),
          check('DiT W8A8 离线缓存', 'qwenDiTW8A8', params.qwenDiTW8A8), check('DiT SmoothQuant ＋ ConvRot', 'qwenSmoothQuant', params.qwenSmoothQuant));
      }
      if (installation?.convrot) {
        grid.append(selectRow('流式权重缓存预算', 'qwenWeightCache',
          [['minimum', '按层读取'], ...[1, 2, 4, 6, 8].map(value => ['gib' + value, value + ' GiB']), ['custom', '自定义 MiB']], 'minimum'),
          fieldNumber('自定义预算（MiB）', 'qwenWeightCacheMiB', params.qwenWeightCacheMiB, 1024, 16384));
      }
    }
    if (model?.id === 'ltx-2.5-distilled') {
      grid.append(selectRow('LTX 后端', 'ltxBackend', controls.ltxBackend, draft.operation === 'video.keyframes' ? 'c_metal' : 'auto'),
        selectRow('视频 VAE', 'ltxVAE', [['conv', '原版'], ['prunavaed', 'PrunaVAED', !installation?.pruna]], 'conv'),
        selectRow('LTX 加速', 'ltxSpeed', controls.ltxSpeed, 'quality'),
        check('Fast AV', 'ltxFastAV', params.ltxFastAV ?? true), check('批量视频注意力', 'ltxBatch', params.ltxBatch));
    }
    if (model?.id === 'seedvr2-3b') {
      const vae = draft.params.seedVR2VAE || {};
      const track = (control, name, convert = value => value) => {
        const input = control.querySelector('input, select');
        input.addEventListener('change', () => {
          const value = input.type === 'checkbox' ? input.checked : input.value;
          draft.params.seedVR2VAE = { ...draft.params.seedVR2VAE, [name]: convert(value) };
          saveDraft();
        });
        grid.append(control);
      };
      const vaeSelect = (label, id, options, fallback) => track(
        fieldSelect(label, '_' + id, options, vae[id] ?? fallback), id);
      if (video) {
        const mode = [['buffered', '缓冲'], ['streaming', '流式逐帧']];
        vaeSelect('VAE 编码', 'encoder', mode, 'buffered');
        vaeSelect('VAE 解码', 'decoder', mode, 'buffered');
      }
      vaeSelect('VAE 卷积', 'convolution',
        [['singleton2d', 'Singleton 2D'], ['conv3d', '原生 3D'], ['mixed3d', '混合 3D'], ['tiled3d', '瓦片 3D']], 'singleton2d');
      vaeSelect('时序历史缓存', 'history', [['compact', '紧凑'], ['frames', '帧缓存']], 'compact');
      const budget = vae.allocator_cache_bytes ? vae.allocator_cache_bytes / 1024 ** 3 : 4;
      track(fieldNumber('VAE 分配缓存（GiB）', '_allocatorCache', budget, 0, 16), 'allocator_cache_bytes',
        value => Number(value) * 1024 ** 3);
    }
    if (page !== 'upscale' && !video) {
      grid.append(check('Tiny VAE 编码', 'tinyEncoder', params.tinyEncoder), check('Tiny VAE 解码', 'tinyDecoder', params.tinyDecoder),
        check('实时采样预览', 'tinyPreview', params.tinyPreview), fieldNumber('预览步间隔', 'tinyPreviewInterval', params.tinyPreviewInterval, 1, 16));
      if (page !== 'template') {
        const upscalers = catalog.models.filter(item => item.coreML).flatMap(item => item.installations.map(entry => [entry.id, item.name]));
        if (upscalers.length) grid.append(selectRow('生成后自动超分', 'autoUpscale', [['', '关闭'], ...upscalers], '', true));
      }
    }
    if (page === 'template' && templateID === 'outpaint') grid.append(selectRow('扩图倍率', 'expansion', [[1.25, '1.25×'], [1.5, '1.5×'], [2, '2×']], 1.5));
    const loras = installation?.loras || [];
    if (loras.length) {
      grid.append(selectRow('LoRA 应用策略', 'loraStrategy', [['auto', '自动'], ...model.strategies.filter(value => value !== 'auto').map(value =>
        [value, ({ inference_time: '运行时加载', disk_premerge: '预合并到磁盘', in_memory_merge: '内存合并' })[value] || value])], 'auto', true));
      const picked = draft.params.adapters || [];
      for (const lora of loras) {
        const previous = picked.find(row => row.id === lora.id);
        const toggle = check(lora.name, '_lora', !!previous);
        const strength = fieldNumber('强度', '_strength', previous?.strength ?? lora.strength, -8, 8, .05);
        const checked = toggle.querySelector('input'), value = strength.querySelector('input');
        value.disabled = !previous;
        const change = () => {
          value.disabled = !checked.checked;
          const next = (draft.params.adapters || []).filter(row => row.id !== lora.id);
          if (checked.checked) next.push({ id: lora.id, strength: Number(value.value) });
          draft.params.adapters = next; saveDraft();
        };
        checked.addEventListener('change', change); value.addEventListener('input', change);
        grid.append(toggle, strength);
      }
    }
    return grid;
  }

  function renderInputs(draft, model, name, kind, role) {
    const group = node('div', null, 'input-zone'), box = node('div', null, 'drop-zone'), list = node('div', null, 'reference-list');
    const refs = role ? (draft.roles[role] ? [draft.roles[role]] : []) : draft.refs;
    const limit = role ? 1 : maxReferences(model, draft.operation);
    const session = getSession()?.workspace;
    const targetPage = key();
    const add = async (files, existing) => {
      if (refs.length + (existing ? 1 : files.length) > limit && limit !== 1) throw new Error('此工作流最多接受 ' + limit + ' 个素材。');
      const items = [];
      if (existing) items.push(existing);
      else for (const file of limit === 1 ? files.slice(0, 1) : files) items.push(await upload(file, kind));
      if (getSession()?.workspace !== session || drafts[targetPage] !== draft) return;
      if (role) draft.roles[role] = items[0];
      else draft.refs = limit === 1 ? items : [...draft.refs, ...items].filter((item, index, all) => all.findIndex(row => row.id === item.id) === index).slice(0, limit);
      saveDraft();
      if (targetPage === key()) { previewID = ''; renderComposer(); }
    };
    const mediaType = kind === 'video' ? '.mp4,.mov,.m4v,video/mp4,video/quicktime' : 'image/*,.heic,.heif';
    box.append(node('strong', name), actions(buildFileInput(kind === 'video' ? '上传视频' : '上传图片', mediaType, add, limit > 1),
      button('从素材库选', action(() => pickMedia(kind, item => add([], item))))));
    box.addEventListener('dragover', event => { event.preventDefault(); box.classList.add('is-over'); });
    box.addEventListener('dragleave', () => box.classList.remove('is-over'));
    box.addEventListener('drop', async event => {
      event.preventDefault(); box.classList.remove('is-over');
      try { await add([...event.dataTransfer.files]); } catch (error) { toast(error.message, true); }
    });
    if (draft.operation === 'video.keyframes' && !role) box.append(node('small', '按顺序添加首帧和尾帧。可用上下箭头调换。'));
    group.append(box);
    for (const [index, ref] of refs.entries()) {
      const entry = node('div', null, 'reference'), info = node('div');
      const ordinal = draft.operation === 'video.keyframes' ? (index === 0 ? '首帧' : '尾帧') : (limit > 1 ? '参考图 ' + (index + 1) : '');
      info.append(node('strong', ref.name, 'ref-name'), node('small', [ordinal, sizeLabel(ref)].filter(Boolean).join(' · ')));
      const thumb = ref.kind === 'image' ? mediaElement(ref, true) : node('span', '▷', 'media-icon');
      entry.append(thumb, info);
      const swap = offset => {
        const at = index + offset;
        [draft.refs[index], draft.refs[at]] = [draft.refs[at], draft.refs[index]]; saveDraft(); renderComposer();
      };
      const up = button('↑', () => swap(-1)), down = button('↓', () => swap(1));
      up.setAttribute('aria-label', '向前移动参考图'); down.setAttribute('aria-label', '向后移动参考图');
      up.disabled = index === 0; down.disabled = index === refs.length - 1;
      const replace = updated => {
        if (role) draft.roles[role] = updated;
        else draft.refs.splice(index, 1, updated);
        saveDraft(); renderComposer();
      };
      const prepare = ref.kind === 'image' ? select('_prepare', [['', '缩放参考图'], ...imagePrep], '', action(async event => {
        const preset = event.target.value;
        if (preset) replace(await studio('prepare', { id: ref.id, preset }));
      })) : null;
      prepare?.setAttribute('aria-label', '处理参考图片尺寸');
      entry.append(actions(button('预览', () => inspect(ref)), !role && limit > 1 ? up : null, !role && limit > 1 ? down : null, prepare,
        ref.kind === 'image' && model?.id === 'qwen-image-2.1' ? button('标注', () => annotate(ref, replace)) : null,
        button('移除', () => {
          if (role) delete draft.roles[role]; else draft.refs.splice(index, 1);
          saveDraft(); renderComposer();
        })));
      list.append(entry);
    }
    group.append(list);
    return group;
  }

  async function submitDraft(target, trigger, error) {
    if (busyToken || target !== form || !activeDraft.installation) return;
    updateForm();
    const draftKey = key();
    const current = activeDraft, isTemplate = page === 'template', upside = page === 'upscale';
    const token = { draft: draftKey, workspace: getSession()?.workspace };
    busyToken = token;
    trigger.disabled = true; trigger.textContent = '正在提交…'; error.hidden = true;
    const media = mediaOperations.has(current.operation) && !isTemplate ? current.refs.map(item => item.id) : [];
    const payload = { ...current.params, requestID: createID(), installation: current.installation,
      operation: isTemplate ? 'image.edit' : current.operation, prompt: current.prompt, media };
    if (isTemplate) {
      payload.template = templateID;
      payload.roles = Object.fromEntries(Object.entries(current.roles).map(([role, asset]) => [role, asset.id]));
    }
    if (payload.expansion !== undefined) payload.expansion = Number(payload.expansion);
    if (payload.zImageBudget !== undefined) payload.zImageBudget = Number(payload.zImageBudget);
    try {
      let created;
      try { created = await studio('submit', payload); }
      catch (error) { if (!(error instanceof TypeError)) throw error; created = await studio('submit', payload); }
      if (getSession()?.workspace !== token.workspace) return;
      previewID = created[0]?.id; previewKey = null;
      jobs = [...created, ...jobs.filter(job => !created.some(entry => entry.id === job.id))];
      renderPreview();
      if (window.matchMedia('(max-width: 720px)').matches && key() === draftKey) {
        const behavior = matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth';
        $(page).querySelector('.canvas')?.scrollIntoView({ block: 'start', behavior });
      }
      toast(created.length > 1 ? '已提交创作与后续超分。' : '已提交到 Mac 的任务队列。');
      refreshJobs();
    } catch (failure) {
      error.textContent = failure.message || '任务提交失败。'; error.hidden = false;
      toast(error.textContent, true);
    } finally {
      if (busyToken === token) busyToken = null;
      trigger.disabled = false; trigger.textContent = isTemplate ? '运行模板 ↗' : upside ? '提交超分 ↗' : '开始生成 ↗';
    }
  }

  function mediaElement(item, mini = false) {
    const media = node(item.kind === 'video' ? 'video' : 'img');
    if (item.kind === 'video') {
      media.src = item.url; media.preload = mini ? 'none' : 'metadata';
      media.playsInline = true; media.controls = !mini;
    } else {
      media.alt = mini ? '' : item.name || '图片预览';
      media.src = mini && item.preview?.endsWith('/preview') ? item.preview + '?size=320'
        : item.full ? item.url : item.preview || item.url;
      if (mini) { media.loading = 'lazy'; media.decoding = 'async'; }
    }
    return media;
  }
  function sizeLabel(asset) { return asset?.width && asset?.height ? asset.width + ' × ' + asset.height : ''; }
  function downloaded(item) {
    const anchor = link('下载', item.url + '?download=1', 'button');
    anchor.download = '';
    return anchor;
  }
  function showDialog(title, children) {
    $('dialog-title').textContent = title;
    setChildren($('dialog-content'), children);
    if (!$('media-dialog').open) $('media-dialog').showModal();
  }
  function sampleCaption() {
    const row = node('div', null, 'sample-caption'), copy = node('div');
    copy.append(node('h3', '琥珀折光'), node('p', '竖纹玻璃、暖色硬光与长阴影。'));
    const use = button('使用示例提示词', () => {
      const input = form.elements.prompt;
      const existing = input.value.trimEnd();
      const next = !existing ? amberPrompt : existing.includes(amberPrompt) ? existing : existing + '\n' + amberPrompt;
      if (next.length > input.maxLength) { toast('提示词已接近长度上限，无法添加示例。', true); return; }
      input.value = next; updateForm(); input.focus(); input.setSelectionRange(next.length, next.length);
    });
    use.title = '将示例添加到当前提示词';
    use.append(icon('arrow'));
    row.append(copy, use);
    return row;
  }
  function jobProgress(job) {
    const meter = node('progress');
    meter.max = job.total > 0 ? job.total : 1;
    if (job.total > 0 && job.completed >= 0) meter.value = Math.min(job.total, job.completed);
    meter.setAttribute('aria-label', '任务进度');
    return meter;
  }
  function renderPreview() {
    if (!['create', 'upscale', 'template'].includes(page)) return;
    const panel = $(page).querySelector('[data-preview]');
    if (!panel || !activeDraft) return;
    const ref = page === 'template' ? Object.values(activeDraft.roles)[0]
      : mediaOperations.has(activeDraft.operation) ? activeDraft.refs[0] : null;
    const inScope = job => page === 'template' ? job.label?.includes('创作模板') : job.operation === activeDraft.operation;
    const selected = previewID === '' ? null : jobs.find(job => job.id === previewID) || (ref ? null : jobs.find(inScope));
    const hash = JSON.stringify([selected?.id, selected?.state, selected?.position, selected?.completed,
      selected?.total, selected?.phase, selected?.livePreview, ref?.id, jobScope]);
    if (hash === previewKey) return;
    previewKey = hash;
    const sample = !selected && !ref && page === 'create' && activeDraft.operation === 'image.generate';
    const title = node('div', null, 'canvas-title'), name = node('h2', selected?.output ? '生成结果' : '画布预览');
    name.prepend(icon('frame'));
    const status = selected ? states[selected.state] : sample ? '提示词示例' : ref ? '输入素材'
      : form.querySelector('input[type=file]') ? '等待素材' : '待生成';
    title.append(name, node('small', status));
    const canvas = node('div', null, 'preview');
    if (selected?.output) canvas.append(mediaElement(selected.output));
    else if (selected?.livePreview) canvas.append(mediaElement({ url: selected.livePreview, kind: 'image', name: '实时采样预览' }));
    else if (ref) canvas.append(mediaElement(ref));
    else if (sample) {
      canvas.classList.add('sample-photo');
      canvas.append(picture('/amber-study.webp', '琥珀色竖纹玻璃在石台上投下折射光影'));
    }
    else {
      const input = form.querySelector('input[type=file]'), video = activeDraft.operation.startsWith('video.');
      const add = !selected && input ? button('选择' + (activeDraft.operation === 'video.upscale' ? '视频' : '图片'), () => input.click()) : null;
      const heading = selected ? states[selected.state] : input ? '添加第一份素材' : '等待生成';
      const detail = selected ? progressText(selected) : input ? '在画布中查看源素材和生成结果。' : '填写提示词并提交后，可在此查看进度和结果。';
      canvas.append(emptyState(video ? 'video' : 'image', heading, detail, add));
    }
    const buttons = selected?.output ? actions(button('查看详情', () => inspect(selected.output, selected)),
      downloaded(selected.output), button('继续使用', () => reuseMedia(selected.output))) : selected && !terminal.has(selected.state) ?
      actions(button('查看进度', () => { location.hash = '#jobs'; }), button('取消任务', action(() => mutateJob('cancel', selected)))) : null;
    const error = selected?.error ? node('p', selected.error, 'form-message') : null;
    setChildren(panel, [title, canvas, sample && sampleCaption(),
      selected && !terminal.has(selected.state) && jobProgress(selected), buttons, error]);
  }

  function useInComposer(item, target) {
    const chosenPage = target.endsWith('.upscale') ? 'upscale' : 'create';
    const model = catalog.models.find(row => row.id === drafts[chosenPage]?.model && row.operations.includes(target) && row.installations.length)
      || catalog.models.find(row => row.operations.includes(target) && row.installations.length)
      || catalog.models.find(row => row.operations.includes(target));
    const oldPage = page;
    page = chosenPage;
    drafts[chosenPage] = draftDefaults(model, model?.installations[0], target);
    drafts[chosenPage].refs = [item];
    page = oldPage;
    previewID = null; saveDraft(); $('media-dialog').close();
    if (location.hash === '#' + chosenPage) renderComposer(); else location.hash = '#' + chosenPage;
  }
  function reuseMedia(item) {
    const buttons = item.kind === 'video' ? [['视频超分', 'video.upscale']] :
      [['继续编辑', 'image.edit'], ['图生视频', 'video.image'], ['图片超分', 'image.upscale']];
    showDialog('使用此素材', [actions(...buttons.map(([title, operation]) => button(title, () => useInComposer(item, operation))))]);
  }
  function inspect(item, job) {
    const view = node('div', null, 'media-large');
    const output = item.kind === 'image' && item.url.endsWith(item.id) && job?.output
      ? { ...item, full: true } : item;
    const show = media => { view.replaceChildren(mediaElement(media)); };
    show(output);
    const controls = actions(downloaded(item), button('用作素材', () => reuseMedia(item)));
    if (job?.comparison) {
      controls.prepend(button('查看原始输入', () => show({ ...job.comparison, name: '原始输入' })));
      if (job.lastFrame) controls.prepend(button('查看尾帧', () => show({ url: job.lastFrame, kind: 'image', name: '尾帧' })));
      controls.prepend(button('查看输出', () => show(output)));
      if (job.comparison.kind === output.kind) {
        controls.prepend(button('拖动对比', () => view.replaceChildren(comparison(output, job.comparison))));
      }
    }
    const details = [sizeLabel(item), item.frames > 1 ? item.frames + ' 帧' : '', item.fps > 1 ? item.fps + ' FPS' : '',
      job?.route, job ? '种子 ' + job.seed : '', job?.prompt].filter(Boolean);
    showDialog(jobTitle(job) || item.name, [view, controls, node('p', details.join('\n'), 'media-details')]);
  }
  function comparison(output, source) {
    const stage = node('div', null, 'comparison');
    const stack = node('div', null, 'comparison-stack');
    const result = mediaElement(output);
    const original = mediaElement({ ...source, name: '原始素材' });
    original.classList.add('comparison-original');
    const divider = node('div', null, 'comparison-divider');
    divider.setAttribute('aria-hidden', 'true');
    stack.append(result, original, divider);
    const range = node('input');
    range.type = 'range'; range.min = '0'; range.max = '100'; range.value = '50';
    range.setAttribute('aria-label', '原始素材与输出对比分界线');
    const position = () => stack.style.setProperty('--split', range.value + '%');
    range.addEventListener('input', position); position();
    const labels = node('div', null, 'comparison-labels');
    labels.append(node('span', '原始'), node('span', '输出'));
    if (output.kind === 'video') {
      original.muted = true; original.controls = false;
      const sync = () => {
        if (original.readyState && Math.abs(original.currentTime - result.currentTime) > .08) original.currentTime = result.currentTime;
      };
      result.addEventListener('play', () => { sync(); void original.play().catch(() => {}); });
      result.addEventListener('pause', () => original.pause());
      result.addEventListener('seeked', sync);
      result.addEventListener('timeupdate', sync);
      original.addEventListener('loadedmetadata', sync);
    }
    stage.append(stack, labels, range);
    return stage;
  }
  function galleryCard(item, title, callback) {
    const card = node('article', null, 'gallery-card'), control = button('', callback);
    control.setAttribute('aria-label', '查看 ' + title);
    if (item.kind === 'video') {
      const mark = node('span', '视频', 'video-thumb'); mark.prepend(icon('video')); control.append(mark);
    } else control.append(mediaElement(item, true));
    card.append(control, node('strong', title), node('small', sizeLabel(item)));
    return card;
  }
  async function pickMedia(kind, onPick) {
    const recent = await studio('library');
    const items = [...recent.outputs.map(row => row.output), ...recent.uploads].filter(item => item?.kind === kind);
    const gallery = node('div', null, 'gallery');
    setChildren(gallery, items.length ? items.map(item => galleryCard(item, item.name, action(async () => {
      await onPick(item); $('media-dialog').close();
    }))) : [message('素材库中还没有这种素材。')]);
    showDialog('选择素材', [gallery]);
  }
  function annotate(ref, replace) {
    const board = node('div', null, 'paint-board'), image = mediaElement(ref), canvas = node('canvas');
    const context = canvas.getContext('2d'), points = [];
    let stroke;
    const brush = (event, start) => {
      const rect = canvas.getBoundingClientRect();
      const point = [Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)),
        Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height))];
      if (start) { stroke = []; points.push(stroke); }
      stroke.push(point);
      draw(context, '#ff4b4b');
    };
    const draw = (target, color) => {
      target.strokeStyle = target.fillStyle = color;
      target.lineWidth = Math.max(4, Math.min(canvas.width, canvas.height) * .026);
      target.lineCap = target.lineJoin = 'round';
      for (const trail of points) {
        target.beginPath(); target.moveTo(...trail[0]);
        if (trail.length === 1) target.lineTo(trail[0][0] + .01, trail[0][1]);
        else for (const p of trail.slice(1)) target.lineTo(...p);
        target.stroke();
      }
    };
    image.addEventListener('load', () => { canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; });
    canvas.addEventListener('pointerdown', event => { canvas.setPointerCapture(event.pointerId); brush(event, true); });
    canvas.addEventListener('pointermove', event => { if (stroke) brush(event, false); });
    for (const end of ['pointerup', 'pointercancel']) canvas.addEventListener(end, () => { stroke = null; });
    board.append(image, canvas);
    const save = mask => action(async () => {
      if (!points.length || !image.naturalWidth) throw new Error('请先在图上画出编辑区域。');
      const out = node('canvas'); out.width = canvas.width; out.height = canvas.height;
      const target = out.getContext('2d');
      if (mask) { target.fillStyle = '#000'; target.fillRect(0, 0, out.width, out.height); }
      else target.drawImage(image, 0, 0);
      draw(target, mask ? '#fff' : '#ff4b4b');
      const blob = await new Promise(resolve => out.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('无法生成标注。');
      const stored = await upload(new File([blob], mask ? '蒙版.png' : '标注.png', { type: 'image/png' }), 'image');
      if (mask && !mediaOperations.has(activeDraft.operation)) throw new Error('请先切换到参考编辑。');
      if (mask && page !== 'template') {
        if (activeDraft.refs.length >= maxReferences(models().find(model => model.id === activeDraft.model), activeDraft.operation)) {
          throw new Error('当前工作流没有新的蒙版参考位。');
        }
        activeDraft.refs.push(stored); saveDraft(); renderComposer();
      } else replace(stored);
      $('media-dialog').close();
    });
    showDialog('标注参考图', [board, actions(button('撤销笔画', () => {
      points.pop(); context.clearRect(0, 0, canvas.width, canvas.height); draw(context, '#ff4b4b');
    }), page !== 'template' ? button('添加蒙版', save(true)) : null, button('保存标注', save(false), 'primary'))]);
  }

  function progressText(job) {
    const count = job.total > 0 && job.completed > 0 ? job.completed + ' / ' + job.total : '';
    return [states[job.state], job.position ? '队列第 ' + job.position + ' 位' : '',
      !terminal.has(job.state) ? phases[job.phase] || '' : '', count].filter(Boolean).join(' · ');
  }
  function jobTitle(job) { return job ? job.label || operations[job.operation] || '任务' : ''; }
  async function mutateJob(path, job, ask) {
    if (ask && !confirm(ask)) return;
    await studio(path, { id: job.id });
    await refreshJobs();
    if (page === 'library') await renderLibrary();
  }
  async function reuseJob(job) {
    const recipe = await studio('reuse', { id: job.id });
    const target = recipe.template ? 'template/' + recipe.template : recipe.operation.endsWith('.upscale') ? 'upscale' : 'create';
    const available = await studio('library');
    const entries = [...available.uploads, ...available.outputs.map(row => row.output)];
    const byID = id => entries.find(item => item?.id === id);
    const { installation, operation, prompt, media, roles, template, ...params } = recipe;
    const model = catalog.models.find(row => row.installations.some(item => item.id === installation));
    drafts[target] = { model: model?.id || '', installation, operation, prompt, params,
      refs: (media || []).map(byID).filter(Boolean),
      roles: Object.fromEntries(Object.entries(roles || {}).map(([role, id]) => [role, byID(id)]).filter(([, item]) => item)) };
    previewID = null; saveDraft();
    if (location.hash === '#' + target) renderComposer(); else location.hash = '#' + target;
  }
  function renderJobs() {
    if (page !== 'jobs') return;
    const manager = getSession()?.canManage === true;
    let selected = jobs;
    if (jobFilter === 'active') selected = selected.filter(job => !terminal.has(job.state));
    if (jobFilter === 'done') selected = selected.filter(job => job.state === 'succeeded');
    if (jobFilter === 'failed') selected = selected.filter(job => ['failed', 'cancelled', 'interrupted'].includes(job.state));
    const active = jobs.filter(job => !terminal.has(job.state)).length;
    const done = jobs.filter(job => job.state === 'succeeded').length;
    const hash = JSON.stringify([jobFilter, jobScope, jobs.length, active, done, selected]);
    if (hash === jobKey) return;
    jobKey = hash;
    setTabs($('job-filters'), [['all', '全部', false, jobs.length], ['active', '进行中', false, active],
      ['done', '已完成', false, done], ['failed', '已停止', false, jobs.length - active - done]], jobFilter, filter => {
      jobFilter = filter; renderJobs();
    });
    const scope = manager ? button(jobScope === 'own' ? '查看 Mac / API 任务' : '返回此浏览器任务', action(async () => {
      jobScope = jobScope === 'own' ? 'host' : 'own'; scopeVersion++; jobs = []; jobKey = null; renderJobs();
      await refreshJobs();
    })) : null;
    setChildren($('job-options'), [scope, $('refresh') || button('刷新', action(refreshJobs))]);
    $('job-status').hidden = selected.length > 0;
    if (!selected.length) {
      const own = jobScope === 'own';
      $('job-status').replaceChildren(jobs.length
        ? emptyState('jobs', '没有符合此筛选的任务', '', button('查看全部任务', () => { jobFilter = 'all'; renderJobs(); }))
        : emptyState('jobs', '队列中还没有任务', own ? '提交创作后，可以在这里查看进度、下载和复用结果。' : 'Mac 与本地 API 的任务会显示在这里。',
          own ? link('去创作', '#create', 'button') : null));
    }
    setChildren($('job-list'), selected.map(job => {
      const row = node('li', null, 'card'), left = node('div', null, 'job-info');
      const title = node('h3', jobTitle(job));
      const time = new Date(job.created).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      const stateTag = node('span', states[job.state] || job.state, 'tag');
      stateTag.dataset.state = job.state;
      left.append(title, actions(stateTag, node('small', [job.source, job.model, time].filter(Boolean).join(' · '))));
      if (!terminal.has(job.state)) {
        left.append(node('p', progressText(job)));
        left.append(jobProgress(job));
      } else if (job.state === 'succeeded') {
        left.append(node('p', [sizeLabel(job), job.elapsed > 0 ? job.elapsed.toFixed(1) + ' 秒' : '', job.route].filter(Boolean).join(' · ')));
      }
      if (job.error) left.append(node('p', job.error, 'job-error'));
      const buttons = actions(job.output ? button('查看结果', () => inspect(job.output, job)) : null, job.output && downloaded(job.output));
      if (terminal.has(job.state)) {
        buttons.append(button('复用设置', action(() => reuseJob(job))), button('删除记录', action(() => mutateJob('delete', job, '删除这项任务记录？')), 'ghost'));
      } else if (!['cancelling', 'cleanup_pending'].includes(job.state)) buttons.append(button('取消', action(() => mutateJob('cancel', job)), 'action-danger'));
      const thumb = node('div', null, 'job-thumb');
      thumb.append(job.output?.kind === 'image' ? mediaElement(job.output, true) : icon(job.operation.startsWith('video.') ? 'video' : 'image'));
      row.append(thumb, left, buttons);
      return row;
    }));
  }

  async function renderLibrary() {
    const workspace = getSession()?.workspace;
    const scope = libraryScope;
    const view = await studio('library' + (scope === 'host' ? '?scope=host' : ''));
    if (page !== 'library' || getSession()?.workspace !== workspace || scope !== libraryScope) return;
    library = view;
    renderGallery();
  }
  function renderGallery() {
    const manager = getSession()?.canManage === true;
    const add = buildFileInput('导入素材', 'image/*,.heic,.heif,video/mp4,video/quicktime,.mov,.m4v,.mp4', async files => {
      for (const file of files) await upload(file, /\.(mov|m4v|mp4)$/i.test(file.name) ? 'video' : 'image');
      await renderLibrary();
      toast('已添加到素材库。');
    }, true);
    setChildren($('library-options'), [manager ? button(libraryScope === 'own' ? '查看 Mac 结果' : '返回我的素材', action(async () => {
      libraryScope = libraryScope === 'own' ? 'host' : 'own'; await renderLibrary();
    })) : null, libraryScope === 'own' ? add : null]);
    setTabs($('library-filters'), [['all', '全部'], ['image', '图片'], ['video', '视频'], ['uploads', '已上传', libraryScope === 'host']],
      libraryFilter, filter => { libraryFilter = filter; renderGallery(); });
    const items = [];
    if (libraryFilter !== 'uploads') {
      for (const job of library.outputs) {
        if (libraryFilter !== 'all' && libraryFilter !== job.output?.kind) continue;
        if (!job.output) continue;
        const tile = galleryCard(job.output, jobTitle(job), () => inspect(job.output, job));
        tile.append(actions(downloaded(job.output), button('移入废纸篓', action(() => mutateJob('trash', job, '将这张作品移入 Mac 废纸篓？')), 'ghost')));
        items.push(tile);
      }
    }
    for (const asset of library.uploads) {
      if (['all', 'uploads', asset.kind].includes(libraryFilter)) items.push(galleryCard(asset, asset.name, () => inspect(asset)));
    }
    const isOwn = libraryScope === 'own';
    const empty = () => emptyState('library', libraryFilter === 'all' ? '素材库还是空的' : '这个分类还没有素材',
      isOwn ? '导入图片或视频，生成的作品也会保存在这里。' : 'Mac 和本地 API 的创作结果会出现在这里。',
      isOwn ? button('导入素材', () => add.querySelector('input').click()) : null);
    setChildren($('library-list'), items.length ? items : [empty()]);
  }

  function posewearShowcase(app) {
    const hero = node('article', null, 'app-hero'), copy = node('div', null, 'app-hero-copy');
    const samples = app.url + 'samples/';
    const brand = node('span', 'PoseWear', 'app-wordmark'); brand.lang = 'en';
    copy.append(brand, node('h2', app.name), node('p', '选人物、服装与姿势，在同一张图上完成换装与造型。'),
      link('打开人物工作室 ↗', app.url, 'button primary'));

    const book = node('div', null, 'lookbook'), plates = node('div', null, 'look-plates');
    const figure = (source, label, classes) => {
      const item = node('figure', null, classes);
      item.append(picture(source), node('figcaption', label)); return item;
    };
    const person = figure(samples + 'demo-person.webp', '人物', 'look-input look-person');
    const reference = figure(samples + 'demo-outfit.webp', '服装', 'look-input look-ref');
    const result = figure('/look-outfit.webp', '换装结果', 'look-result');
    result.firstChild.alt = '同一人物换上参考图中的牛仔套装';
    plates.append(person, reference, result);
    const controls = node('div', null, 'look-controls');
    const picker = node('div', null, 'look-picker');
    picker.setAttribute('role', 'group'); picker.setAttribute('aria-label', 'PoseWear 样片');
    for (const [mode, title] of [['outfit', '换装'], ['pose', '换姿势']]) {
      const choice = button(title, () => {
        const pose = mode === 'pose';
        reference.firstChild.src = samples + (pose ? 'demo-pose-front.webp' : 'demo-outfit.webp');
        reference.lastChild.textContent = pose ? '姿势' : '服装';
        result.firstChild.src = pose ? '/look-pose.webp' : '/look-outfit.webp';
        result.firstChild.alt = pose ? '同一人物按姿势参考抬起手臂' : '同一人物换上参考图中的牛仔套装';
        result.lastChild.textContent = title + '结果';
        for (const item of picker.children) item.setAttribute('aria-pressed', String(item === choice));
      });
      choice.setAttribute('aria-pressed', String(mode === 'outfit')); picker.append(choice);
    }
    controls.append(node('span', '样片预览'), picker);
    book.append(plates, controls); hero.append(copy, book);
    return hero;
  }

  function templateCard(template, samples) {
    const tile = link(null, '#template/' + template.id, 'template-card');
    tile.setAttribute('aria-label', '打开模板 · ' + template.name);
    const art = node('div', null, 'template-art'); art.dataset.template = template.id;
    art.setAttribute('aria-hidden', 'true');
    const shots = {
      outfit: [samples + 'demo-outfit.webp'], identity: ['/cover.webp', '/look-outfit.webp'],
      face: ['/cover.webp'], outpaint: ['/amber-study.webp'], transparent: ['/template-cutout.webp']
    };
    if (samples && shots[template.id]) art.append(...shots[template.id].map(src => picture(src)));
    else art.append(icon('apps'));
    const head = node('div', null, 'template-title');
    head.append(node('h3', template.name), icon('arrow'));
    const detail = node('p', template.detail);
    detail.id = 'detail-' + template.id; tile.setAttribute('aria-describedby', detail.id);
    tile.append(art, head, detail); return tile;
  }

  async function renderApps() {
    const current = getSession()?.workspace;
    const available = await api('/api/apps');
    if (page !== 'apps' || getSession()?.workspace !== current) return;
    apps = available;
    const admin = getSession()?.canManage === true;
    setChildren($('app-admin'), [admin ? buildFileInput('安装 .tcapp', '.tcapp', async ([file]) => {
      await upload(file, 'app', base + 'host/upload'); await renderApps(); toast('应用已安装。');
    }) : null]);
    setChildren($('app-list'), apps.map(app => {
      if (app.official) return posewearShowcase(app);
      const card = node('article', null, 'card tile');
      const foot = actions();
      foot.append(link('打开应用 ↗', app.url, 'button'));
      if (admin) foot.append(button('移除', action(async () => {
        if (!confirm('从 Mac 移除此应用包？')) return;
        await host('uninstallApp', { id: app.id }); await renderApps();
      })));
      card.append(icon('apps'), node('h3', app.name), node('p', app.description),
        node('small', app.permissions.join('、')), foot);
      return card;
    }));
    const official = apps.find(app => app.official);
    setChildren($('template-list'), catalog.templates.map(template => templateCard(template, official ? official.url + 'samples/' : null)));
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return '—';
    return bytes >= 1024 ** 3 ? (bytes / 1024 ** 3).toFixed(1) + ' GiB' : (bytes / 1024 ** 2).toFixed(1) + ' MiB';
  }
  function selectInstallation(model, installation, upside) {
    const target = upside ? 'upscale' : 'create';
    const old = page;
    page = target;
    drafts[target] = draftDefaults(model, installation);
    page = old;
    previewID = null; saveDraft(); location.hash = '#' + target;
  }
  async function modelCommand(name, payload, notice) {
    await host(name, payload);
    if (notice) toast(notice);
    await refreshDevices();
  }
  function showModels() {
    if (page !== 'models') return;
    const target = $('models'), currentOpen = new Set([...target.querySelectorAll('details[open]')].map(row => row.dataset.id));
    const admin = getSession()?.canManage === true;
    const refresh = button('刷新模型列表', action(async () => { await loadCatalog(true); showModels(); }));
    const intro = heading('模型中心', '选择 Mac 已安装的模型与 LoRA，或查看可下载版本。', refresh);
    const rows = node('div', null, 'models-section');
    const status = node('div', null, 'card device-card'); status.id = 'download-status'; status.hidden = true;
    const search = node('input');
    search.type = 'search'; search.placeholder = '按名称、版本或安装搜索';
    search.setAttribute('aria-label', '搜索模型');
    const searchBox = node('div', null, 'model-search'); searchBox.append(icon('search'), search);
    const term = target.querySelector('input[type=search]')?.value || '';
    search.value = term;
    const noMatches = emptyState('search', '没有匹配的模型', '可以换一个名称，或清除搜索查看整个目录。',
      button('清除搜索', () => { search.value = ''; filterModels(); search.focus(); }));
    const filterModels = () => {
      const query = search.value.trim().toLocaleLowerCase();
      for (const entry of rows.children) entry.hidden = !entry.textContent.toLocaleLowerCase().includes(query);
      noMatches.hidden = [...rows.children].some(entry => !entry.hidden);
    };
    search.addEventListener('input', filterModels);
    target.replaceChildren(intro, status, searchBox, rows, noMatches);
    const installedFirst = [...catalog.models].sort((a, b) => Number(!!b.installations.length) - Number(!!a.installations.length));
    for (const model of installedFirst) {
      const details = node('details'); details.dataset.id = model.id;
      details.open = currentOpen.has(model.id);
      const summary = node('summary', null, 'model-summary');
      const state = node('small', model.installations.length ? model.installations.length + ' 个安装' : '未安装');
      if (model.installations.length) state.className = 'installed';
      summary.append(icon(model.kind === 'video' ? 'video' : 'models'), node('strong', model.name), state, icon('chevron'));
      details.append(summary);
      for (const installation of model.installations) {
        const row = node('div', null, 'model-row');
        row.append(node('div', installation.name + (installation.turbo ? ' · Turbo 8 步' : '')), actions(
          button(model.operations.every(value => value.endsWith('.upscale')) ? '用于超分' : '用于创作',
            () => selectInstallation(model, installation, model.operations.every(value => value.endsWith('.upscale')))),
          admin ? button('预加载', action(() => modelCommand('preload', { id: installation.id }, '模型已加载。'))) : null,
          admin ? button('设为 Mac 默认', action(() => modelCommand('selectModel', { id: installation.id }, '已在 Mac 上选定。'))) : null,
          admin && !installation.id.startsWith('configured-') ? button('取消登记', action(async () => {
            if (confirm('从模型中心取消登记？权重文件保留。')) await modelCommand('unregisterModel', { id: installation.id });
          }), 'ghost') : null));
        details.append(row);
      }
      for (const variant of model.variants) {
        const row = node('div', null, 'model-row'), copy = node('div');
        copy.append(node('strong', variant.name), node('p', variant.preparation, 'inline-note'));
        const providers = variant.provider ? [[variant.provider, variant.provider === 'modelscope' ? 'ModelScope' : 'Hugging Face']]
          : [['modelscope', 'ModelScope'], ['huggingface', 'Hugging Face']];
        const source = select('_provider', providers, providers[0][0]);
        source.setAttribute('aria-label', '下载源');
        const download = payload => ({ model: model.id, variant: variant.id, provider: source.value, ...payload });
        row.append(copy, admin ? actions(source, button('查看下载大小', action(async () => {
          const plan = await host('planDownload', download()); toast(plan.files + ' 个文件 · 需下载 ' + formatBytes(plan.bytes));
        })), button('下载到 Mac', action(async () => {
          if (confirm('在 Mac 上下载 ' + variant.name + '？')) await modelCommand('download', download(), '已开始下载。');
        }), 'primary')) : null);
        details.append(row);
      }
      if (model.download) {
        const source = link('获取 Core ML 模型 ↗', model.download, 'button');
        source.target = '_blank'; source.rel = 'noopener noreferrer';
        details.append(actions(source));
      }
      if (admin) {
        details.append(actions(button('在 Mac 上选择模型文件', action(() => modelCommand('pickModel', { model: model.id })))));
        if (model.supportsLoRA) details.append(actions(buildFileInput('上传 LoRA', '.safetensors', async ([file]) => {
          await upload(file, 'lora', base + 'host/upload', { 'X-TurboCider-Model': model.id });
          await loadCatalog(true); showModels(); toast('LoRA 已登记。');
        })));
      }
      rows.append(details);
    }
    if (admin) {
      const extras = node('details'); extras.dataset.id = 'extras'; extras.open = currentOpen.has('extras');
      extras.append(node('summary', 'LoRA、VAE 与 ANE'));
      for (const adapter of catalog.registeredLoRAs || []) {
        extras.append(settingRow(adapter.name, adapter.models.join('、'), button('取消登记', action(async () => {
          if (confirm('取消登记此 LoRA？原始权重保留。')) await modelCommand('unregisterLoRA', { id: adapter.id });
        }))));
      }
      for (const vae of catalog.tinyVAE) {
        extras.append(settingRow(vae.name, vae.installed ? '已安装' : '可下载',
          button('下载', action(() => modelCommand('tinyVAE', { id: vae.id })))));
      }
      extras.append(settingRow('PrunaVAED', catalog.pruna ? '已安装' : 'LTX 视频解码器',
        button('下载', action(() => modelCommand('pruna')))));
      for (const part of catalog.ane) {
        extras.append(settingRow(part.name, part.model, button('编译', action(() => modelCommand('compileANE', { id: part.id }))),
          button('取消登记', action(async () => {
            if (confirm('取消登记此 ANE 分区？')) await modelCommand('removeANE', { id: part.id });
          }))));
      }
      const aneModel = select('_aneModel', catalog.models.filter(item => item.supportsANE).map(item => [item.id, item.name]), catalog.preferred);
      aneModel.setAttribute('aria-label', 'ANE 分区适用模型');
      extras.append(actions(button('在 Mac 上导入 LoRA 配置', action(() => modelCommand('pickLoRA'))),
        aneModel, button('在 Mac 上导入 ANE 分区', action(() => modelCommand('pickANE', { model: aneModel.value }))),
        button('在 Mac 上导入模型配置', action(() => modelCommand('pickConfiguration'))),
        button('在 Mac 上更改模型库目录', action(() => modelCommand('pickModelRoot')))));
      rows.append(extras);
    }
    filterModels();
    showDownload();
  }
  function settingRow(title, detail, ...options) {
    const row = node('div', null, 'model-row');
    const copy = node('div');
    copy.append(node('strong', title), node('p', detail, 'inline-note'));
    row.append(copy, actions(...options));
    return row;
  }
  function showDownload() {
    const target = $('download-status');
    if (!target || !devices) return;
    const current = devices.transfer;
    target.hidden = !devices.download && !devices.downloadMessage;
    const next = node('div', null, 'subhead');
    next.append(node('h2', devices.download ? 'Mac 正在处理模型' : '最近的模型操作'));
    if (devices.download && getSession()?.canManage) next.append(button('取消下载', action(() => modelCommand('cancelDownload'))));
    const meter = node('progress');
    meter.setAttribute('aria-label', '模型下载进度');
    if (current?.total > 0) { meter.max = current.total; meter.value = current.completed; }
    setChildren(target, [next, devices.download && meter,
      node('p', current?.total > 0 ? formatBytes(current.completed) + ' / ' + formatBytes(current.total) : devices.downloadMessage, 'inline-note')]);
  }

  function showSettings() {
    if (page !== 'settings') return;
    const session = getSession(), admin = session?.canManage === true, status = devices || {};
    const panel = node('div', null, 'models-section');
    const summary = node('section', null, 'card device-card');
    const metrics = node('div', null, 'metrics');
    for (const [label, value] of [['Mac CPU', status.cpu], ['Mac GPU', status.gpu], ['App 驻留内存', formatBytes(status.resident)],
      ['物理内存', formatBytes(status.memory)]]) {
      const tile = node('div', null, 'metric');
      tile.append(node('small', label), node('b', typeof value === 'number' ? Math.round(value) + '%' : value || '—'));
      metrics.append(tile);
    }
    summary.append(node('h2', '运行状态'), metrics, node('p', (status.busy ? '推理中' : '队列空闲') +
      ' · ' + (status.queue || 0) + ' 项等待' + (status.loadedModel ? ' · ' + status.loadedModel : ''), 'inline-note'));
    if (admin) {
      const release = button('释放模型内存', action(() => modelCommand('unload')));
      release.disabled = !status.canUnload;
      summary.append(actions(release, button('释放超分模型', action(() => modelCommand('releaseUpscaler')))));
    }
    const services = node('section', null, 'card device-card');
    services.append(node('h2', '本地 API'), node('p', status.apiRunning
      ? '已在 Mac 的 127.0.0.1:' + status.apiPort + ' 上运行，与网页共用任务队列。'
      : 'Mac 上的脚本或 ComfyUI 可以通过本地 API 提交任务。', 'inline-note'));
    if (admin && status.apiAvailable) services.append(actions(button(status.apiRunning ? '停止本地 API' : '启动本地 API',
      action(() => modelCommand(status.apiRunning ? 'stopAPI' : 'startAPI')))));
    const identity = node('section', null, 'card device-card');
    identity.append(node('h2', '当前连接'), node('p', (session?.name || '浏览器') +
      (session?.direct ? ' · 局域网直连' : admin ? ' · 已授权管理 Mac' : ' · 已配对'), 'inline-note'));
    if (!session?.direct) {
      const logout = button('解除配对', action(async () => {
        if (!confirm('解除此浏览器的配对？')) return;
        await api('/api/logout', {}); location.replace('/');
      }));
      logout.id = 'logout'; identity.append(actions(logout));
    }
    panel.append(summary, services, identity);
    if (admin) {
      const cache = node('section', null, 'card device-card');
      const days = select('_days', [0, 7, 14, 30, 60, 90].map(value => [value, value ? value + ' 天' : '关闭自动清理']), status.cacheDays || 0);
      days.setAttribute('aria-label', '张量缓存保留期限');
      cache.append(node('h2', '张量缓存'), node('p', (status.cacheEntries ?? '—') + ' 项 · ' + formatBytes(status.cacheBytes), 'inline-note'),
        actions(days, button('读取统计', action(() => modelCommand('cacheRefresh'))),
          button('保存保留期限', action(() => modelCommand('cacheRetention', { days: Number(days.value) }))),
          button('清理缓存', action(async () => {
            if (confirm('清理 Mac 已登记的张量缓存？')) await modelCommand('cachePrune', { days: Number(days.value) });
          })), button('清除计算编译缓存', action(() => modelCommand('clearRuntimeCache')))),
        actions(button('登记 Mac 的张量目录', action(() => modelCommand('pickDiagnosticFolder')))),
        node('p', status.cacheMessage, 'inline-note'));
      panel.append(cache);
    }
    $('settings').replaceChildren(heading('设置', '查看当前 Mac 的资源与服务状态。', button('刷新状态', action(refreshDevices))), panel);
  }

  async function loadCatalog(force = false) {
    const workspace = getSession()?.workspace;
    if (catalog && catalogWorkspace === workspace && !force) return catalog;
    if (catalogPromise?.workspace === workspace) return catalogPromise.value;
    const value = studio('catalog');
    const pending = { value, workspace };
    catalogPromise = pending;
    try {
      const data = await value;
      if (getSession()?.workspace === workspace) { catalog = data; catalogWorkspace = workspace; }
      return data;
    } finally { if (catalogPromise === pending) catalogPromise = null; }
  }
  async function refreshJobs() {
    const workspace = getSession()?.workspace, scope = jobScope, version = scopeVersion;
    if (!workspace) return;
    const reply = await studio('jobs' + (scope === 'host' && page === 'jobs' ? '?scope=host' : ''));
    if (getSession()?.workspace !== workspace || scopeVersion !== version) return;
    jobs = reply.items;
    $('job-count').textContent = reply.queued;
    $('job-count').hidden = reply.queued === 0;
    $('queue').classList.toggle('is-busy', reply.busy);
    $('queue').classList.remove('is-offline');
    $('queue-text').textContent = reply.queued ? '排队 ' + reply.queued : reply.busy ? 'Mac 推理中' : 'Mac 已连接';
    renderJobs(); renderPreview();
  }
  async function refreshDevices() {
    if (!getSession()) return;
    const previous = devices;
    devices = await studio('device');
    if (previous?.download && !devices.download) {
      await loadCatalog(true);
      if (page === 'models') showModels();
    }
    if (page === 'settings' && !document.activeElement?.matches('#settings button, #settings select, #settings input')) showSettings();
    if (page === 'models') showDownload();
  }
  async function poll() {
    if (pollBusy || !getSession() || document.hidden) return;
    pollBusy = true;
    try {
      await refreshJobs();
      if (['models', 'settings'].includes(page)) await refreshDevices();
    } catch (error) {
      if (getSession()) { $('queue-text').textContent = '连接中断'; $('queue').classList.remove('is-busy'); $('queue').classList.add('is-offline'); }
    } finally { pollBusy = false; }
  }

  async function openPage({ name, parameter }) {
    if (page !== name || (name === 'template' && templateID !== parameter)) previewID = null;
    page = name; templateID = name === 'template' ? parameter : undefined;
    if (name !== 'jobs' && jobScope === 'host') { jobs = []; scopeVersion++; }
    try {
      if (['create', 'upscale', 'template', 'models'].includes(name) && !catalog) $(name).replaceChildren(message('正在读取 Mac 的模型…'));
      const workspace = getSession()?.workspace;
      await loadCatalog();
      if (page !== name || getSession()?.workspace !== workspace) return;
      if (['create', 'upscale', 'template'].includes(name)) { renderComposer(); await refreshJobs(); }
      else if (name === 'apps') await renderApps();
      else if (name === 'jobs') { jobKey = null; await refreshJobs(); }
      else if (name === 'library') await renderLibrary();
      else if (name === 'models') { showModels(); await refreshDevices(); }
      else if (name === 'settings') { showSettings(); await refreshDevices(); }
    } catch (error) {
      if (getSession() && page === name) toast(error.message || '无法加载工作台。', true);
    }
  }

  window.addEventListener('tc-session', ({ detail }) => {
    drafts = preference('turbocider.drafts.' + detail.workspace) || {};
    jobs = []; devices = null; catalog = null; previewID = null; busyToken = null;
    jobScope = libraryScope = 'own'; jobFilter = libraryFilter = 'all'; jobKey = null;
    clearInterval(timer); timer = setInterval(poll, 2100);
  });
  window.addEventListener('tc-disconnect', () => {
    clearInterval(timer);
    if ($('media-dialog').open) $('media-dialog').close();
  });
  window.addEventListener('tc-route', ({ detail }) => openPage(detail));
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
  $('refresh').addEventListener('click', action(refreshJobs));
  document.addEventListener('paste', event => {
    if (!['create', 'upscale', 'template'].includes(page) || !activeDraft || !event.clipboardData?.files.length) return;
    const file = [...event.clipboardData.files].find(value => value.type.startsWith('image/'));
    const input = form?.querySelector('input[type=file][accept^="image"]');
    if (!file || !input) return;
    event.preventDefault();
    const transfer = new DataTransfer();
    transfer.items.add(file); input.files = transfer.files;
    input.dispatchEvent(new Event('change'));
  });
})();
