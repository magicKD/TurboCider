'use strict';

// Browser transports shared by the full studio and the floating workbench.
window.PoseWearImageProviders = (() => {
  const providers = {
    openai: {
      name: 'OpenAI', base: 'https://api.openai.com/v1', model: 'gpt-image-1',
      models: ['gpt-image-1', 'gpt-image-1-mini', 'gpt-image-2', 'dall-e-2']
    },
    gemini: {
      name: 'Gemini', base: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-2.5-flash-image',
      models: ['gemini-2.5-flash-image', 'gemini-3-pro-image-preview', 'gemini-3.1-flash-image']
    },
    seedream: {
      name: '豆包 Seedream', base: 'https://ark.cn-beijing.volces.com/api/v3', model: 'doubao-seedream-5-5-261028',
      models: ['doubao-seedream-5-5-261028', 'doubao-seedream-5-0-flash-260915', 'doubao-seedream-4-5-251128', 'doubao-seedream-4-0-250828']
    },
    custom: {
      name: '自定义中转', base: '', model: 'gpt-image-1',
      models: ['gpt-image-1', 'gpt-image-2', 'grok-imagine-image-2.0', 'gemini-2.5-flash-image', 'flux-kontext-pro', 'qwen-image-edit', 'doubao-seedream-4-0-250828']
    }
  };

  const officialHost = /^(api\.openai\.com|generativelanguage\.googleapis\.com|ark\.[a-z0-9-]+\.(volces|bytepluses)\.com)$/;
  const imageSizes = { '2:3': '1024x1536', '3:4': '1024x1536', '9:16': '1024x1536', '1:1': '1024x1024', '4:3': '1536x1024', '16:9': '1536x1024' };
  const seedreamSizes = { auto: '1728x2304', '2:3': '1664x2496', '3:4': '1728x2304', '9:16': '1440x2560', '1:1': '2048x2048', '4:3': '2304x1728', '16:9': '2560x1440' };
  const textOnlyWarning = '当前模型未接收参考图，本次仅按提示词生成。';

  function protocol(settings) {
    if (settings.prov === 'openai') return 'edits';
    if (settings.prov === 'custom') return settings.proto;
    return settings.prov;
  }

  function apiBase(settings) {
    const base = (settings.base || providers[settings.prov].base).trim().replace(/\/+$/, '');
    if (protocol(settings) === 'gemini' && /\/v1(beta|alpha)?$/.test(base)) return base;
    if (settings.prov === 'gemini') return base + '/v1beta';
    if (settings.prov === 'seedream') return base;
    return /\/v\d+$/.test(base) ? base : base + '/v1';
  }

  function request(settings, path, options) {
    const url = apiBase(settings) + path;
    const headers = new Headers(options.headers);
    if (protocol(settings) === 'gemini') headers.set('x-goog-api-key', settings.key);
    else headers.set('Authorization', 'Bearer ' + settings.key);
    if (!settings.proxy) return fetch(url, { ...options, headers });
    headers.set('x-target-url', url);
    if (settings.prov === 'custom' || !officialHost.test(new URL(url).hostname)) headers.set('x-proxy-custom', '1');
    return fetch('/api/proxy', { ...options, headers });
  }

  function redact(text, key) {
    const value = String(text);
    return key ? value.split(key).join('***') : value;
  }

  const safetyReason = /safety system|safety_violations|moderation_blocked|content_policy_violation|content policy|SensitiveContent|sensitive content|PROHIBITED_CONTENT|IMAGE_SAFETY|\bSAFETY\b|BLOCKLIST|SPII|blocked by|内容安全|敏感/i;
  const categories = [
    [/minor|child|CSAM/i, '涉及未成年人'], [/sexual|SEXUALLY_EXPLICIT|nudity|porn|色情/i, '性感/暴露内容'],
    [/violen|gore|血腥|暴力/i, '暴力或血腥'], [/self-?harm/i, '自残'], [/hate|HATE_SPEECH/i, '仇恨'],
    [/harass/i, '骚扰'], [/illicit|DANGEROUS/i, '危险或违法内容'], [/SPII|personal/i, '个人隐私信息'],
    [/copyright|RECITATION|trademark/i, '版权内容']
  ];

  function safetyError(message, data, requestId, key) {
    const text = message + ' ' + JSON.stringify(data || '');
    if (!safetyReason.test(text)) return null;
    const error = Error(redact(message || 'safety', key));
    error.safe = {
      cats: categories.filter(([match]) => match.test(text)).map(([, label]) => label),
      where: /InputImage|input image/i.test(text) ? 'in' : /OutputImage|output/i.test(text) ? 'out' : '',
      rid: redact(requestId || text.match(/req_[A-Za-z0-9]+/)?.[0] || '', key)
    };
    error.raw = redact(message.trim(), key) || redact(text, key).slice(0, 600);
    error.zh = '被服务商的安全审核拦下了';
    return error;
  }

  const statusMessages = {
    400: '请求参数有误：模型、画幅或图片格式可能不受支持',
    401: '密钥无效或已过期，请检查 API Key',
    403: '没有权限：密钥无此模型权限、账户未验证，或所在地区不受支持',
    404: '地址或模型不存在：请检查 Base URL 和模型名',
    413: '图片太大，请换一张更小的照片',
    429: '请求过于频繁或额度已用完，请稍后再试或检查账户余额',
    500: '服务商内部错误，请稍后重试', 502: '网关错误（中转服务不可用）',
    503: '服务暂时不可用，请稍后重试', 504: '服务商响应超时，请重试'
  };

  function apiError(reply, data, settings) {
    const detail = data?.error?.message || data?.message || reply.statusText;
    const message = redact(typeof detail === 'string' ? detail : JSON.stringify(detail), settings.key);
    const requestId = data?.error?.request_id || reply.headers.get('x-request-id');
    const blocked = safetyError(message, data, requestId, settings.key);
    if (blocked) {
      blocked.status = reply.status;
      return blocked;
    }
    let reason = statusMessages[reply.status] || '请求失败（HTTP ' + reply.status + '）';
    const unsupported = message.match(/Argument not supported:\s*(\w+)/i);
    if (/API_KEY_INVALID|API key not valid|invalid api key|Incorrect API key|AuthenticationError|InvalidApiKey/i.test(message + ' ' + JSON.stringify(data))) {
      reason = statusMessages[401];
    } else if (unsupported) {
      reason = '该服务商不支持 ' + unsupported[1] + ' 参数，请检查「接口协议」设置';
    }
    const error = Error(message);
    error.status = reply.status;
    error.zh = reason + (message ? '\n服务商返回：' + message.slice(0, 300) : '');
    return error;
  }

  function nice(error) {
    if (error.name === 'AbortError') return '已取消';
    if (error instanceof TypeError) return '无法连接到服务商：网络不通，或该地址不允许浏览器跨域访问（CORS）。\n可在 API 设置中开启「通过本站代理转发」。';
    return error.zh || error.message || String(error);
  }

  async function testConnection(settings, model, signal) {
    const gemini = protocol(settings) === 'gemini';
    const path = gemini ? '/models/' + encodeURIComponent(model) : '/models';
    const reply = await request(settings, path, { method: 'GET', signal });
    const data = await reply.json().catch(() => null);
    if (!reply.ok) {
      if (settings.prov === 'seedream' && reply.status === 404) {
        if (reply.headers.get('content-type')?.includes('text/html')) {
          throw Error('本站代理未生效（/api/proxy 返回了网页内容），请稍后再试或联系站长');
        }
        return '地址可达，方舟未提供模型列表。请通过生成验证密钥与模型。';
      }
      throw apiError(reply, data, settings);
    }
    if (gemini && (data?.name || data?.displayName)) return '连接成功：模型 ' + (data.displayName || model) + ' 可用';
    if (gemini || !Array.isArray(data?.data) || data.data.some(entry => typeof entry?.id !== 'string')) {
      throw Error('接口未返回模型信息，请检查 Base URL 与接口协议。');
    }
    const ids = data.data.map(entry => entry.id);
    let message = '连接成功：共 ' + ids.length + ' 个模型';
    if (ids.length) message += ids.includes(model) ? '，包含 ' + model : '，但未找到 ' + model + '（中转可能不列出，可直接尝试）';
    return message;
  }

  async function readResponse(job, path, body) {
    const options = { method: 'POST', body, signal: job.signal };
    if (!(body instanceof FormData)) {
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(body);
    }
    const reply = await request(job.settings, path, options);
    const data = await reply.json().catch(() => null);
    job.signal.throwIfAborted();
    if (!reply.ok) throw apiError(reply, data, job.settings);
    return data;
  }

  function base64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.slice(reader.result.indexOf(',') + 1));
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
  }

  async function dataURL(blob) {
    return 'data:' + (blob.type || 'image/jpeg') + ';base64,' + await base64(blob);
  }

  async function resultBlob(data, job) {
    const image = data?.data?.[0];
    if (image?.b64_json) return (await fetch('data:image/png;base64,' + image.b64_json)).blob();
    if (!image?.url) throw Error('服务商没有返回图片数据' + (data?.error?.message ? '：' + redact(data.error.message, job.settings.key) : ''));
    const { url } = image;
    let reply = await fetch(url, { signal: job.signal }).catch(() => null);
    if (!reply?.ok) reply = await fetch('/api/proxy', {
      signal: job.signal, headers: { 'x-target-url': url, 'x-proxy-kind': 'image' }
    }).catch(() => null);
    if (!reply?.ok) throw Error('结果图片下载失败，请直接打开：' + redact(url, job.settings.key));
    return reply.blob();
  }

  async function generateLocal(job) {
    const { images, prompt, ratio, checkpoint, signal } = job;
    const form = new FormData();
    form.set('prompt', prompt);
    if (checkpoint) form.set('checkpoint', checkpoint);
    if (ratio !== 'auto') {
      const [width, height] = ratio.split(':').map(Number);
      const unit = 1024 / Math.max(width, height);
      const pixels = n => Math.round(n * unit / 8) * 8;
      form.set('size', pixels(width) + 'x' + pixels(height));
    }
    images.forEach((image, i) => form.append('image[]', image, 'reference-' + (i + 1) + '.jpg'));
    const reply = await fetch('/api/local/generate', { method: 'POST', headers: { 'x-posewear-local': '1' }, body: form, signal });
    if (!reply.ok) throw Error(await reply.text());
    return { blob: await reply.blob() };
  }

  async function generateGemini(job) {
    const parts = [{ text: job.prompt }];
    for (const image of job.images) {
      parts.push({ inline_data: { mime_type: image.type || 'image/jpeg', data: await base64(image) } });
    }
    const generationConfig = { responseModalities: ['TEXT', 'IMAGE'] };
    if (job.ratio !== 'auto') generationConfig.imageConfig = { aspectRatio: job.ratio };
    const data = await readResponse(job, '/models/' + encodeURIComponent(job.model) + ':generateContent', {
      contents: [{ role: 'user', parts }], generationConfig
    });
    const candidates = data?.candidates || [];
    const output = candidates.flatMap(candidate => candidate.content?.parts || []);
    const image = output.map(part => part.inlineData || part.inline_data).find(part => part?.data);
    if (image) return { blob: await (await fetch('data:' + (image.mimeType || image.mime_type || 'image/png') + ';base64,' + image.data)).blob() };

    const candidate = candidates[0];
    const reason = data?.promptFeedback?.blockReason || candidate?.finishReason;
    if (reason && /SAFETY|PROHIBITED|BLOCKLIST|SPII|OTHER/i.test(reason)) {
      const ratings = (candidate?.safetyRatings || []).filter(rating => rating.blocked || /HIGH|MEDIUM/.test(rating.probability));
      const detail = [reason, ...ratings.map(rating => rating.category), candidate?.finishMessage].filter(Boolean).join(' ');
      const blocked = safetyError(detail, null, data?.responseId, job.settings.key);
      if (blocked) throw blocked;
    }
    const text = redact(output.map(part => part.text).filter(Boolean).join(' '), job.settings.key);
    throw Error('模型没有返回图片' + (reason ? '（' + reason + '，可能触发了安全策略）' : '') + (text ? '\n模型回复：' + text.slice(0, 300) : ''));
  }

  async function generations(job, params, images = job.images) {
    const body = { model: job.model, prompt: job.prompt, n: 1, ...params };
    if (images.length) body.image = await Promise.all(images.map(dataURL));
    return resultBlob(await readResponse(job, '/images/generations', body), job);
  }

  async function generateSeedream(job) {
    const params = { size: seedreamSizes[job.ratio] || seedreamSizes.auto, response_format: 'b64_json', watermark: false };
    if (/seedream-4/.test(job.model)) params.sequential_image_generation = 'disabled';
    try {
      return { blob: await generations(job, params) };
    } catch (error) {
      if (!job.images.length || error.safe || error.status !== 400 || !/image|参考/i.test(error.message)) throw error;
      return { blob: await generations(job, params, []), warn: textOnlyWarning };
    }
  }

  async function generateXai(job) {
    const body = { model: job.model, prompt: job.prompt, n: 1, response_format: 'b64_json' };
    if (job.ratio !== 'auto') body.aspect_ratio = job.ratio;
    const urls = await Promise.all(job.images.slice(0, 5).map(dataURL));
    if (urls.length === 1) body.image = { url: urls[0] };
    if (urls.length > 1) body.images = urls.map(url => ({ url }));
    const path = urls.length ? '/images/edits' : '/images/generations';
    return {
      blob: await resultBlob(await readResponse(job, path, body), job),
      warn: job.images.length > 5 ? 'xAI 参考图最多 5 张，已只发送前 5 张。' : ''
    };
  }

  async function edits(job, size, field) {
    const form = new FormData();
    const single = job.model === 'dall-e-2';
    form.set('model', job.model);
    form.set('prompt', job.prompt);
    const images = single ? job.images.slice(0, 1) : job.images;
    images.forEach((image, i) => form.append(field, image, 'ref-' + (i + 1) + '.jpg'));
    form.set('n', '1');
    form.set('size', single && size === 'auto' ? '1024x1024' : size);
    return resultBlob(await readResponse(job, '/images/edits', form), job);
  }

  async function generateOpenAI(job) {
    const size = imageSizes[job.ratio] || (/^gpt-image/.test(job.model) ? 'auto' : '1024x1024');
    if (protocol(job.settings) === 'json' || !job.images.length) return { blob: await generations(job, { size }) };
    const field = job.settings.prov === 'custom' ? job.settings.field : 'image[]';
    try {
      return { blob: await edits(job, size, field) };
    } catch (error) {
      // 只对「图片字段名不对」这一种已知情况换字段重试一次；其余错误直接报出，不静默换协议或丢参考图
      if (error.name === 'AbortError' || error.safe || error.status !== 400 || !/image/i.test(error.message) || /safety|moderation|reject/i.test(error.message)) throw error;
      const alternate = field === 'image[]' ? 'image' : 'image[]';
      return { blob: await edits(job, size, alternate), warn: '已自动改用另一种图片字段名重试成功。' };
    }
  }

  function generateImage(job) {
    switch (protocol(job.settings)) {
      case 'local': return generateLocal(job);
      case 'gemini': return generateGemini(job);
      case 'seedream': return generateSeedream(job);
      case 'xai': return generateXai(job);
      default: return generateOpenAI(job);
    }
  }

  return { providers, generateImage, testConnection, nice };
})();
