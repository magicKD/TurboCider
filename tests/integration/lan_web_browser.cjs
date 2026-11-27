'use strict';

// Run against turbocider-lan-web-tests --serve with a real Qwen Turbo install.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { chromium, webkit, devices } = require(process.env.TURBOCIDER_PLAYWRIGHT || 'playwright-core');

const [state, assets] = process.argv.slice(2);
const appID = 'org.turbocider.people-studio';
const done = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
const sample = name => path.join(assets, 'samples', name);
const read = name => fs.readFile(path.join(state, name), 'utf8').then(JSON.parse);
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function waitFor(readValue, ready, label, seconds = 180) {
  const until = Date.now() + seconds * 1000;
  while (Date.now() < until) {
    const value = await readValue();
    if (ready(value)) return value;
    await delay(300);
  }
  throw new Error('Timed out waiting for ' + label);
}

async function pair(page, link, name) {
  await page.goto(link);
  await page.locator('#device-name').fill('Acceptance ' + name);
  await page.locator('#pair-form button').click();
  await page.locator('#app-list a').waitFor({ timeout: 15_000 });
}

async function checkLayout(page, label) {
  const widths = await page.evaluate(() => [document.documentElement.clientWidth, document.documentElement.scrollWidth]);
  assert.ok(widths[1] <= widths[0] + 1, label + ' has horizontal overflow: ' + widths);
}

async function openStudio(page) {
  await page.locator('#app-list a').click();
  await page.waitForFunction(() => window.TurboCider?.platform === 'browser' && window.APP);
  await page.locator('#top [data-tab=studio]').click();
  await page.locator('#studio').waitFor();
}

async function startJob(page, workflow) {
  const submitting = page.waitForResponse(response => {
    const request = response.request();
    return request.method() === 'POST' && new URL(request.url()).pathname === '/api/call'
      && request.postDataJSON().method === 'jobs.submit';
  }, { timeout: 45_000 });
  await page.locator('#wbGo').click();
  const reply = await submitting;
  const jobs = await reply.json();
  assert.equal(reply.status(), 200, JSON.stringify(jobs));
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].workflow, workflow);
  return jobs[0].id;
}

async function readJobs() {
  try { return await read('jobs.json'); }
  catch (error) { if (error.code === 'ENOENT') return []; throw error; }
}

async function submitStudio(page, label) {
  const pending = page.waitForResponse(response => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === '/api/studio/submit', { timeout: 45_000 });
  await page.getByRole('button', { name: label }).click();
  const response = await pending;
  const value = await response.json();
  assert.equal(response.status(), 201, JSON.stringify(value));
  return value[0].id;
}

async function completed(id, label, timeout) {
  const job = await waitFor(async () => (await readJobs()).find(row => row.id === id),
    row => row && done.has(row.state), label, timeout);
  assert.equal(job.state, 'succeeded', job.error);
  return job;
}

async function communityApp(config, browser) {
  const manager = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const latest = await waitFor(() => read('browser-test.json'), value => value.pair !== config.pair, 'new pairing link');
  await pair(manager, latest.pair, 'Manager');
  const bundle = path.resolve(state, 'sample.tcapp');
  execFileSync('/usr/bin/zip', ['-q', bundle, 'index.html', 'manifest.json', 'app.css', 'main.js'], {
    cwd: path.resolve(__dirname, '../../examples/mini-apps/image-sample')
  });
  await manager.locator('#app-admin input').setInputFiles(bundle);
  await manager.getByRole('heading', { name: 'Image Sample' }).waitFor();
  await manager.getByRole('link', { name: '打开应用 ↗', exact: true }).click();
  const child = await waitFor(() => manager.frames().find(frame => new URL(frame.url()).pathname.startsWith('/_frame/')),
    frame => !!frame, 'sandbox document');
  await child.waitForFunction(() => window.TurboCider?.sdkVersion === 1);
  const negotiated = await child.evaluate(() => window.TurboCider.call('handshake'));
  assert.equal(negotiated.appID, 'dev.turbocider.image-sample');
  assert.equal(await child.evaluate(() => {
    try { return parent.document.title; } catch (error) { return error.name; }
  }), 'SecurityError');
  assert.equal(await child.evaluate(() => fetch('/api/studio/catalog').then(() => false, () => true)), true);
  await child.locator('input[type=file]').setInputFiles(sample('demo-person.webp'));
  const stored = await child.evaluate(async () => {
    const record = await TurboCider.media.import(document.querySelector('input[type=file]').files[0]);
    const image = new Image();
    image.src = record.url;
    await image.decode();
    return [record.url.startsWith('blob:'), image.naturalWidth > 0];
  });
  assert.deepEqual(stored, [true, true]);
  console.log('Authorized package install, opaque-origin SDK, private image import, and package isolation passed');
  await manager.close();
}

async function launchWorkspace(config, safari, browser) {
  const marker = path.join(state, 'direct-web');
  await fs.writeFile(marker, '');
  await waitFor(() => fetch(new URL('/api/session', config.web)).then(reply => reply.json()), value => value.directAvailable, 'direct access toggle');
  const narrow = await browser.newPage({ ...devices['Pixel 7'], viewport: { width: 320, height: 700 } });
  await narrow.goto(config.web);
  await narrow.locator('#create .sample-caption').waitFor();
  await checkLayout(narrow, '320px touch navigation and creation');
  await narrow.locator('[data-nav=apps]').click();
  await narrow.locator('.look-result img').evaluate(image => image.decode());
  await checkLayout(narrow, '320px PoseWear contact sheet');
  await narrow.close();
  const phone = await safari.newPage({ ...devices['iPhone 13'], acceptDownloads: true });
  await phone.goto(config.web);
  await phone.locator('#create form').waitFor();
  await checkLayout(phone, 'iPhone full creation workspace');
  const example = phone.getByRole('button', { name: '使用示例提示词' });
  const prompt = phone.locator('#create textarea');
  await prompt.fill('保留这段草稿。');
  await example.click();
  const combined = await prompt.inputValue();
  assert.ok(combined.startsWith('保留这段草稿。\n') && combined.includes('琥珀'));
  await example.click();
  assert.equal(await prompt.inputValue(), combined);
  await phone.screenshot({ path: path.join(state, 'web-iphone-create.png'), fullPage: true });
  await phone.locator('[name="_model"]').selectOption('qwen-image-2.1');
  await phone.locator('[name="_operation"]').selectOption('image.edit');
  await phone.locator('#create input[type=file]').setInputFiles([sample('demo-person.webp'), sample('demo-outfit.webp')]);
  await phone.locator('#create .reference').nth(1).waitFor();
  await phone.getByRole('button', { name: '向后移动参考图' }).first().click();
  await phone.locator('#create textarea').fill('Put the clothes from image 1 on the person from image 2. Fashion studio photography.');
  const portrait = phone.getByRole('button', { name: '长竖图', exact: true });
  await portrait.click();
  assert.equal(await portrait.getAttribute('aria-pressed'), 'true');
  await phone.locator('#create [name=width]').fill('400');
  assert.equal(await portrait.getAttribute('aria-pressed'), 'false');
  await portrait.click();
  const id = await submitStudio(phone, '开始生成 ↗');
  const job = await completed(id, 'browser creative image');
  assert.equal(job.request.qwen21_reference_pixel_budget, 512 * 512);
  assert.equal(job.request.residency, 'streamed');
  assert.equal(job.request.width, 384);
  assert.equal(job.request.height, 768);
  await phone.locator('[data-nav=library]').click();
  await phone.getByRole('button', { name: '查看 参考编辑' }).waitFor();
  await phone.getByRole('button', { name: '查看 参考编辑' }).click();
  await phone.getByRole('button', { name: '查看原始输入' }).waitFor();
  await phone.getByRole('button', { name: '关闭' }).click();

  const desktop = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await desktop.goto(config.web);
  await desktop.locator('#create form').waitFor();
  await desktop.screenshot({ path: path.join(state, 'web-desktop-create.png'), fullPage: true });
  await desktop.locator('[data-nav=apps]').click();
  await desktop.getByRole('link', { name: '打开人物工作室 ↗' }).waitFor();
  const pose = desktop.getByRole('button', { name: '换姿势', exact: true });
  await pose.click();
  await desktop.locator('.look-result img').evaluate(image => image.decode());
  assert.equal(await pose.getAttribute('aria-pressed'), 'true');
  assert.ok((await desktop.locator('.look-result img').getAttribute('src')).endsWith('look-pose.webp'));
  await desktop.screenshot({ path: path.join(state, 'web-desktop-apps.png'), fullPage: true });
  await desktop.getByRole('link', { name: '打开模板' }).first().click();
  await desktop.locator('#template form input[type=file]').first().waitFor();
  await checkLayout(desktop, 'desktop template');
  await desktop.locator('[data-nav=jobs]').click();
  await desktop.locator('#job-status').waitFor();
  assert.equal(await desktop.locator('#job-list li').count(), 0);
  await desktop.locator('[data-nav=models]').click();
  await desktop.getByRole('heading', { name: '模型中心' }).waitFor();
  await desktop.getByRole('searchbox').fill('does-not-match-any-model');
  await desktop.getByRole('button', { name: '清除搜索' }).click();
  assert.equal(await desktop.getByRole('searchbox').inputValue(), '');
  assert.equal(await desktop.getByRole('button', { name: '下载到 Mac' }).count(), 0);
  const video = await phone.evaluate(async name => {
    const file = document.querySelectorAll('#library input[type=file]')[0];
    return !!file && name;
  }, 'clip');
  assert.equal(video, 'clip');
  await phone.locator('#library input[type=file]').setInputFiles(path.join(state, 'clip.mp4'));
  await phone.getByRole('button', { name: '查看 clip.mp4' }).waitFor();
  await phone.getByRole('button', { name: '查看 clip.mp4' }).click();
  await phone.waitForFunction(() => document.querySelector('#media-dialog video')?.readyState >= 1, null, { timeout: 15_000 });
  const duration = await phone.locator('#media-dialog video').evaluate(element => element.duration);
  assert.ok(duration > 0 && duration < 1);
  await phone.getByRole('button', { name: '关闭' }).click();
  console.log('Direct phone creation, private galleries, ordered references, model catalog and real Safari video playback passed');

  if (process.env.TURBOCIDER_WEB_TEST_RELAY) {
    await phone.locator('[data-nav=upscale]').click();
    await phone.locator('#upscale [name="_model"]').selectOption('relayvsr-2x');
    await phone.locator('#upscale [name="_operation"]').selectOption('video.upscale');
    await phone.locator('#upscale input[type=file]').setInputFiles(path.join(state, 'clip.mp4'));
    await phone.locator('#upscale .reference').waitFor();
    const upscale = await completed(await submitStudio(phone, '提交超分 ↗'), 'RelayVSR video upscale', 180);
    assert.equal(upscale.request.width, 128);
    assert.equal(upscale.request.height, 128);
    console.log('Streaming RelayVSR video upscale completed through the Web queue');
  }
  if (process.env.TURBOCIDER_WEB_TEST_LTX) {
    await desktop.locator('[data-nav=create]').click();
    await desktop.locator('#create [name="_model"]').selectOption('ltx-2.5-distilled');
    await desktop.locator('#create [name="_operation"]').selectOption('video.generate');
    await desktop.locator('#create textarea').fill('A slow cinematic tilt over green leaves on a small houseplant with bright window light.');
    for (const [name, value] of [['width', '256'], ['height', '256'], ['frames', '9']]) {
      await desktop.locator('#create [name="' + name + '"]').fill(value);
    }
    const audio = desktop.getByLabel('输出音频');
    if (await audio.count()) await audio.uncheck();
    const ltx = await completed(await submitStudio(desktop, '开始生成 ↗'), 'real LTX video', 240);
    assert.equal(ltx.request.frames, 9);
    console.log('LTX generated a real MP4 via browser controls');
  }

  await fs.rm(marker);
  await phone.locator('#pair-status').waitFor({ timeout: 10_000 });
  await desktop.locator('#pair-status').waitFor({ timeout: 10_000 });
  await phone.close(); await desktop.close();
}

async function main() {
  const config = await read('browser-test.json');
  const safari = await webkit.launch({ executablePath: process.env.TURBOCIDER_WEBKIT || undefined });
  let desktopBrowser;
  try {
    desktopBrowser = await chromium.launch({ executablePath: process.env.TURBOCIDER_CHROME || undefined });
    const iphone = await safari.newContext({ ...devices['iPhone 13'], acceptDownloads: true });
    const phone = await iphone.newPage();
    await pair(phone, config.pair, 'iPhone');
    await checkLayout(phone, 'iPhone app list');
    await openStudio(phone);
    await checkLayout(phone, 'iPhone portrait workspace');
    await phone.locator('#wbFile').setInputFiles(sample('demo-person.webp'));
    await phone.locator('#wbPersons img').first().waitFor();
    await phone.locator('#wbGFile').setInputFiles(sample('demo-outfit.webp'));
    await phone.locator('#wbG .pick').first().waitFor();
    await phone.locator('#wbAR').selectOption('1:2');
    const outfitID = await startJob(phone, 'people.outfit');
    await waitFor(readJobs, jobs => jobs.some(job => job.id === outfitID && !done.has(job.state)), 'active outfit');

    // The harness invokes the SwiftUI submission entry point against this same
    // NativeJobStore while the actual browser's request is executing.
    const registration = {
      installation_id: config.install,
      request: { model_variant: 'qwen-image-2.1-turbo', prompt: 'A red apple on a white plate, bright daylight',
        width: 384, height: 384, steps: 8, residency: 'streamed', allow_approximation: true }
    };
    const [api] = await Promise.all([
      fetch(new URL('/v1/jobs', config.api), { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(registration) }),
      fs.writeFile(path.join(state, 'submit-desktop'), '')
    ]);
    const apiJob = await api.json();
    assert.equal(api.status, 202, JSON.stringify(apiJob));
    await waitFor(readJobs, jobs => jobs.filter(job => !done.has(job.state)).length === 3, 'three overlapping submissions');
    console.log('Phone, desktop and HTTP API accepted concurrently');

    // Reopening the site with the same browser session must restore the task.
    await phone.goto(new URL('/#jobs', config.web).href);
    await phone.locator('#job-list li').first().waitFor();
    assert.equal(await phone.locator('#job-list li').count(), 1);
    await waitFor(readJobs, jobs => jobs.length === 3 && jobs.every(job => done.has(job.state)), 'three terminal jobs');
    const results = await readJobs();
    assert.ok(results.every(job => job.state === 'succeeded'), JSON.stringify(results.map(job => [job.state, job.error])));
    assert.ok(results.some(job => job.id === apiJob.id && job.automationRequestJSON));
    assert.ok(results.some(job => job.request.output.endsWith('/desktop.png')));
    await phone.locator('#job-list [data-state=succeeded]').waitFor();
    await phone.locator('#job-filters [data-filter=done]').click();
    await phone.locator('#job-list .job-thumb img').evaluate(image => image.decode());
    await phone.screenshot({ path: path.join(state, 'web-iphone-jobs.png'), scale: 'css', animations: 'disabled' });
    await phone.locator('#job-filters [data-filter=active]').click();
    await phone.getByRole('button', { name: '查看全部任务' }).click();
    const downloadLink = phone.locator('#job-list a[download]');
    await downloadLink.waitFor();
    const downloaded = phone.waitForEvent('download');
    await downloadLink.click();
    await (await downloaded).saveAs(path.join(state, 'browser-outfit.png'));
    const apiArtifact = await fetch(new URL('/v1/jobs/' + apiJob.id + '/artifact', config.api));
    assert.equal(apiArtifact.status, 200);
    assert.equal(apiArtifact.headers.get('content-type'), 'image/png');
    console.log('Outfit, desktop and local API all produced real images, including browser and API downloads');

    const cookie = await iphone.cookies();
    const tablet = await safari.newContext({ ...devices['iPad (gen 7)'] });
    await tablet.addCookies(cookie);
    const pad = await tablet.newPage();
    await pad.goto(new URL('/#apps', config.web).href);
    await pad.locator('#app-list a').waitFor();
    await checkLayout(pad, 'iPad app list');
    await openStudio(pad);
    await checkLayout(pad, 'iPad portrait workspace');
    await pad.setViewportSize({ width: 1080, height: 810 });
    await checkLayout(pad, 'iPad landscape workspace');

    const android = await desktopBrowser.newContext({ ...devices['Pixel 7'] });
    await android.addCookies(cookie);
    const pixel = await android.newPage();
    await pixel.goto(new URL('/#apps', config.web).href);
    await pixel.locator('#app-list a').waitFor();
    await checkLayout(pixel, 'Android app list');
    await openStudio(pixel);
    await checkLayout(pixel, 'Android workspace');

    const computer = await desktopBrowser.newContext({ viewport: { width: 1440, height: 900 } });
    await computer.addCookies(cookie);
    const pc = await computer.newPage();
    await pc.goto(new URL('/#apps', config.web).href);
    await pc.locator('#app-list a').waitFor();
    await checkLayout(pc, 'desktop app list');
    await openStudio(pc);
    await checkLayout(pc, 'desktop workspace');
    await pc.waitForFunction(() => ['wbClearAll', 'wbExample'].some(id => {
      const button = document.getElementById(id);
      return !button.hidden && button.getClientRects().length;
    }));
    if (await pc.locator('#wbClearAll').isVisible()) await pc.locator('#wbClearAll').click();
    await pc.locator('#wbExample').click({ timeout: 25_000 });
    await pc.locator('#wbPo .pick').first().waitFor({ timeout: 35_000 });
    await pc.locator('#wbG [data-rm=costume]').first().click();
    const poseID = await startJob(pc, 'people.pose');
    const pose = await waitFor(async () => (await readJobs()).find(job => job.id === poseID),
      job => job && done.has(job.state), 'pose result');
    assert.equal(pose.state, 'succeeded', pose.error);
    const poseFile = JSON.parse(pose.resultJSON).output;
    assert.ok((await fs.stat(poseFile)).size > 0);
    assert.equal(pose.request.steps, 8);
    assert.equal(pose.request.qwen21_reference_pixel_budget, 262144);
    await pc.locator('#theme').click();
    await pc.locator('[data-theme-choice=dark]').click();
    await pc.getByRole('link', { name: '返回 TurboCider 应用页' }).click();
    await pc.locator('#app-list a').waitFor();
    assert.equal(await pc.locator('html').getAttribute('data-theme'), 'dark');
    console.log('iPhone, iPad portrait/landscape, Android and desktop layout passed. Pose edit produced a real image.');
    await communityApp(config, desktopBrowser);
    await launchWorkspace(config, safari, desktopBrowser);
  } finally {
    await desktopBrowser?.close();
    await safari.close();
    await fs.writeFile(path.join(state, 'stop-test'), '');
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
