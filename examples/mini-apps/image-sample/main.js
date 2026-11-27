'use strict';
const host = window.TurboCider;
const form = document.querySelector('form');
const status = document.querySelector('output');
const output = document.querySelector('main img');
const start = document.querySelector('#start');
const cancel = document.querySelector('#cancel');
const save = document.querySelector('#save');
let active;

cancel.onclick = () => host.jobs.cancel(active).catch(error => { status.textContent = error.message; });
form.onsubmit = async event => {
  event.preventDefault();
  start.disabled = true;
  save.hidden = output.hidden = true;
  try {
    const file = form.querySelector('input').files[0];
    const references = file ? [{ role: 'reference', mediaID: (await host.media.import(file)).id }] : [];
    const plan = await host.workflow.plan({
      requestID: host.createRequestID(), engine: 'qwen-image-2.1-turbo', workflow: 'example.image',
      shots: [{ prompt: form.querySelector('textarea').value, references }]
    });
    let [job] = await host.jobs.submit(plan.id);
    active = job.id;
    cancel.disabled = false;
    for await (const state of host.jobs.watch(active)) {
      job = state;
      status.textContent = `${job.phase} · ${job.completed}/${job.total}`;
    }
    status.textContent = job.error || job.state;
    if (job.output) {
      output.src = job.output.url;
      output.hidden = save.hidden = false;
      save.onclick = () => host.media.save(job.output.id);
    }
  } catch (error) { status.textContent = error.message; }
  finally { cancel.disabled = true; active = null; start.disabled = false; }
};
