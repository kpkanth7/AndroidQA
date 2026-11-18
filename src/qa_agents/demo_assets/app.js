const byId = id => document.getElementById(id);
const form = byId('run-form');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const home = {screen: 'home', ui_tree: {buttons: ['Settings']}, info: {}};
const screenNames = {home: 'Home', settings: 'Settings', network: 'Network & Internet', display: 'Display', apps: 'Apps', camera: 'Camera'};
const tasks = {
  wifi: ['Wi-Fi', 'Wi-Fi turned off and back on.'],
  bluetooth: ['Bluetooth', 'Bluetooth is on.'],
  dark_mode: ['dark mode', 'Dark mode is enabled.'],
  camera_permission: ['camera access', 'Camera access is allowed.'],
};
const sources = {
  rule: 'Rule-based', rule_replan: 'Rule recovery', llm: 'LLM',
  llm_with_rule_fallback: 'LLM + rule fallback', unsupported_goal: 'Unsupported goal',
  rule_fallback_invalid_or_unavailable: 'Rule fallback',
};
let steps = [];
let position = 0;
let timer;
let busy = false;
let configured = false;
let recording;
const images = new Map();

function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value != null) node.textContent = value;
  if (className) node.className = className;
  return node;
}

function selectedTask() { return form.elements.scenario.value; }
function normalize(value) { return (value || '').toLowerCase().replace(/[-‑ ]/g, ''); }

function drawPhone(step) {
  const display = byId('display');
  if (step.image) {
    display.className = 'display';
    display.replaceChildren(images.get(step.image));
    return;
  }
  const {screen, ui_tree: tree, info} = step.observation;
  display.className = 'display ' + screen + (info.dark_mode ? ' dark' : '');
  const statusBar = element('div', 'Android', 'system');
  display.replaceChildren(statusBar, element('h3', screenNames[screen] || screen, 'screen-title'));
  function addRow(label, enabled) {
    const row = element('div', null, 'device-row');
    const name = element('span', label);
    row.append(name);
    if (normalize(label) === normalize(step.target)) row.classList.add('aim');
    if (typeof enabled === 'boolean') {
      name.append(element('small', enabled ? 'On' : 'Off', 'state-label'));
      const toggle = element('span', null, enabled ? 'switch on' : 'switch');
      if (step.changed?.includes(label)) toggle.classList.add('changed');
      toggle.setAttribute('aria-hidden', 'true');
      row.append(toggle);
    }
    display.append(row);
  }
  for (const label of [...(tree.buttons || []), ...(tree.list || [])]) addRow(label);
  for (const [label, enabled] of Object.entries(tree.toggles || {})) addRow(label, enabled);
  display.append(element('span', null, 'gesture'));
}

function status(value) {
  byId('status').textContent = value;
  byId('status').dataset.state = value.toLowerCase();
}

function pause() {
  clearTimeout(timer);
  timer = undefined;
  byId('play').textContent = 'Play';
}

function showStep() {
  const step = steps[position];
  const last = position === steps.length - 1;
  if (last) pause();
  drawPhone(step);
  byId('phase').textContent = step.role + ' · ' + (position + 1) + ' of ' + steps.length;
  byId('caption').textContent = step.caption;
  byId('play').disabled = last;
  byId('next').disabled = last;
  byId('result').hidden = !last;
  status(step.verdict || 'Playback');
}

function play() {
  pause();
  if (position === steps.length - 1) return;
  byId('play').textContent = 'Pause';
  timer = setTimeout(() => {
    position++;
    showStep();
    if (position < steps.length - 1) play();
  }, steps[position].delay || 1550);
}

function start(sequence) {
  pause();
  steps = sequence;
  position = 0;
  byId('controls').hidden = false;
  showStep();
  if (!reducedMotion.matches) play();
}

byId('play').onclick = () => timer === undefined ? play() : pause();
byId('next').onclick = () => { pause(); position++; showStep(); };
byId('replay').onclick = () => start(steps);
reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) pause(); });

function preview() {
  pause();
  steps = [];
  byId('controls').hidden = byId('result').hidden = byId('error').hidden = true;
  byId('goal').textContent = form.querySelector('input[name=scenario]:checked + span').textContent;
  byId('run').textContent = 'Run ' + tasks[selectedTask()][0] + ' test';
  byId('mode').textContent = 'Android simulation';
  byId('phase').textContent = 'Simulated phone';
  byId('caption').textContent = 'Run the test to follow each action.';
  if (configured) {
    byId('planner-note').textContent = byId('planner').value === 'llm'
      ? 'Live LLM planner · uses the server’s key' : 'Rule-based simulation · 0 model calls';
  }
  status('Ready');
  drawPhone({observation: home});
}

function actionLabel(p) {
  if (p.intent === 'open_settings' || p.intent === 'open_app') return 'Open Settings';
  if (p.intent === 'toggle') return 'Switch ' + p.target + ' ' + (p.params.state || '');
  if (p.intent === 'tap') return 'Tap ' + p.target;
  if (p.intent === 'verify') return 'Check ' + (p.target || 'the goal');
  return p.intent;
}

function timeline(events, scenario) {
  const get = name => events.find(e => e.event === name)?.payload;
  const meta = get('run_metadata');
  let observation = {screen: meta.mock_initial_state.screen, info: meta.mock_initial_state, ui_tree: meta.mock_initial_ui_tree || home.ui_tree};
  const sequence = [];
  const add = (role, caption, extra = {}) => sequence.push({role, caption, observation, ...extra});
  add('Planner', get('plan').goal, {delay: 850});
  let lastCheck;
  for (const {event, payload: p} of events) {
    if (event === 'action') {
      const touches = ['tap', 'toggle', 'open_settings', 'open_app'].includes(p.intent);
      if (touches) add('Executor', actionLabel(p), {target: p.target || 'Settings', delay: 780});
      const previous = observation;
      observation = p.observation;
      if (p.intent === 'verify') continue;
      const toggles = observation.ui_tree.toggles || {};
      const changed = Object.keys(toggles).filter(k => typeof previous.ui_tree.toggles?.[k] === 'boolean' && previous.ui_tree.toggles[k] !== toggles[k]);
      let caption = p.ok ? p.message : 'Action failed. ' + p.message;
      if (p.intent === 'toggle') {
        const key = Object.keys(toggles).find(k => normalize(k) === normalize(p.target));
        if (key) caption = (p.ok ? key + ' is ' : 'Switch failed. ' + key + ' is ') + (toggles[key] ? 'on.' : 'off.');
      }
      add('Executor', caption, {changed});
    } else if (event === 'verify') {
      lastCheck = p;
      if (!p.passed) {
        add('Verifier', 'Check failed. ' + p.reason);
      } else if (p.reason !== 'No specific rule') {
        const toggles = observation.ui_tree.toggles || {};
        const target = Object.keys(toggles).find(label => p.reason.includes(label));
        add('Verifier', target ? 'Confirmed ' + target + ' is ' + (toggles[target] ? 'on.' : 'off.') : p.reason);
      }
    } else if (event === 'replan') {
      add('Supervisor', 'Recovery requested. Retry from the current screen.');
    } else if (event === 'plan_fallback' || event === 'plan_rejection') {
      add('Planner', p.reason);
    } else if (event === 'finish') {
      const passed = p.status === 'passed';
      const recoveries = get('run_metrics').replan_count;
      let caption = p.reason || 'Run ended.';
      if (passed && lastCheck?.passed) {
        caption = tasks[scenario][1] + (recoveries ? ' Verified after ' + recoveries + (recoveries === 1 ? ' recovery.' : ' recoveries.') : ' Verified.');
      }
      add('Supervisor', caption, {verdict: passed ? 'Passed' : 'Failed'});
    }
  }
  return sequence;
}

function metrics(items) {
  byId('metrics').replaceChildren(...items.map(([label, value]) => {
    const group = element('div');
    group.append(element('dt', label), element('dd', value));
    return group;
  }));
}

function renderRun(data, scenario) {
  const get = name => data.events.find(e => e.event === name)?.payload;
  const counts = get('run_metrics');
  const plan = get('plan');
  const client = get('run_metadata').clients[0];
  metrics([
    ['Planner', (sources[plan.source] || plan.source) + (client ? ' · ' + client.model : '')],
    ['Runner duration', counts.end_to_end_latency_ms + ' ms'],
    ['Role handoffs', counts.agent_message_count], ['Replans', counts.replan_count],
  ]);
  byId('trace').replaceChildren(...data.events.map(({event, payload}) => {
    const li = element('li');
    const detail = element('details');
    const label = payload.kind || payload.message || payload.reason || payload.operation || payload.status || '';
    detail.append(element('summary', event.replaceAll('_', ' ') + (label ? ' · ' + label : '')), element('pre', JSON.stringify(payload, null, 2)));
    li.append(detail);
    return li;
  }));
  byId('report').textContent = data.report;
  byId('report-panel').hidden = false;
  byId('mode').textContent = 'Simulation · Step-by-step playback';
  start(timeline(data.events, scenario));
}

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || 'Unable to load the demo. Try again.');
  }
  return response.json();
}

function lock(value) {
  busy = value;
  byId('inputs').disabled = value;
  byId('watch').disabled = value;
  byId('run').disabled = value || !configured;
}

function failure(error) {
  pause();
  byId('controls').hidden = byId('result').hidden = true;
  byId('error').textContent = error.message;
  byId('error').hidden = false;
  byId('phase').textContent = 'Request failed';
  byId('caption').textContent = 'Try again when ready.';
  status('Error');
}

form.onchange = preview;
form.onsubmit = async event => {
  event.preventDefault();
  if (busy || !configured) return;
  preview();
  lock(true);
  status('Running');
  byId('caption').textContent = 'Executing the selected test…';
  revealViewer();
  const scenario = selectedTask();
  try {
    const data = await json('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({scenario, planner: byId('planner').value, inject_failure: byId('failure').checked})});
    renderRun(data, scenario);
  } catch (error) { failure(error); }
  finally { lock(false); }
};

function revealViewer() {
  if (matchMedia('(max-width: 740px)').matches) document.querySelector('.viewer').scrollIntoView({block: 'start'});
}

async function loadRecording() {
  const data = await json('/bluetooth-session.json');
  await Promise.all(data.steps.map(async step => {
    const image = new Image();
    image.className = 'frame';
    image.alt = 'Pixel 6 Bluetooth settings after ' + step.action + '. Bluetooth is on.';
    image.src = step.image;
    await image.decode();
    images.set(step.image, image);
  }));
  return data;
}

byId('watch').onclick = async () => {
  if (busy) return;
  preview();
  lock(true);
  byId('mode').textContent = 'Recorded Pixel 6 emulator · Android 13';
  byId('goal').textContent = 'Turn Bluetooth on';
  byId('phase').textContent = 'Recorded session';
  byId('caption').textContent = 'Loading the emulator recording…';
  status('Loading');
  revealViewer();
  try {
    recording ||= await loadRecording();
    const {model, provider, steps: recorded, duration_ms, handoffs, replans, verdict} = recording;
    const sequence = recorded.map((s, i) => ({
      role: 'Planner · recorded call ' + (i + 1), image: s.image,
      caption: s.action + '. ' + provider + ' / ' + model + ' · ' + s.latency_ms + ' ms. Android reports ' + s.manager_state + '.',
      delay: 2700,
    }));
    sequence.push({image: recorded.at(-1).image, role: 'Verifier', caption: 'Bluetooth verified on. Android’s Bluetooth manager reports ON.', verdict});
    metrics([['Planner', model], ['Recorded duration', duration_ms + ' ms'], ['Role handoffs', handoffs], ['Replans', replans]]);
    byId('trace').replaceChildren(...recorded.map(s => element('li', s.action + ' · ' + s.latency_ms + ' ms · ' + s.verification)));
    byId('report-panel').hidden = true;
    start(sequence);
  } catch (error) { failure(error); }
  finally { lock(false); }
};

async function init() {
  preview();
  try {
    const config = await json('/api/config');
    byId('planner-field').hidden = !config.planners.includes('llm');
    configured = true;
    byId('planner-note').textContent = 'Rule-based simulation · 0 model calls';
    byId('run').disabled = busy;
  } catch {
    byId('planner-note').textContent = 'Simulation unavailable. Reload to retry or watch the recording.';
  }
}
init();
