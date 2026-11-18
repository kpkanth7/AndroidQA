// Session history stores the runner's response so inspecting a run never submits it again.
const historyKey = 'androidqa.runs.v1';
const historyLimit = 12;
let recent = [];
try {
  const saved = JSON.parse(sessionStorage.getItem(historyKey) || '[]');
  if (Array.isArray(saved)) recent = saved.filter(r => Object.hasOwn(tasks, r?.scenario) &&
    typeof r.fault === 'boolean' && Array.isArray(r.data?.events) &&
    ['run_metrics', 'finish', 'plan', 'run_metadata'].every(name => r.data.events.some(e => e.event === name))
  ).slice(0, historyLimit);
} catch { /* Storage is optional in private browsing. */ }

function runPayload(entry, type) {
  return entry.data.events.find(e => e.event === type).payload;
}

function saveRun(entry) {
  recent.unshift(entry);
  recent = recent.slice(0, historyLimit);
  let persisted = false;
  try {
    sessionStorage.setItem(historyKey, JSON.stringify(recent));
    persisted = true;
  } catch { /* The current page can still inspect in-memory runs. */ }
  drawHistory();
  if (!persisted) byId('history-note').textContent = 'Browser storage is unavailable. Runs are available until this page reloads.';
}

function drawHistory() {
  if (!recent.length) return;
  byId('history-note').textContent = 'Latest 12 tests in this tab. Open a result to inspect or replay it.';
  byId('run-list').replaceChildren(...recent.map(entry => {
    const row = element('div', null, 'history-row');
    const meta = runPayload(entry, 'run_metrics');
    const source = runPayload(entry, 'plan').source;
    const title = tasks[entry.scenario][0] + ' · ' + (entry.fault ? 'First switch fails' : 'Normal run');
    const button = element('button', title);
    button.onclick = () => {
      if (busy) return;
      form.elements.scenario.value = entry.scenario;
      byId('failure').checked = entry.fault;
      byId('planner').value = byId('planner-field').hidden ? 'rule' : entry.planner;
      preview();
      showComparisons();
      renderRun(entry.data, entry.scenario, entry);
      pause();
      position = steps.length - 1;
      showStep();
      document.querySelector('#result').scrollIntoView({block:'start'});
    };
    const detail = [new Date(entry.at).toLocaleTimeString(), sources[source] || source,
      runPayload(entry, 'finish').status, meta.end_to_end_latency_ms + ' ms',
      meta.agent_message_count + ' handoffs', meta.replan_count + ' replans'];
    row.append(button, element('small', detail.join(' · ')));
    return row;
  }));
}

const plannerLabels = {'rules':'Rules', 'gpt-oss-20b':'GPT-OSS 20B', 'gpt-oss-120b':'GPT-OSS 120B'};
let measurements;
let measuredRequest;
async function showComparisons() {
  // Read the current selection after the fetch so fast task changes always win.
  try {
    measuredRequest ||= json('/comparisons.json');
    measurements ||= await measuredRequest;
  } catch {
    measuredRequest = null;
    byId('comparison-note').textContent = 'Measurements could not load. Change the task to retry.';
    return;
  }
  const task = selectedTask();
  const fault = byId('failure').checked;
  const runs = measurements.scenarios[task][fault ? 'fault' : 'normal'];
  byId('comparison-goal').textContent = form.querySelector('input[name=scenario]:checked + span').textContent +
    ' · ' + (fault ? 'First switch fails' : 'Normal run');
  byId('comparison-note').textContent = fault
    ? 'One measured run per planner. A failed switch invokes rule-based recovery. Times are individual observations.'
    : 'Three measured runs per planner. Summary times are medians. Open any repetition to see its actions.';
  const start = measurements.initial_state;
  const initial = element('p', 'Starting screen: ' + start.screen + '. Wi-Fi ' + (start.wifi_on ? 'on' : 'off') +
    ', Bluetooth ' + (start.bluetooth_on ? 'on' : 'off') + ', dark mode ' + (start.dark_mode ? 'on' : 'off') +
    ', camera permission ' + (start.camera_permission ? 'allowed' : 'denied') + '.');
  const groups = Object.entries(plannerLabels).map(([key, label]) => {
    const group = runs.filter(r => r.planner === key);
    const sorted = group.map(r => r.ms).sort((a,b) => a-b);
    const block = element('div', null, 'planner-group');
    const time = sorted[Math.floor(sorted.length / 2)];
    block.append(element('h4', label), element('p', time + ' ms · ' +
      group.filter(r => r.status === 'passed').length + '/' + group.length + ' verified'));
    for (const run of group) block.append(measuredDetail(run));
    return block;
  });
  byId('comparison-data').replaceChildren(initial, ...groups);
}

function measuredDetail(run) {
  const detail = element('details');
  const fallback = run.planner !== 'rules' && run.source !== 'llm';
  const summary = 'Run ' + run.repeat + ' · ' + run.ms + ' ms · ' + run.status + (fallback ? ' · rule fallback' : '');
  detail.append(element('summary', summary), element('p', run.calls.length + ' model calls · ' +
    run.handoffs + ' handoffs · ' + run.replans + ' replans'));
  const list = element('ol');
  for (const action of run.actions) {
    const states = Object.entries(action.toggles).map(([k,v]) => k + (v ? ' on' : ' off')).join(', ');
    list.append(element('li', actionLabel(action) + ' → ' + action.screen +
      (states ? ' (' + states + ')' : '') + '. ' + (action.ok ? '' : action.message + '. ') + action.check));
  }
  detail.append(list);
  for (const note of run.notes) detail.append(element('p', 'After ' + note.after_actions + ' actions · ' +
    note.kind.replaceAll('_',' ') + ' · ' + note.reason));
  if (run.calls.length) {
    detail.append(element('strong', 'Recorded model requests'));
    const calls = element('ol');
    for (const call of run.calls) calls.append(element('li', call.operation + ' · ' + call.latency_ms + ' ms · ' +
      (call.success ? 'succeeded' : call.error_type || 'failed')));
    detail.append(calls);
  }
  return detail;
}

let decisionPending = false;
function updateDecisionPanel(step) {
  const enabled = decisionLimit && activeRun?.planner === 'rule' && !step.verdict &&
    step.completed < activeRun.data.events.filter(e => e.event === 'action').length;
  byId('decision-panel').hidden = !enabled;
  byId('ask').disabled = decisionPending;
  byId('decision-answer').replaceChildren();
}

byId('ask').onclick = async () => {
  if (decisionPending || !activeRun) return;
  pause();
  const entry = activeRun, step = steps[position];
  decisionPending = true;
  byId('ask').disabled = true;
  const output = byId('decision-answer');
  output.textContent = 'Asking Groq about the ' + step.observation.screen + ' screen…';
  try {
    const result = await json('/api/decision', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scenario:entry.scenario, inject_failure:entry.fault, completed:step.completed})});
    if (entry !== activeRun || steps[position] !== step) return;
    const label = result.model + ' · ' + result.latency_ms + ' ms. ';
    if (!result.accepted) {
      output.textContent = label + 'No action executed. ' + result.reason;
      return;
    }
    const toggles = Object.entries(result.after.ui_tree.toggles || {}).map(([k,v])=>k + (v ? ' on' : ' off'));
    output.textContent = label + actionLabel(result.action) + '. ' + result.message +
      '. Resulting screen: ' + result.after.screen + (toggles.length ? ' (' + toggles.join(', ') + ')' : '') +
      '. Step check: ' + result.check.reason + '. Goal ' + (result.goal_check.passed ? 'met.' : 'still pending.');
  } catch (error) {
    if (entry === activeRun && steps[position] === step) output.textContent = error.message;
  } finally {
    decisionPending = false;
    byId('ask').disabled = false;
  }
};
drawHistory();
showComparisons();
