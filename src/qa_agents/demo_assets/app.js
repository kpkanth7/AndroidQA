const form = document.querySelector('#run-form');
const text = (tag, value, className) => { const el = document.createElement(tag); el.textContent = value; if (className) el.className = className; return el; };
form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = document.querySelector('#run'), status = document.querySelector('#status'), error = document.querySelector('#error');
  button.disabled = true; button.firstChild.textContent = 'Running simulation '; status.textContent = 'RUNNING'; status.className = 'status'; error.hidden = true;
  document.querySelector('#empty').hidden = true; document.querySelector('#output').hidden = true;
  try {
    const response = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({scenario:document.querySelector('#scenario').value, planner:document.querySelector('#planner').value, inject_failure:document.querySelector('#failure').checked})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Unable to run the simulation.');
    const find = name => data.events.find(event => event.event === name)?.payload || {};
    const plan = find('plan'), metrics = find('run_metrics'), finish = find('finish');
    const client = find('run_metadata').clients?.[0];
    status.textContent = (finish.status || 'unknown').toUpperCase(); status.className = 'status ' + (finish.status === 'passed' ? 'good' : 'bad');
    document.querySelector('#subtitle').textContent = [
      plan.goal || 'Simulation complete',
      client && `${client.provider} / ${client.model}${client.configured ? '' : ' (unavailable)'}`,
    ].filter(Boolean).join(' · ');
    const source = {llm:'LLM', llm_with_rule_fallback:'LLM + fallback', rule_fallback_invalid_or_unavailable:'Rule fallback', unsupported_goal:'Unsupported goal'}[plan.source] || 'Rule-based';
    const elapsed = metrics.end_to_end_latency_ms;
    const duration = elapsed < 1000 ? `${elapsed} ms` : `${(elapsed / 1000).toFixed(2)} s`;
    const metricBox = document.querySelector('#metrics'); metricBox.replaceChildren();
    [['Plan source',source],['Duration',duration],['Role handoffs',String(metrics.agent_message_count)],['Replans',String(metrics.replan_count)]].forEach(([label,value]) => {const box=text('div','','metric');box.append(text('small',label),text('strong',value));metricBox.append(box)});
    const trace = document.querySelector('#trace'); trace.replaceChildren();
    data.events.forEach(event => {let role, title, detail='';const p=event.payload;
      if(event.event==='plan'){role='Planner';title='Plan summary';detail=`${p.steps.length} recorded steps · ${source}`}
      else if(event.event==='action'){role='Executor';title=`${p.ok ? '✓' : '✕'} ${p.intent.replaceAll('_',' ')}${p.target ? ': '+p.target : ''}`;detail=p.message}
      else if(event.event==='verify'){role='Verifier';title=p.passed ? 'Check passed' : 'Check failed';detail=p.reason}
      else if(event.event==='replan'){role='Supervisor';title='Recovery requested';detail=p.reason}
      else if(event.event==='plan_fallback'){role='Planner';title='Rule fallback used';detail=p.reason}
      else if(event.event==='plan_rejection'){role='Planner';title='Action rejected';detail=p.reason}
      else if(event.event==='agent_message'){role='Handoff';title=`${p.from} → ${p.to}`;detail=p.kind.replaceAll('_',' ')}
      else if(event.event==='llm'){role='Planner';title=p.success ? 'Provider request succeeded' : 'Provider request failed';detail=`${p.provider} · ${p.operation} · ${p.latency_ms} ms${p.error_type ? ' · '+p.error_type : ''}`}
      else if(event.event==='finish'){role='Supervisor';title=`Run ${p.status}`;detail=p.reason}
      else return;
      const li=document.createElement('li'), content=document.createElement('div');content.append(text('span',title));if(detail)content.append(text('small',detail));li.append(text('span',role,'role'),content);trace.append(li);
    });
    document.querySelector('#report').textContent=data.report; document.querySelector('#output').hidden=false;
  } catch (e) {status.textContent='ERROR';status.className='status bad';error.textContent=e.message;error.hidden=false;}
  finally {button.disabled=false;button.firstChild.textContent='Run simulation ';}
});

async function loadConfig() {
  const note = document.querySelector('#planner-note');
  try {
    const response = await fetch('/api/config');
    if (!response.ok) throw new Error('Unable to load demo configuration. Reload to try again.');
    const config = await response.json();
    if (config.planners.includes('llm')) {
      document.querySelector('#planner').append(new Option('Groq LLM · live request', 'llm'));
      note.textContent = 'The Groq option uses the host’s server-side key. If planning fails, the run uses a rule-based fallback.';
    } else {
      note.textContent = 'Public demo · rule-based planning. No model requests or API key required.';
    }
    document.querySelector('#run').disabled = false;
  } catch (error) {
    note.textContent = error.message;
  }
}
loadConfig();
