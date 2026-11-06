import json
import os
import subprocess
import sys
from pathlib import Path

def test_demo_runs(tmp_path):
    pkg = 'qa_agents.run_test'
    logs_dir = tmp_path / 'logs'
    cmd = [sys.executable, '-m', pkg, '--goal', 'Toggle Wi‑Fi off and on', '--env', 'mock', '--logs_dir', str(logs_dir)]
    project_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(project_root / 'src')}
    assert subprocess.call(cmd, env=env) == 0
    assert os.path.exists(logs_dir / 'qa_run.json')
    with open(logs_dir / 'qa_run.json') as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0
    finish = next(record for record in data if record['event'] == 'finish')
    final_verification = [record for record in data if record['event'] == 'verify'][-1]
    metrics = next(record for record in data if record['event'] == 'run_metrics')
    assert finish['payload']['status'] == 'passed'
    assert final_verification['payload']['reason'] == 'Verified Wi‑Fi: desired=True actual=True'
    assert metrics['payload']['agent_message_count'] > 0
    assert data[0]['event'] == 'run_metadata'
    assert data[0]['payload']['llm_enabled'] is False


def test_transient_mock_failure_is_replanned(tmp_path):
    logs_dir = tmp_path / 'recovery'
    project_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(project_root / 'src')}
    cmd = [
        sys.executable, '-m', 'qa_agents.run_test', '--env', 'mock',
        '--inject-failure-once', 'toggle', '--logs_dir', str(logs_dir),
    ]

    assert subprocess.call(cmd, env=env) == 0
    with open(logs_dir / 'qa_run.json') as f:
        data = json.load(f)

    assert next(record for record in data if record['event'] == 'finish')['payload']['status'] == 'passed'
    assert len([record for record in data if record['event'] == 'replan']) == 1
    assert next(record for record in data if record['event'] == 'run_metrics')['payload']['replan_count'] == 1
    failed_action_index = next(i for i, record in enumerate(data) if record['event'] == 'action' and not record['payload']['ok'])
    failed_verification = next(record for record in data[failed_action_index + 1:] if record['event'] == 'verify')
    assert failed_verification['payload']['passed'] is False


def test_unrecovered_failure_returns_nonzero_status(tmp_path):
    logs_dir = tmp_path / 'failed-recovery'
    project_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(project_root / 'src')}
    cmd = [
        sys.executable, '-m', 'qa_agents.run_test', '--env', 'mock',
        '--inject-failure-once', 'toggle', '--max-replans', '0', '--logs_dir', str(logs_dir),
    ]

    assert subprocess.call(cmd, env=env) == 1
    with open(logs_dir / 'qa_run.json') as f:
        data = json.load(f)
    assert next(record for record in data if record['event'] == 'finish')['payload']['status'] == 'failed'
