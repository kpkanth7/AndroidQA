# Multi-Agent Android QA

Run repeatable QA tasks against a simulated Android Settings device. Each run
produces a JSON trace and a Markdown report.

## Mock pipeline

- **Planner** selects actions using deterministic rules or an optional LLM. Recovery uses deterministic rules.
- **Executor** sends actions to the simulator and returns observed results.
- **Verifier** checks each result and independently checks the goal.
- **Supervisor** decides whether to continue, replan, or fail, and writes the report.

---

## Run the current project

The mock environment is the supported local demo. It requires no Android emulator,
API key, or paid model access.

### Browser demo

After installing the project, start the browser demo with:

```bash
python -m qa_agents.demo
```

Open `http://localhost:7860`. Choose a mock scenario and either the rule-based
planner or Groq. The recovery option injects one temporary toggle failure.
Groq requests run on the server; the key is never sent to the browser. Runs
are isolated and temporary. The interface shows the actual plan source, so a
Groq failure is visible as a rule fallback.

The public release uses only rule-based planning. Start the same mode locally:

```bash
python -m qa_agents.demo --rule-only
```

This mode rejects LLM requests at the API even if server credentials are present.
The UI shows only the rule-based planner. Groq remains available in the local
command above without `--rule-only`; keep that mode private.

`render.yaml` configures the public rule-only service with no Groq key. Push the
repository to your own GitHub account and create a Render Blueprint from it.
Never commit or upload `.env`. The free service may sleep after inactivity;
a visitor request wakes it. Verify `/api/config` returns only `rule`, then
complete a simulation from a fresh browser session after the service has slept.

The demo permits one run at a time with a small single-process rate limit.
Temporary traces are removed after each run and are not durable. Public LLM
access would need an access gate and provider quota controls before launch.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"
```

Run the deterministic Wi-Fi demo:

```bash
android-qa --goal "Toggle Wi-Fi off and on" --env mock
# or: ./scripts/run_demo.sh
```

Other deterministic mock scenarios:

```bash
android-qa --env mock --goal "Turn Bluetooth on"
android-qa --env mock --goal "Enable dark mode"
android-qa --env mock --goal "Allow camera permission"
```

The mock also supports `Turn Bluetooth off`, `Disable dark mode`, and
`Deny camera permission`. Use `--mock-initial-state` with a JSON object to
override the starting screen and any setting values. Unspecified values use
the default state. The Wi-Fi cycle requires an observed on-to-off transition
followed by a final on state.

```bash
android-qa --env mock --goal "Turn Bluetooth off" \
  --mock-initial-state '{"screen":"network","bluetooth_on":true}'
```

Exercise the recovery path with a one-time simulated toggle failure:

```bash
android-qa --env mock --inject-failure-once toggle --logs_dir artifacts/recovery-demo
```

The resulting `qa_run.json` records the failed action, a bounded replan request,
role handoffs, per-role processing time, and run-level latency. An unrecovered
failure exits with status code `1`, which makes the command suitable for CI.

## Benchmark the mock suite

Run all four mock scenarios repeatedly without using an LLM:

```bash
android-qa-benchmark --repeats 3 --output-dir artifacts/benchmark
```

The default `canonical-v1` suite uses the four original tasks.
`--suite extended-v1` adds Wi-Fi starting off and the three reverse-setting goals
from varied screens. To run the expanded suite:

```bash
android-qa-benchmark --suite extended-v1 --repeats 3 --output-dir artifacts/extended-rule
```

The benchmark saves raw traces, summaries, and a SHA-256 digest of the Python
source. Add `--llm-planner` to evaluate Groq. Set `--pause-between-runs SECONDS`
to pace requests across runs when quota is constrained.

Use a new output directory for every benchmark; the command refuses to replace
existing results. The median averages the two middle samples when necessary;
p95 uses the nearest-rank method. LLM summaries distinguish overall task
success, success with only LLM-generated actions, and rule fallback rate, and
report provider-call failures and the number of available metric samples.

`end_to_end_latency_ms` measures the runner after argument parsing through
report creation and excludes final trace serialization. The benchmark also
records `process_wall_latency_ms` for the full child process, excluding
deliberate pauses between runs. Role timings add the elapsed time of each
role method call. Provider request time is reported separately as a subset of
Planner time. Failed requests and explicit retries each have their own
trace event. Handoffs are in-process data transfers between role calls.

In mock mode, Groq chooses one action at a time from the goal, visible
controls, and visible action history. Planner validates the selected action
and Verifier checks the goal against simulator state. This mode bypasses the
goal-only cache and records any rule fallback. `llm_plan_acceptance_rate`
counts runs without fallback, including unsuccessful runs.

Each run writes `qa_run.json` and `report.md` under `artifacts/latest/` by
default. Use a different output folder for a named run:

```bash
android-qa --env mock --logs_dir artifacts/my-run
```

Run the test suite:

```bash
pytest -q
```

## Optional LLM planning

Copy the safe template and enter your own key locally:

```bash
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`, validate the configuration, then run:

```bash
android-qa-health
android-qa --env mock --llm_planner --model openai/gpt-oss-20b
```

The run log records the configured provider/model plus planner-call latency and
token usage, but never stores API credentials. The final report is deterministic
and uses only verified run events. Groq uses OpenAI-compatible requests at
`https://api.groq.com/openai/v1`.

`.env` and generated run output are ignored by Git. Never commit a real key.

## AndroidWorld emulator

The AndroidWorld path supports `settings_wifi` ("Toggle Wi-Fi off and on")
and `settings_bluetooth` ("Turn Bluetooth on" or "Turn Bluetooth off") on
a Pixel 6 API 33 emulator. The Planner uses rules by default; `--llm_planner`
enables bounded observation-driven planning with explicit rule fallback.
The Verifier checks Android's reported Wi-Fi or Bluetooth manager state,
independently of the visible switch. The runner records the emulator environment in the
trace and report. The browser demo and benchmark still use the simulation.

For the local setup under `artifacts/androidworld-setup`, boot the AVD from
the repository root in one terminal:

```bash
setup="$PWD/artifacts/androidworld-setup"
ANDROID_HOME="$setup/sdk" ANDROID_AVD_HOME="$setup/avd" \
  "$setup/sdk/emulator/emulator" -avd AndroidWorldAvd -no-snapshot -grpc localhost:8554 -port 5554
```

Once Android has booted, run AndroidQA in another terminal:

```bash
setup="$PWD/artifacts/androidworld-setup"
"$setup/venv/bin/android-qa" --env android --task settings_wifi \
  --adb-path "$setup/sdk/platform-tools/adb" --save_frames --logs_dir artifacts/android-wifi
```

Run the small emulator pilot separately from the mock benchmark:

```bash
"$setup/venv/bin/python" scripts/emulator_pilot.py \
  --adb-path "$setup/sdk/platform-tools/adb" \
  --output-dir artifacts/emulator-pilot-new --save-frames
```

The pilot saves source snapshots, per-run traces, independent verified passes,
LLM-only passes, fallbacks, and provider errors. It prepares the opposite
Bluetooth state with rule runs before each measured Bluetooth task.
Use a fresh output directory. This adapter uses UIAutomator; an old enabled
AndroidWorld accessibility forwarder can crash and block Settings and should
be disabled on this dedicated test emulator before running.

The project is inspired by Agent-S. An integration would need an Android
observation and action bridge and a vision-capable grounding backend.

---

## Acknowledgements

- Based on [Himanshu200102/Multi-Agent-QA-System](https://github.com/Himanshu200102/Multi-Agent-QA-System).
  This checkout adds the runnable demo, observation-driven planner, independent
  verification, bounded recovery, evaluation tooling, and emulator Settings adapter.
- **Agent-S (`gui-agents`)** by Simular
- **android_world / AndroidEnv** by Google Research
