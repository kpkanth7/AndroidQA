# AndroidQA

[Live demo](https://android-qa-demo.onrender.com)

I built AndroidQA to explore whether an LLM could work through a small Android
QA task, recover when an action failed, and prove that the task actually passed.
I developed it from an existing Python QA scaffold, adding observation-driven
planning, independent verification, recovery, evaluation tooling, and emulator support.

## How it works

I kept four roles: **Planner** chooses the next action, **Executor** performs it,
**Verifier** checks the observed result, and **Supervisor** decides whether to
continue, recover, or fail. They run within one Python process. Only Planner
optionally uses an LLM; the other roles use deterministic logic.

The LLM sees the goal, current controls, and action history. Its actions must
match the visible controls, and a run passes only when the independent verifier
confirms the goal. Failed requests and rule fallbacks stay visible in the trace.

I started with simulated Wi-Fi, Bluetooth, dark mode, and camera permission
tasks, then connected the same pipeline to a Pixel 6 Android 13 emulator through
AndroidWorld. The emulator supports a Wi-Fi off/on cycle and Bluetooth on/off,
with Android's reported state checked separately from the visible switch.

## What I measured

I compared rules with Groq GPT-OSS 20B and 120B planners on eight simulated
scenarios, repeated three times per configuration.

| Planner | Verified passes | LLM-only passes | Rule fallbacks |
| --- | ---: | ---: | ---: |
| Rules | 24/24 | — | — |
| GPT-OSS 20B | 24/24 | 20/24 | 4/24 |
| GPT-OSS 120B | 24/24 | 23/24 | 1/24 |

The separate emulator pilot passed 3/3 tasks: two with only LLM actions and one
with rule fallback. I also verified recovery from interrupted Wi-Fi and
Bluetooth actions. The test suite has 81 passing tests. These are small,
controlled evaluations; broader Android coverage is still future work.

## Try it

The public demo runs simulated tasks with rules, including an optional temporary
failure. Your latest 12 runs stay in the browser tab. Compare the selected task
against measured Rules, GPT-OSS 20B, and GPT-OSS 120B runs with the same
starting state and failure setting. Normal runs have three repetitions per
planner. The separate injected-failure pilot has one per planner and task.

During a rules playback, **Ask Groq for the next action** pauses the replay and
runs one GPT-OSS 20B decision on a separate copy of the selected screen. This
uses at most one provider request. A shared Upstash counter reserves each
attempt before inference and caps the public demo at 24 per UTC day. Full
public tests always use rules. Replaying results and comparing measurements
use zero model calls. The free Render service can take a moment to wake up.

To run it locally with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m qa_agents.demo --rule-only
```

Open `http://localhost:7860`. For local Groq experiments, configure `.env` using
`.env.example` and start without `--rule-only`. To run the public one-action mode
locally, configure the two Upstash REST variables and Groq in `.env`, then start with `--public-decisions`.
The CLI, benchmark runner, and emulator pilot save traces that distinguish verified success from LLM-only success.
