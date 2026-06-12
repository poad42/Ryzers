# CaP-X

A ROCm-enabled container for [**CaP-X**](https://github.com/capgym/cap-x) — *Code-as-Policies eXtended*, a framework for benchmarking and improving coding agents for robot manipulation (NVIDIA / UC Berkeley / Stanford / CMU).

This Ryzer runs the **CaP-Gym evaluation path** on AMD Ryzen AI hardware: a coding-agent LLM generates Python that composes perception + control primitives to solve manipulation tasks in simulation. It supports two simulator families:

- **Robosuite** (1.5.x)
- **LIBERO-PRO** (robosuite 1.4.x)

Everything GPU-bound runs on ROCm: the perception models (SAM3, Contact-GraspNet) on ROCm PyTorch, and MuJoCo rendering via EGL on the AMD GPU. Motion planning (PyRoKi IK) runs on CPU `jax`, exactly as CaP-X ships upstream.

> **Validated on:** AMD Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151) / Strix Halo, Ubuntu 24.04, ROCm 7.2.

---

## What's included vs. not

| Component | Status | Notes |
|---|---|---|
| Robosuite eval | ✅ | `--build-arg CAPX_SIM=robosuite` (default) |
| LIBERO-PRO eval | ✅ | `--build-arg CAPX_SIM=libero` |
| SAM3 / SAM2 / OWLv2 perception | ✅ | ROCm PyTorch |
| Contact-GraspNet | ✅ | PyTorch fork, no custom CUDA kernels |
| PyRoKi IK / trajopt | ✅ | CPU `jax` (matches upstream lockfile) |
| MuJoCo rendering | ✅ | EGL on AMD GPU |
| cuRobo | ❌ | Hand-written CUDA/NVRTC kernels, no ROCm fork. Gated off in the default configs anyway — PyRoKi is the default motion layer. |
| CaP-RL (verl / vLLM / flash-attn) | ❌ | The training path. Portable to ROCm (verl ships ROCm Dockerfiles using ROCm/vLLM + ROCm/aiter), but it's a separate, larger image. |
| BEHAVIOR / Isaac Sim (`b1k`) | ❌ | NVIDIA Isaac Sim, out of scope. |

> Robosuite (1.5.x) and LIBERO (robosuite 1.4.x) pin **conflicting** robosuite versions, so each image targets exactly one family — pick it at build time.

---

## Build & Run

```bash
# Robosuite (default)
ryzers build capx
ryzers run                 # runs the sign-of-life test

# LIBERO-PRO — set CAPX_SIM=libero in config.yaml build_arguments first,
# then build under a distinct name:
ryzers build --name capx-libero capx
ryzers run --name capx-libero
```

`ryzers run` executes [`test.sh`](test.sh), which has **three stages**:

1. **Sign-of-life** — ROCm torch sees the GPU; the stack + simulator import; MuJoCo renders via EGL.
2. **Oracle eval (always runs, no LLM / no keys)** — runs a full CaP-Gym episode using the environment's ground-truth `oracle_code` through the real simulator + PyRoKi IK pipeline, and asserts task **success (reward 1.0)**. This is the deterministic correctness check for the eval pipeline on ROCm. It needs no API key.
3. **LLM eval (optional)** — if `CAPX_LLM_SERVER_URL` is set, runs a short real *agentic* eval (the LLM writes code → the sim executes it → reward) against any OpenAI-compatible endpoint.

Expected tail (stages 1–2; stage 3 skipped when no LLM is configured):

```
================ [1/3] CaP-X / ROCm sign-of-life ================
GPU ok  : True
  device 0: Radeon 8060S Graphics
capx + pyroki import OK
robosuite 1.5.1 import OK
EGL render OK, frame shape (64, 64, 3)
================ [2/3] Oracle eval (no LLM): franka_robosuite_pick_place_code_env ================
PyRoKi ready (warmed JAX JIT) after 19s
Success
ORACLE EVAL PASSED (reward 1.0)
================ [3/3] LLM eval (optional) ================
SKIPPED — set CAPX_LLM_SERVER_URL ...
================ CaP-X tests PASSED ================
```

### Enabling the LLM eval stage

Set these env vars (in `config.yaml` or via `-e`) to make `ryzers run` exercise a
real agentic eval against an OpenAI-compatible endpoint:

| Variable | Meaning | Example |
|---|---|---|
| `CAPX_LLM_SERVER_URL` | OpenAI-compatible `/chat/completions` URL | `http://127.0.0.1:11434/v1/chat/completions` |
| `CAPX_LLM_MODEL` | model name to request | `gemma4`, `google/gemini-3.1-pro-preview` |
| `CAPX_LLM_TRIALS` | number of trials (default 2) | `2` |

> **Disclaimer.** The LLM-eval stage was validated end-to-end against a **local
> llama.cpp `llama-server`** (built with HIP, serving a vision GGUF) — the full
> loop closes: LLM → code → SAM3/GraspNet/PyRoKi perception → sim → reward +
> saved artifacts. It works identically against **OpenRouter** or any other
> OpenAI-compatible server. Task *success rate* depends entirely on the model:
> small local models often write buggy code and score low; that is a model
> outcome, not an infra problem. The **oracle eval (stage 2)** is the
> infra-correctness check and always returns reward 1.0 when the ROCm pipeline
> is healthy.

For an interactive shell (to launch full benchmark evals yourself):

```bash
ryzers run bash
```

---

## Running an evaluation

Evals need two things: an **LLM endpoint** and (optionally) **perception weights**.

### 1. LLM provider

#### Option A — OpenRouter (default, like upstream)

Get a key at <https://openrouter.ai/keys> and set `OPENROUTER_API_KEY` in [`config.yaml`](config.yaml). Inside the container, start CaP-X's OpenRouter proxy (it exposes an OpenAI-compatible API on `:8110`), then launch an eval:

```bash
# inside `ryzers run bash`
cd /ryzers/cap-x
echo "$OPENROUTER_API_KEY" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py \
    --key-file .openrouterkey --port 8110 &

# Robosuite single-turn benchmark
python3 capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "openrouter/google/gemini-3.1-pro-preview" \
    --total-trials 10 --num-workers 4
```

The perception servers (SAM3, GraspNet, PyRoKi) listed in the YAML's `api_servers` are **auto-launched** by `launch.py`.

#### Option B — Local / on-prem OpenAI-compatible LLM

CaP-X's LLM client (`capx/llm/client.py`) posts a standard OpenAI `chat/completions` payload to whatever `--server-url` you give it, for any model name not in its built-in OpenRouter/GPT/Claude lists. So **any** OpenAI-compatible server works — run it on the host and point the eval at it.

Two ROCm-friendly servers:

- **llama.cpp `llama-server`** (built with HIP for your GFX arch):
  ```bash
  # on the host, serve a vision-capable GGUF (CaP-X sends image prompts)
  llama-server --model gemma-4-12B-it-Q4_K_M.gguf \
               --mmproj mmproj-gemma-4-12B-it-bf16.gguf \
               --host 0.0.0.0 --port 11434 --n-gpu-layers 999 --jinja
  ```
- **vLLM (ROCm build)** or **Ollama (ROCm)** — both expose `/v1/chat/completions`.

Then, inside the container (Ryzers uses host networking, so `127.0.0.1` reaches the host server):

```bash
cd /ryzers/cap-x
python3 capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model gemma-4-12B-it \
    --server-url http://127.0.0.1:11434/v1/chat/completions \
    --total-trials 10 --num-workers 2
```

> **Vision is required.** CaP-X sends `image_url` content, so the served model must accept images (e.g. a `-VL` / multimodal model with its `mmproj`). A text-only model will not work for the perception-grounded configs.

### 2. Perception weights

#### Option A — SAM3 (upstream default, gated)

SAM3 weights (`facebook/sam3`) are **gated** on HuggingFace. Request access at <https://huggingface.co/facebook/sam3>, create a token at <https://huggingface.co/settings/tokens>, and set `HF_TOKEN` in `config.yaml`. The entrypoint logs in and the weights download on first use (cached via the `~/.cache/huggingface` volume mount). The default `env_configs/**` YAMLs already use SAM3.

#### Option B — SAM2 + OWLv2 (ungated, no token)

If you don't have SAM3 access, swap to SAM2 + OWLv2, which are **not gated**. The control APIs already support this via a `use_sam3=False` path. Inside the container:

```bash
cd /ryzers/cap-x
# register an ungated-perception API variant
cat >> capx/integrations/__init__.py <<'PY'
register_api("FrankaControlApiSam2", lambda env: FrankaControlApi(env, use_sam3=False))
PY
```

Then write a config that points `apis:` at `FrankaControlApiSam2` and swaps the SAM3 `api_server` for the SAM2 + OWLv2 servers:

```yaml
# env_configs/cube_stack/franka_robosuite_cube_stack_sam2.yaml
env:
  _target_: capx.envs.tasks.franka.franka_pick_place.FrankaPickPlaceCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_cubes_low_level
    privileged: false
    apis: [FrankaControlApiSam2]
api_servers:
  - {_target_: capx.serving.launch_owlvit_server.main, port: 8117, host: 127.0.0.1}
  - {_target_: capx.serving.launch_sam2_server.main, device: cuda, port: 8113, host: 127.0.0.1}
  - {_target_: capx.serving.launch_contact_graspnet_server.main, port: 8115, host: 127.0.0.1}
  - {_target_: capx.serving.launch_pyroki_server.main, port: 8116, host: 127.0.0.1, robot: panda_description, target_link: panda_hand}
record_video: true
output_dir: ./outputs/eval_robosuite_cube_stack_sam2
trials: 10
num_workers: 2
```

> The SAM2 server is hard-wired to port **8113** and OWLv2 to **8117** (`capx/integrations/vision/{sam2,owlvit}.py`) — match those in the config.

### LIBERO-PRO example

```bash
# inside `ryzers run bash` (libero image), with an LLM server reachable
cd /ryzers/cap-x
python3 capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model <your-model> --server-url <your-endpoint> \
    --total-trials 5 --num-workers 1
```

Outputs (per-trial rewards + videos) land in `outputs/<model>/<config>/` — surfaced on the host via the `outputs` volume mount.

---

## Notes, fixes & troubleshooting

This package bakes in several fixes discovered while porting CaP-X (whose dependency set is `uv`-native and assumes CUDA) to a ROCm PyTorch base on Python 3.12:

- **torch is never reinstalled.** A generated `constraints.txt` pins the base image's ROCm torch/torchvision so no transitive dep can pull a CUDA wheel.
- **`jax` is CPU-only**, matching CaP-X's lockfile (plain `jax==0.4.29`, no GPU plugin). PyRoKi IK is a CPU workload upstream too — this is faithful, not a downgrade.
- **`PyOpenGL-accelerate==3.1.6` pin dropped** — it has no cp312 wheel and fails to compile; it's an optional perf shim.
- **`setuptools<81`** — SAM3 imports `pkg_resources`, removed in setuptools ≥81.
- **`--no-build-isolation`** on the editable submodule installs (robosuite, contact_graspnet, LIBERO-PRO) — their `setup.py` import numpy at build time.
- **LIBERO `future>=1.0`** — LIBERO pins `future==0.18.2`, whose `standard_library` does `import imp` (removed in Python 3.12), which breaks `pyglet → pyrender →` the GraspNet server. Upgrading `future` fixes it and LIBERO still imports.
- **LIBERO non-interactive init** — LIBERO calls `input()` for a dataset path on first import; the build pre-seeds `~/.libero/config.yaml` so runtime is non-interactive.
- **Trimmed submodule clone** — only the submodules the eval needs are inited (skips curobo/b1k/verl), saving significant build time and image size.

Other tips:

- **`invalid device function` / no GPU:** your part may need an HSA override. Set `HSA_OVERRIDE_GFX_VERSION` in `config.yaml` (e.g. `11.5.1` for gfx1151, `11.0.0` for gfx1103).
- **Large local models are slow on an iGPU.** A 12B-class vision model is a good speed/quality balance; smaller models often write buggy code and score low — that's a model-quality outcome, not an infra problem.
- **Trial timeout:** very slow local models can exceed CaP-X's per-trial wall-clock (`TRIAL_TIMEOUT_SECONDS` in `capx/envs/runner.py`). Lower `--max-tokens`, use fewer workers (less GPU contention), or raise the timeout.

---

## References

- CaP-X: <https://github.com/capgym/cap-x> · [paper](https://arxiv.org/abs/2603.22435) · [project page](https://capgym.github.io/)
- LIBERO-PRO: <https://github.com/uynitsuj/LIBERO-PRO>
- Robosuite: <https://github.com/ARISE-Initiative/robosuite>
