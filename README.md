# Sketch2Shape

Turn a rough 2D sketch into a textured, exportable 3D model. Draw on a canvas or upload a sketch,
optionally describe the object, and get a `.glb` (texture embedded) and a zipped `.obj` you can open
in Blender, Windows 3D Viewer or any glTF viewer.

While a model is being made, the app shows what is happening right now: a progress bar, the current
step with its elapsed and usual time, a checklist of finished steps, and each intermediate result as it
appears (cleaned sketch, candidate images with their scores, cut-out, then the untextured 3D shape,
which can be rotated while the texture is painted).

Built for TechZephyr 4.0 (IIT Bhubaneswar), problem statement "Sketch2Shape AI".

> Numbers marked **TBD** are filled in from `scripts/eval_batch.py` runs on the hardware listed below.

## How it works

```
sketch (canvas or upload) + optional description
  1 preprocess    OpenCV: flatten, invert if dark, even out paper lighting, threshold, drop specks,
                  crop and centre, square to 1024 px, close gaps, thicken thin lines, fill a silhouette
  2 understand    name the object: one vision-LLM call if OPENAI_API_KEY (or ANTHROPIC_API_KEY) is set,
                  otherwise CLIP zero-shot over ~170 everyday objects; the same call suggests real colours
                  and surface (keeping any the user typed) and judges the drawing clean or rough; build the
                  prompt (a typed description always comes first)
  3 generate      SDXL + xinsir scribble ControlNet: a photorealistic studio photo that keeps the sketch's
                  outline (no realism words if the description asks for a style such as cartoon or toy);
                  a rough doodle gets lighter control for the first 60% of the steps, so it becomes a
                  real-looking object instead of a copy of the cartoon
  4 score         gate 1: sketch-line recall vs image edges, silhouette IoU, CLIP agreement
                  -> below threshold: regenerate with new seeds and stronger conditioning (max 2 rounds)
  5 prepare       background removal (fallback: colour threshold), crop, centre, margin
  6 reconstruct   Hunyuan3D-2: shape (turbo + FlashVDM), floater/degenerate-face clean-up, then texture
  7 validate      gate 2: enough faces, finite, not flat, no loose fragments, texture present
                  -> shape fails: new seed; texture fails: colours projected from the image
  8 export        normalise (centre, 1 unit, Y-up, on y = 0), GLB + OBJ zip, reload test
```

The run is driven by a small explicit state machine (`sketch2shape/pipeline.py`). Stages are plain
functions; only the controller decides what runs next, using a transition table:

| From | Outcome | Next |
|---|---|---|
| preprocess | ok / empty | understand / failed (asks to redraw) |
| generate | ok | score |
| score | pass / retry / give up | prepare / generate / prepare (best candidate, flagged) |
| shape | ok / retry / use best / error | check shape / shape / texture / failed |
| check shape | pass / retry / accept flagged | texture / shape / texture |
| texture | ok / fallback | check texture / export (projected colours) |
| check texture | pass / fallback | export |
| export | ok | done |

Every loop has an attempt limit and the whole run has a time budget (`config.yaml`). Once any mesh
exists, an unexpected error exports the best result so far instead of returning nothing.

**GPU worker processes.** The three heavy steps (SDXL, Hunyuan3D-2 shape, Hunyuan3D-2 texture) each
run in a short-lived worker process (`sketch2shape/worker.py`) pinned to its GPU, which loads the
model, does one job, writes files and exits. Keeping all models in one process ran out of Kaggle's
30 GB of CPU RAM (peak 31 GB while loading the texture model); separate processes give memory back
after every step, and a crash inside one model cannot take down the app. CLIP and background removal
are small and run on the CPU in the main process.

**Why no agent framework.** The control flow is short and known in advance; the uncertainty is in the
model outputs, so the pipeline measures them (gates 1 and 2) instead of asking an LLM what to do next.
That keeps runs repeatable (fixed seeds, logged in `manifest.json`) and adds no latency. The full
architecture evaluation is in `docs/Sketch2Shape Architecture Decision.pdf`.

## Models and third-party components

| Component | Used for | Licence |
|---|---|---|
| [Stable Diffusion XL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | sketch to image | CreativeML Open RAIL++-M |
| [xinsir/controlnet-scribble-sdxl-1.0](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0) | keeps the sketch's lines | see model card |
| [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | fp16-safe SDXL VAE | MIT |
| [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) (shape turbo + paint) | image to textured 3D, background removal | Tencent Hunyuan community / non-commercial licence |
| [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | naming the object, scoring | MIT |
| OpenAI API (optional; first available of `gpt-4.1-mini`, `gpt-5-mini`, `gpt-4o-mini`) | naming the object and its shape from the sketch (one call, image sent at low detail, well under a cent) | external service, needs `OPENAI_API_KEY` |
| Claude API (optional alternative, `claude-opus-5-5`) | same, used when only `ANTHROPIC_API_KEY` is set | external service |
| diffusers, transformers, OpenCV, trimesh, Gradio | libraries | Apache-2.0 / MIT / BSD |

External services: model weights download from Hugging Face on first run; the LLM naming call is
optional and the pipeline runs fully without it (any API error, refusal or missing key falls back to
CLIP). Keep keys in environment variables or Kaggle Secrets, never in the code. Run details show which
one named the object (`named_by`: `openai:<model>`, `anthropic:<model>`, `clip` or `none`).

## Hardware and run time

| | |
|---|---|
| Tested on | Kaggle notebook, 2x NVIDIA T4 (15 GB usable each), Python 3.13.15, PyTorch 2.11.0 + CUDA 12.8 |
| GPU placement | SDXL worker on GPU 0; Hunyuan3D-2 shape worker, then texture worker, on GPU 1; CLIP and background removal on CPU |
| Time per model, end to end | **TBD** median, **TBD** slowest (from `scripts/eval_batch.py`; includes about 1 min of model loading per run) |
| Single GPU | set `gpus.hunyuan: 0`; workers run one after another, so a 16 GB GPU is enough |

Measured per stage on the T4s (Kaggle trial run, 31 Oct 2026):

| Stage | Time | Peak GPU memory |
|---|---|---|
| SDXL + ControlNet, 2 images at 1024 px, 25 steps | 62-69 s | 13.4 GB (GPU 0) |
| Hunyuan3D-2 shape (turbo + FlashVDM, 5 steps), 40k faces | 7-11 s | 6.2-9.4 GB (GPU 1) |
| Hunyuan3D-2 texture | 39-66 s | 12.8 GB (GPU 1) |
| First-run model loading: SDXL / shape / texture | 41 s / 57 s / 66 s | |
| Compiling the two CUDA extensions (setup, once) | about 25 s | |

## Setup and running

### On Kaggle (free, recommended)

1. Create a notebook from `notebooks/kaggle_run.ipynb` (File, Import Notebook). It clones https://github.com/RohanSaikumar/sketch2shape.
2. Session options: Accelerator **GPU T4 x2**, Internet **On**. Optionally add `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) and `HF_TOKEN` as secrets.
3. Run All (change `REPO_URL` in the first cell only for a fork). It runs setup, the offline tests and all example sketches, shows the results, then the app with a public link.

### On a Linux machine with an NVIDIA GPU

```bash
git clone https://github.com/RohanSaikumar/sketch2shape.git && cd sketch2shape
TORCH_CUDA_ARCH_LIST=8.9 bash scripts/setup.sh      # use your GPU's compute capability
python app.py                                        # web app on http://localhost:7860
python -m sketch2shape.cli examples/sketches/mug.png --prompt "a white ceramic coffee mug"
```

PyTorch with CUDA must already be installed. `scripts/setup.sh` clones Hunyuan3D-2 to `/tmp/Hunyuan3D-2`
(set `HY_DIR` to change it), compiles its two CUDA extensions and applies a one-line compatibility patch.

### Command line and batch evaluation

```bash
python -m sketch2shape.cli sketch.png --prompt "a red ceramic mug" [--candidates 1] [--no-texture]
python scripts/eval_batch.py examples/sketches --prompts examples/prompts.json --out results
python -m pytest -q tests          # offline tests, no GPU needed
```

## Inputs and outputs

- **Input:** a drawing from the canvas, or a PNG/JPG/WebP of a sketch (canvas export, scan or phone photo). Optional text description.
- **Output:** `runs/<run id>/model.glb` and `model_obj.zip`, plus every intermediate:
  `01_input.png`, `02_lineart.png`, `03_control.png`, `04_sketch_mask.png`, `05_candidate_*.png`,
  `06_prepared.png`, `07_shape_try*.glb` and `manifest.json` (inputs, settings, seeds, scores, checks,
  timings, warnings and library versions).

## Results

**TBD**: table from `results/results.md`, with input sketch, chosen image and a render of each model.

## Limitations

- One sketch is one view: hidden sides are plausible guesses, not recovered geometry.
- Thin parts (chair legs, handles) can thicken or merge.
- The object name comes from Claude or from a fixed CLIP list; unusual objects may be misnamed. A typed description fixes this.
- Textures are generated, so fine sketch details (patterns, text) are not reproduced exactly.
- Meshes are for viewing and further editing, not print-ready or animation-ready topology.
- Needs an NVIDIA GPU with at least 16 GB (two GPUs, or one large one, recommended).

## Troubleshooting

| Problem | Fix |
|---|---|
| `'StableDiffusionXLControlNetPipeline' object has no attribute 'enable_vae_slicing'` | Newer diffusers; already handled (`pipe.vae.enable_slicing()`). Update the code. |
| `... contains custom code in pipeline.py ... Pass trust_remote_code=True` | Run `scripts/setup.sh`; it patches Hunyuan's multiview loader. |
| CUDA out of memory while texturing | Handled automatically: retries with CPU offload. Only one 3D model is on GPU 1 at a time. |
| "Your notebook tried to allocate more memory than is available" (CPU RAM) | Fixed by running each heavy model in its own worker process. Check `/kaggle/working/ram.log`; a failed step's log is `runs/<id>/worker_<task>_*.log`. |
| A step fails with "... worker failed" | Open the worker log named in the message (`runs/<id>/worker_*.log`) for the model's own error. |
| CUDA extension build fails with pip | `scripts/setup.sh` falls back to `python setup.py install`; its log is in `/tmp/s2s_build_<name>.log`. |
| Model looks mirrored in projected-colour fallback | Set `export.flip_projection_x: true`. |
| "The sketch looks empty" | Draw darker or thicker lines, or upload a higher-contrast photo. |
| gradio.live link does not open | Keep the notebook tab active; rerun the app cell for a new link. |
