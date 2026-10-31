#!/usr/bin/env bash
# One-time environment setup on Linux with an NVIDIA GPU (tested on Kaggle "GPU T4 x2", Python 3.13).
# Installs Hunyuan3D-2, builds its two CUDA extensions (or installs prebuilt wheels), applies one
# compatibility patch for newer diffusers, and installs this project's requirements.
# Safe to re-run: finished steps are skipped or are quick no-ops.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HY_DIR="${HY_DIR:-/tmp/Hunyuan3D-2}"
HY_REF="${HY_REF:-main}"
WHEELS_DIR="${WHEELS_DIR:-/kaggle/working/wheels}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"   # 7.5 = T4; e.g. 8.6 for RTX 30xx, 8.9 for RTX 40xx

if [ ! -d "$HY_DIR/.git" ]; then
  echo "== Cloning Hunyuan3D-2 ($HY_REF)"
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 "$HY_DIR"
  git -C "$HY_DIR" checkout "$HY_REF"
fi
echo "== Hunyuan3D-2 commit: $(git -C "$HY_DIR" rev-parse --short HEAD)"

echo "== Installing Hunyuan3D-2 requirements"
pip install -q -r "$HY_DIR/requirements.txt"
pip install -q -e "$HY_DIR"

if python -c "import torch, custom_rasterizer, mesh_processor" 2>/dev/null; then
  echo "== CUDA extensions already installed"
elif ls "$WHEELS_DIR"/*.whl >/dev/null 2>&1; then
  echo "== Installing prebuilt CUDA extension wheels from $WHEELS_DIR"
  pip install -q --force-reinstall --no-deps "$WHEELS_DIR"/*.whl
else
  for ext in custom_rasterizer differentiable_renderer; do
    echo "== Compiling $ext (about 30 s on a Kaggle T4 machine)"
    # pip's build can fail on this setup.py; the plain setup.py install worked on Kaggle (Python 3.13)
    (cd "$HY_DIR/hy3dgen/texgen/$ext" && { pip install -q --no-build-isolation . > "/tmp/s2s_build_$ext.log" 2>&1 \
      || python setup.py install >> "/tmp/s2s_build_$ext.log" 2>&1; })
  done
fi

echo "== Patching Hunyuan's multiview loader for newer diffusers (trust_remote_code)"
python - "$HY_DIR/hy3dgen/texgen/utils/multiview_utils.py" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)"
if "trust_remote_code" not in src:
    if old not in src:
        sys.exit("Could not find the line to patch in " + path)
    open(path, "w").write(src.replace(old, old[:-1] + ", trust_remote_code=True)"))
print("patch in place")
EOF

echo "== Installing Sketch2Shape requirements"
pip install -q -r "$ROOT/requirements.txt"

python -c "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; from hy3dgen.texgen import Hunyuan3DPaintPipeline; print('== Ready: Hunyuan3D-2 shape and texture import OK')"
