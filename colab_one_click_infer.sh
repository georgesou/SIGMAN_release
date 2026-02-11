#!/usr/bin/env bash
set -euo pipefail

# SIGMAN Colab one-click setup + inference
# Example:
# bash colab_one_click_infer.sh \
#   --assets-zip /content/assets_bundle.zip \
#   --image /content/input.jpg \
#   [--pose /content/your_pose.npz]

ASSETS_ZIP="/content/assets_bundle.zip"
INPUT_IMAGE="/content/input.jpg"
POSE_PATH=""
REPO_DIR="/content/SIGMAN_release"
REPO_URL="https://github.com/yyvhang/SIGMAN_release.git"
TMP_ASSET_DIR="/content/_sigman_assets"

usage() {
  cat <<'USAGE'
SIGMAN Colab one-click setup + inference script

Required:
  --assets-zip PATH   Zip containing smplx + template assets
  --image PATH        Input image path

Optional:
  --pose PATH         SMPL-X pose npz path (default: demo pose in repo)
  --repo-dir PATH     Clone/use repo path (default: /content/SIGMAN_release)
  --repo-url URL      Repo URL (default: official SIGMAN_release)
  -h, --help          Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --assets-zip) ASSETS_ZIP="$2"; shift 2 ;;
    --image) INPUT_IMAGE="$2"; shift 2 ;;
    --pose) POSE_PATH="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1"; usage; exit 1 ;;
  esac
done

[[ -f "$ASSETS_ZIP" ]] || { echo "[ERROR] Missing assets zip: $ASSETS_ZIP"; exit 1; }
[[ -f "$INPUT_IMAGE" ]] || { echo "[ERROR] Missing input image: $INPUT_IMAGE"; exit 1; }

echo "[1/8] Clone repo if needed"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --recursive "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

if [[ -z "$POSE_PATH" ]]; then
  POSE_PATH="$REPO_DIR/demo/poses/smplx_demo.npz"
fi
[[ -f "$POSE_PATH" ]] || { echo "[ERROR] Missing pose file: $POSE_PATH"; exit 1; }

echo "[2/8] Install PyTorch + xformers"
python -m pip install -U pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

echo "[3/8] Install project dependencies"
if [[ ! -d diff-gaussian-rasterization ]]; then
  git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
fi
pip install ./diff-gaussian-rasterization

if [[ ! -d gaussian-splatting ]]; then
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git
fi

pip install git+https://github.com/NVlabs/nvdiffrast
pip install -r requirements.txt

echo "[4/8] Download checkpoints"
mkdir -p ckpt/autoencoder ckpt/transformer ckpt/sapiens_1b
wget -O ckpt/autoencoder/autoencoder.safetensors \
  https://huggingface.co/Mr-Hang/SIGMAN/resolve/main/autoencoder.safetensors
wget -O ckpt/transformer/transformer.safetensors \
  https://huggingface.co/Mr-Hang/SIGMAN/resolve/main/transformer.safetensors
wget -O ckpt/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2 \
  https://huggingface.co/facebook/sapiens-pretrain-1b-torchscript/resolve/main/sapiens_1b_epoch_173_torchscript.pt2

echo "[5/8] Unzip and place assets"
rm -rf "$TMP_ASSET_DIR"
mkdir -p "$TMP_ASSET_DIR"
unzip -o "$ASSETS_ZIP" -d "$TMP_ASSET_DIR" >/dev/null

mkdir -p core/modules/deformers/smplx/SMPLX
mkdir -p core/modules/deformers/template

if [[ -d "$TMP_ASSET_DIR/smplx/SMPLX" ]]; then
  cp -r "$TMP_ASSET_DIR/smplx/SMPLX/"* core/modules/deformers/smplx/SMPLX/
elif [[ -d "$TMP_ASSET_DIR/SMPLX" ]]; then
  cp -r "$TMP_ASSET_DIR/SMPLX/"* core/modules/deformers/smplx/SMPLX/
else
  echo "[ERROR] SMPLX folder not found in assets zip"
  exit 1
fi

if [[ -d "$TMP_ASSET_DIR/template" ]]; then
  cp -r "$TMP_ASSET_DIR/template/"* core/modules/deformers/template/
else
  echo "[ERROR] template folder not found in assets zip"
  exit 1
fi

echo "[6/8] Preprocess template if needed"
pushd core/modules/deformers >/dev/null
required_npy=(
  "template/init_uv_smplx_thu.npy"
  "template/init_pcd_smplx_thu.npy"
  "template/init_rot_smplx_thu.npy"
  "template/face_mask_thu.npy"
  "template/hands_mask_thu.npy"
  "template/outside_mask_thu.npy"
)
need_preprocess=0
for f in "${required_npy[@]}"; do
  if [[ ! -f "$f" ]]; then
    need_preprocess=1
    break
  fi
done

if [[ "$need_preprocess" -eq 1 ]]; then
  python preprocess_smplx.py
  python subdivide_smplx.py
  python utils_smplx.py
  python utils_uvpos.py
else
  echo "[INFO] template npy already exists, skip preprocess"
fi
popd >/dev/null

echo "[7/8] Fix .ckpt symlink"
if [[ ! -e .ckpt ]]; then
  ln -s ckpt .ckpt
fi

echo "[8/8] Run inference"
python scripts/test_DiT.py --image_path "$INPUT_IMAGE" --pose_path "$POSE_PATH"

echo "[DONE] Outputs: $REPO_DIR/workspace/outputs"
