# Sourced after changing to the repository root. Explicit overrides are optional.
if [ -z "${MLX_ROOT:-}" ]; then
  PYTHON_CMD=""
  if [ -x "$PWD/.venv/bin/python3" ]; then
    PYTHON_CMD="$PWD/.venv/bin/python3"
  elif [ -x "$PWD/.deps/bin/python3" ]; then
    PYTHON_CMD="$PWD/.deps/bin/python3"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
  else
    echo 'Missing managed dependencies. Run: make setup' >&2
    exit 1
  fi
  MLX_ROOT="$("$PYTHON_CMD" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/mlx")')"
fi
if [ ! -f "$MLX_ROOT/include/mlx/mlx.h" ] || [ ! -f "$MLX_ROOT/lib/libmlx.dylib" ]; then
  echo 'MLX C++ headers/libraries missing. Run make setup or set an explicit MLX_ROOT.' >&2
  exit 1
fi
export MLX_ROOT

# Keep fallback and explicit deployment overrides in each build entry point.
turbocider_mlx_min_macos() {
  otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
    /cmd LC_BUILD_VERSION/{build=1; next}
    build && /minos /{print $2; exit}
  '
}

# Optional native RelayVSR runtime. The local slot is ignored by Git, and an
# explicit SDK path supports builds without an editable workspace.
TURBOCIDER_RELAY_TORCH_ROOT="${TURBOCIDER_RELAY_TORCH_ROOT:-}"
if [[ -z "$TURBOCIDER_RELAY_TORCH_ROOT" && -d "$PWD/.deps/relayvsr-torch" ]]; then
  TURBOCIDER_RELAY_TORCH_ROOT="$PWD/.deps/relayvsr-torch"
fi
if [[ -n "$TURBOCIDER_RELAY_TORCH_ROOT" ]]; then
  TURBOCIDER_RELAY_TORCH_ROOT="$(cd "$TURBOCIDER_RELAY_TORCH_ROOT" && pwd)"
  relay_version_header="$TURBOCIDER_RELAY_TORCH_ROOT/include/torch/headeronly/version.h"
  relay_version="$(awk '
    $2 == "TORCH_VERSION_MAJOR" {major=$3}
    $2 == "TORCH_VERSION_MINOR" {minor=$3}
    $2 == "TORCH_VERSION_PATCH" {patch=$3}
    END {print major "." minor "." patch}
  ' "$relay_version_header" 2>/dev/null || true)"
  if [[ "$relay_version" != "2.14.1" ]]; then
    echo 'RelayVSR requires the arm64 LibTorch 2.14.1 SDK (headers and dylibs).' >&2
    exit 1
  fi
  for relay_library in torch torch_cpu c10 omp; do
    if [[ ! -f "$TURBOCIDER_RELAY_TORCH_ROOT/lib/lib${relay_library}.dylib" ]]; then
      echo "Missing RelayVSR runtime library lib${relay_library}.dylib" >&2
      exit 1
    fi
  done
fi
