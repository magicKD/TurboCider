# Sourced after changing to the repository root. Explicit overrides are optional.
if [ -z "${MLX_ROOT:-}" ]; then
  # 动态查找 Python 解释器，支持不同的环境（uv/conda/pyenv/venv）
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
