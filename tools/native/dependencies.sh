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
