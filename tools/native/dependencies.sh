# Sourced after changing to the repository root. Explicit overrides are optional.
if [ -z "${MLX_ROOT:-}" ]; then
  if [ ! -x "$PWD/.venv/bin/python3" ]; then
    echo 'Missing managed dependencies. Run: make setup' >&2
    exit 1
  fi
  MLX_ROOT="$("$PWD/.venv/bin/python3" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/mlx")')"
fi
if [ ! -f "$MLX_ROOT/include/mlx/mlx.h" ] || [ ! -f "$MLX_ROOT/lib/libmlx.dylib" ]; then
  echo 'MLX C++ headers/libraries missing. Run make setup or set an explicit MLX_ROOT.' >&2
  exit 1
fi
export MLX_ROOT
