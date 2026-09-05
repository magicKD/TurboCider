# Native engine installation directory

This directory is the self-contained location for the native inference
engines that TurboCider orchestrates. It is intentionally not committed;
use the TurboCider bootstrap to populate it.

Expected layout after bootstrapping:

```text
engines/
  h3/          H3 inference engine (binaries and tools)
  ltx-mac/     LTX-2.5 native two-stage engine
  flux2/       FLUX.2 MLX + ANE runtime
  fastmetal/   FastMetal MLX runtime
```

You can either:

- point `TURBOCIDER_ENGINES_DIR` at an existing installation, or
- run `turbocider bootstrap --source <engine-source-root>` to prepare this
  directory from engine source repositories.

TurboCider model packs use `${TURBOCIDER_ENGINES}` for engine paths so a
fresh checkout never depends on a monorepo sibling layout.
