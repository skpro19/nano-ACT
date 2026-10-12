---
description: "Sweep config filenames must mark swept params with + and fixed params with -"
alwaysApply: true
---

## Sweep Config Filenames
- `configs/sweep/*.toml` filenames encode each parameter's value with a prefix that reflects its role in the spec:
  - `+<param>` for every parameter varied in `[grid]` (the thing being swept)
  - `-<param>` for every parameter pinned in `[fixed]`
- Use the `+` prefix **only** for parameters that are actually swept. A fixed value must never carry `+`.
- Examples:
  - `ph+action-mode+k.toml` — `action_mode` and `action_chunk_size` (`k`) are both in `[grid]`
  - `ph+action-mode+k-beta10.toml` — `action_mode` and `k` swept, `beta = 10` fixed
  - `ph100-k100+use-z.toml` — `use_z` swept, `k = 100` fixed
