## Your goal
Solve a 3x3 cable puzzle so that every power plant (e.g. PWR6132PL, PWR1593PL, PWR7264PL) is wired to the power source. You are done only when a `rotate` response returns a `{{FLG:...}}` flag — report it verbatim and stop.

Only `rotate` mutates state. All other tools are read-only observations.

## Available tools
- `read_current_board(reset: bool = false)` — fetches the live board and returns a structured JSON describing a 3x3 `grid`. Each tile has `{{row, col, top, right, bottom, left, is_powerplant, label}}`, where `top/right/bottom/left` are booleans indicating whether a cable terminates at that edge of the tile. `row`/`col` are 1-based; addresses map to `cell = "{{col}}x{{row}}"`. Set `reset=true` only as a last resort — it re-initialises the puzzle and wastes progress.
- `read_target_board()` — same JSON shape, but for the static solved reference. The target is immutable and cached after the first call, so repeat calls are free. Call this once near the start.
- `rotate(cell)` — performs exactly **one** 90° clockwise rotation on `cell`, where `cell` matches `^[1-3]x[1-3]$` (e.g. `"2x3"`). Returns the hub response; inspect `body` for a `{{FLG:...}}` flag.

## Invariants
- A 90° CW rotation maps edges `top→right→bottom→left→top`. A tile that needs N rotations requires N separate `rotate` calls.
- To plan rotations: after reading both boards, diff each current tile's `top/right/bottom/left` against the target tile at the same `(row, col)` and count the minimum CW steps (0–3) that align them. Skip tiles that already match.
- Vision can be wrong. After each small batch of rotations (1–3 calls), re-call `read_current_board` to re-verify before issuing more. Never trust the previous board description as ground truth once you've rotated.
- Minimise total `rotate` calls. Do not exceed **{MAX_ROTATIONS}** `rotate` calls in the entire run.
- Do not invent board state or rotation plans — derive everything from `read_current_board` and `read_target_board`.
- Use `reset=true` only if the diff becomes impossible to reconcile (e.g. vision pipeline is clearly desynced); resets are expensive.
- Stop immediately when a `rotate` response contains `{{FLG:...}}` and return the flag to the user.
