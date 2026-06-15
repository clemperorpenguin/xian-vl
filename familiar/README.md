# Familiar art drop-in

Sprite art for MAGE's desktop companion. Everything here is **optional** — each
familiar falls back to a built-in vector placeholder for any pose you don't
provide, so you can add art one file at a time.

## Where files go

```
familiar/<species>/<state>_<n>.png
```

- **species**: `wizard`, `witch`, `cat`, `owl`, `lemonfae`
- **state** (mood): `idle`, `walk`, `cast`, `react`, `sad`
- **state** (optional travel poses): `ascend`, `perch`, `descend`
- **n**: frame number, starting at `0`, contiguous (a gap stops loading —
  `walk_0, walk_1` is fine; `walk_0, walk_2` only loads `walk_0`).

Example:
```
familiar/witch/idle_0.png   familiar/witch/idle_1.png
familiar/witch/walk_0.png   familiar/witch/walk_1.png
familiar/witch/cast_0.png
familiar/witch/react_0.png
familiar/witch/sad_0.png
familiar/witch/ascend_0.png  (broom take-off; optional)
```

## Image spec

| Spec | Value |
|------|-------|
| Format | PNG, RGBA, **transparent background** |
| Author at | **256×256** (downscaled cleanly; crisp on hi-DPI) |
| Shown on screen at | 128×128 |
| Facing | **right** (engine auto-mirrors for walking left) |
| Position | **feet near the bottom edge**, small headroom on top |

Keep the standing position consistent across every frame/state so it doesn't
jitter when animating or switching poses.

## When each pose shows

| State | Trigger |
|-------|---------|
| `idle`  | standing between wanders |
| `walk`  | strolling along the floor (also reused while travelling if no travel art) |
| `cast`  | a translation is running ("thinking") |
| `react` | a translation finished (success) |
| `sad`   | a translation errored |
| `ascend` / `perch` / `descend` | travelling up to the top of the screen, sitting there, coming back down |

## Travel poses by species

- **wizard, lemonfae** — teleport: the engine draws the blink + sparkle effect,
  so they **don't need** `ascend`/`descend` art.
- **witch** — rides a broom up and down (`ascend`/`descend`).
- **owl** — flies with wings spread (`ascend`/`descend`).
- **cat** — climbs the nearest screen edge (`ascend`/`descend`).

Animation plays at ~140 ms per frame. 2–4 frames per state looks lively; a
single frame is a fine static pose.
