# v1 training

This folder contains the curriculum-ready environment and training script for manual scenario switching.

## Quick start (manual switching)

Train 1v1 only:

```bash
python src/v1/train.py --start-task 1v1 --max-task 1v1
```

Switch to 3v3 and resume from the latest checkpoint:

```bash
python src/v1/train.py --start-task 3v3 --max-task 3v3 --resume-latest
```

Switch to 11v11 and resume:

```bash
python src/v1/train.py --start-task 11v11 --max-task 11v11 --resume-latest
```

## Optional: automatic curriculum switching

If you want the environment to auto-advance based on reward thresholds, enable:

```bash
python src/v1/train.py --auto-curriculum --start-task 1v1 --max-task 11v11
```

## Checkpointing

Save checkpoints every 25 iterations (default):

```bash
python src/v1/train.py --checkpoint-dir runs/checkpoints --checkpoint-every 25
```

Resume from a specific checkpoint:

```bash
python src/v1/train.py --resume-from /path/to/checkpoint
```

Change where checkpoints are saved:

```bash
python src/v1/train.py --checkpoint-dir runs/checkpoints_v1
```

## Bootstrap role policies from a generic checkpoint

1) Train a generic policy and note the checkpoint path printed by the script.
2) Use that checkpoint to initialize role policies:

```bash
python src/v1/train.py --role-policies --bootstrap-from /path/to/checkpoint
```

## Save evaluation frames

Save evaluation frames every 25 iterations (default):

```bash
python src/v1/train.py --save-frames --eval-every 25 --eval-episodes 1 --eval-steps 300 --frames-dir runs/frames
```

Frames are saved under `runs/frames/iter_XXXX/frame_<episode>_<step>.png`.

Change where frames are saved:

```bash
python src/v1/train.py --save-frames --frames-dir runs/frames_v1
```

## Custom layouts (pitch size, player spawns, ball)

Layouts are loaded from JSON. Default path is `src/v1/layouts/<task>.json`.

Use the pitch editor to create a layout:

```bash
python src/v1/pitch_editor.py --task 3v3
```

Save to a custom file:

```bash
python src/v1/pitch_editor.py --task 1v1 --width 1000 --height 600 --out src/v1/layouts/1v1.json
```

When training, point to the layout directory or a specific file:

```bash
python src/v1/train.py --start-task 3v3 --max-task 3v3 --resume-latest \
	--env-config-layout-dir src/v1/layouts
```

Or set a single layout file:

```bash
python src/v1/train.py --start-task 1v1 --max-task 1v1 --env-config-layout-path src/v1/layouts/1v1.json
```

## Policy init JSON map

The policy inheritance map is stored in [src/v1/policy_init_map.json](src/v1/policy_init_map.json).

Example:

```json
{
	"policy_generic": {"source_policy": "policy_generic", "checkpoint": null},
	"policy_target_man": {"source_policy": "policy_generic", "checkpoint": null},
	"policy_playmaker": {"source_policy": "policy_playmaker", "checkpoint": "/path/to/playmaker_ckpt"},
	"policy_anchor": {"source_policy": "policy_anchor", "checkpoint": "/path/to/anchor_ckpt"}
}
```

Use the map:

```bash
python src/v1/train.py --role-policies --init-from-map --init-from /path/to/default_checkpoint
```

Notes:

- `checkpoint: null` falls back to `--init-from`.
- `source_policy` must exist inside the referenced checkpoint.
- Use `--policy-map-file /path/to/map.json` to load a different map.

## Flags

- `--start-task`: Starting scenario. Use `1v1`, `3v3`, or `11v11`.
- `--max-task`: Maximum scenario allowed in this run. Use `1v1`, `3v3`, or `11v11`.
- `--auto-curriculum`: Enable reward-threshold based switching. If unset, switching is manual.
- `--role-policies`: Use role-specific policies. If unset, all agents use `policy_generic`.
- `--bootstrap-from`: Path to a checkpoint used to initialize role policies from `policy_generic`.
- `--init-from`: Default checkpoint used by the policy init map when a policy does not provide its own `checkpoint`.
- `--init-from-map`: Apply the JSON policy init map to set per-policy initialization.
- `--force-init-map`: Apply the init map even when resuming from a checkpoint.
- `--policy-map-file`: Path to the JSON policy init map. Default: `src/v1/policy_init_map.json`.
- `--env-config-layout-path`: Use a single layout JSON for the selected task.
- `--env-config-layout-dir`: Load per-task layout JSON files from a directory.
- `--resume-from`: Resume training from a specific checkpoint.
- `--resume-latest`: Resume from the most recent checkpoint in `--checkpoint-dir`.
- `--iters`: Number of training iterations. Default: 500.
- `--checkpoint-dir`: Directory for checkpoints. Default: `runs/checkpoints`.
- `--checkpoint-every`: Save a checkpoint every N iterations. Default: 25. Use `0` to disable.
- `--save-frames`: Enable evaluation frame capture during training.
- `--frames-dir`: Directory for frame output. Default: `runs/frames`.
- `--eval-every`: Save frames every N iterations. Default: 25. Use `0` to disable.
- `--eval-episodes`: Number of eval episodes per capture. Default: 1.
- `--eval-steps`: Max steps per eval episode. Default: 300.

## Recommended workflow

1) Train 1v1 generic: `--start-task 1v1 --max-task 1v1`.
2) Resume on 3v3: `--start-task 3v3 --max-task 3v3 --resume-latest`.
3) Resume on 11v11: `--start-task 11v11 --max-task 11v11 --resume-latest`.
4) Optionally bootstrap role policies from any checkpoint with `--role-policies --bootstrap-from`.
