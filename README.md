# toy-act

A minimal implementation of Action Chunking with Transformers (ACT) for the robosuite PickPlaceCan task, using robomimic demonstrations. Includes CVAE-based policy training, multi-camera inputs, and rollout evaluation.

<p align="center">
  <img src="assets/rollout/bs64_two_camera_rollout.gif" alt="Two-camera ACT policy rollout on the Can task">
</p>

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Run commands from the repository root.

```bash
uv sync
```

### Prepare the data

Download the robomimic **Can / proficient-human (ph) / low_dim** dataset using the [robomimic dataset guide](https://robomimic.github.io/docs/datasets/overview.html) and place it at `datasets/can/ph/low_dim_v15.hdf5`. Render the two camera views used by the example training configuration:

```bash
uv run python -m scripts.extract_act_dataset \
    --input datasets/can/ph/low_dim_v15.hdf5 \
    --output datasets/can/ph/act_two_camera.hdf5 \
    --cameras agentview robot0_eye_in_hand
```

Rendering requires a working MuJoCo/OpenGL setup.

### Train

```bash
uv run python -m scripts.train_v2 \
    --config configs/train/act_v2/BS-32/BASE.toml \
    --dataset datasets/can/ph/act_two_camera.hdf5
```

Training settings live in the TOML configuration. Checkpoints are saved under `checkpoints/act_v2/`, and logs under `runs/act_v2/`.

### Evaluate

Replace `<checkpoint.pt>` with a checkpoint saved during training:

```bash
uv run python -m scripts.rollout \
    --checkpoint <checkpoint.pt> \
    --dataset datasets/can/ph/low_dim_v15.hdf5 \
    --n-rollouts 20
```

Evaluation opens a live viewer by default. Add `--no-on-screen` for headless evaluation (requires EGL support).

## Code

- `scripts/models/act_v2/` — ACT policy implementation.
- `scripts/train_v2.py` — training.
- `scripts/rollout.py` — rollout evaluation.
- `configs/train/` and `configs/sweep/` — training and ablation configurations.

## Training curve

<table align="center">
  <thead>
    <tr><th>lr</th><th>bs</th><th>num-steps</th><th>beta</th><th>action-loss</th><th>architecture</th></tr>
  </thead>
  <tbody>
    <tr><td>1e-4</td><td>64</td><td>100000</td><td>0.01</td><td>l1</td><td>cvae encoder-decoder</td></tr>
  </tbody>
</table>

<img src="assets/training-run/eval_uniform_panels.png" alt="Evaluation episode horizon and success rate over training steps">

<img src="assets/training-run/batch_metrics_log_scale_uniform_panels.png" alt="Smoothed action, weighted KL, and total losses over training steps (log scale)">

<img src="assets/training-run/denorm_l1_log_scale_uniform_panels.png" alt="Smoothed denormalized joint and gripper L1 errors over training steps (log scale)">

<img src="assets/training-run/latent_log_scale_uniform_panels.png" alt="Latent mu norm and sigma mean over training steps (log scale)">

## Ablations

### Batch size

<img src="assets/eval-curves/batch_size_eval_curves_legend_right.png" alt="Batch-size ablation eval curves">

<p align="center">
  <img src="assets/rollout-two-camera/two_camera_rollout_grid.gif" alt="Batch-size ablation rollout comparison">
</p>

### Number of cameras

<img src="assets/eval-curves/num_cameras_eval_curves_legend_right.png" alt="Num-cameras ablation eval curves">

<p align="center">
  <img src="assets/rollout-camera-ablation/num_cameras_grid.gif" alt="Num-cameras ablation rollout comparison">
</p>

### Action chunk size

<img src="assets/eval-curves/action_chunk_size_eval_curves_bs8_legend_right.png" alt="Action chunk size ablation eval curves">

### Beta

<img src="assets/eval-curves/action_chunk_size_best_success_by_beta_bs8_legend.png" alt="Best eval success rate for action chunk size across beta">

## Acknowledgments

Based on [ACT](https://tonyzhaozh.github.io/aloha/), with demonstrations and simulation provided by [robomimic](https://robomimic.github.io/) and [robosuite](https://robosuite.ai/).
