# Common Commands

Run these commands from the repository root unless noted otherwise.

## Install And Setup

```bash
pip install -e .
pip install -r requirements_isaacgym.txt
git lfs fetch --all
```

Swap in the simulator-specific requirements file you need:
`requirements_newton.txt`, `requirements_isaaclab.txt`, or
`requirements_genesis.txt`.

## Quality Checks

```bash
pre-commit run --all-files
pytest protomotions/tests -q
make -C docs html
```

## Train And Run Inference

```bash
python protomotions/train_agent.py \
  --robot-name smpl \
  --simulator isaacgym \
  --experiment-path examples/experiments/mimic/mlp.py \
  --experiment-name my_run \
  --motion-file data/motions/g1_random_subset_tiny.pt
```

```bash
python protomotions/inference_agent.py \
  --checkpoint data/pretrained_models/motion_tracker/g1-amass/last.ckpt \
  --motion-file data/motions/g1_random_subset_tiny.pt \
  --simulator newton
```

## End-To-End Human Retargeting Pipeline

This runs the biomechanics pipeline that converts treadmill motion capture data
into a packaged ProtoMotions MotionLib. Pass `--pyroki-python` when the PyRoki
environment is not auto-detected.

```bash
python HumanRetargeting/biomechanics_retarget/pipeline.py \
  HumanRetargeting/biomechanics_retarget/treadmill_data/S02 \
  HumanRetargeting/biomechanics_retarget/processed_data/S02 \
  --height 156 \
  --fps 200 \
  --output-fps 30 \
  --pyroki-python <path-to-pyroki-python>
```

Useful variants:

```bash
python HumanRetargeting/biomechanics_retarget/quick_rescale.py --height 156
```

```bash
python HumanRetargeting/biomechanics_retarget/pipeline.py \
  HumanRetargeting/biomechanics_retarget/treadmill_data/S02 \
  HumanRetargeting/biomechanics_retarget/processed_data/S02 \
  --model HumanRetargeting/rescale/smpl_humanoid_lower_body_adjusted_pd.xml \
  --fps 200 \
  --output-fps 30 \
  --pyroki-python <path-to-pyroki-python>
```

For the detailed walkthrough, see
`HumanRetargeting/biomechanics_retarget/README.md`.

## End-To-End AMASS To Robot Retargeting

Use the convenience script to run the full five-step AMASS retargeting flow:
extract keypoints, retarget in PyRoki, extract contacts, convert to
ProtoMotions format, and package the final `.pt` MotionLib.

```bash
./scripts/retarget_amass_to_robot.sh \
  <path-to-protomotions-python> \
  <path-to-pyroki-python> \
  /path/to/amass_train.pt \
  /path/to/output \
  g1 \
  50
```

Set the final argument to `1` to process every motion instead of a sampled
subset.

## Retarget A Single Motion File

```bash
./scripts/retarget_single_motion_to_robot.sh \
  <path-to-protomotions-python> \
  <path-to-pyroki-python> \
  /path/to/walk.motion \
  /path/to/output \
  g1
```

## Visualize Retargeted Results

Use the motion library visualizer after AMASS retargeting:

```bash
python examples/motion_libs_visualizer.py \
  --motion_files /path/to/output/retargeted_g1.pt \
  --robot g1 \
  --simulator isaacgym
```

Use kinematic playback after single-motion retargeting:

```bash
python examples/env_kinematic_playback.py \
  --experiment-path examples/experiments/mimic/mlp.py \
  --motion-file /path/to/output/retargeted_g1_proto/walk.motion \
  --robot-name g1 \
  --simulator isaacgym \
  --num-envs 1
```
