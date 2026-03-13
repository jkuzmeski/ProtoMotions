# Repository Guidelines

## Project Structure & Module Organization
`protomotions/` contains the Python package: `agents/` for RL algorithms, `envs/` and `components/` for task logic, `simulator/` for backend adapters (`isaacgym`, `isaaclab`, `newton`, `genesis`), `robot_configs/` for robot definitions, and `utils/` for shared helpers. Tests live in `protomotions/tests/`. Experiment configs live under `examples/experiments/` by task family (`mimic/`, `masked_mimic/`, `amp/`, etc.). Static assets, pretrained checkpoints, and YAML motion configs are under `data/` and `protomotions/data/assets/`. Documentation sources are in `docs/source/`.

## Build, Test, and Development Commands
Install the package in editable mode inside a simulator-specific environment:

```bash
pip install -e .
pip install -r requirements_isaacgym.txt
```

Use the matching requirements file for your backend (`requirements_newton.txt`, `requirements_isaaclab.txt`, `requirements_genesis.txt`). Common workflows:

```bash
python protomotions/train_agent.py --robot-name smpl --simulator isaacgym --experiment-path examples/experiments/mimic/mlp.py --experiment-name my_run --motion-file path/to/motions.pt
python protomotions/inference_agent.py --checkpoint data/pretrained_models/motion_tracker/g1-amass/last.ckpt --motion-file data/motions/g1_random_subset_tiny.pt --simulator newton
pytest protomotions/tests -q
make -C docs html
```

Run `git lfs fetch --all` after cloning to pull large tracked assets.

## Coding Style & Naming Conventions
Follow documented `pre-commit` checks and run them before review:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The repo uses Ruff for linting and formatting. Match existing Python style: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, explicit docstrings on new public code, and focused commits with one logical change each.

## Testing Guidelines
Add or update `pytest` coverage when changing behavior, especially under `protomotions/simulator/`, `envs/`, or `robot_configs/`. Name tests `test_<behavior>.py` and keep backend-specific assumptions in the test name or docstring. There is no visible global coverage gate in this repo, so prefer targeted regression tests that prove the exact fix.

## Commit & Pull Request Guidelines
Recent history follows short conventional-style subjects such as `fix: ...`, `fix(data): ...`, and `docs(tutorial): ...`. Keep that format, and sign every commit with `git commit -s -m "fix: describe change"`. Pull requests should stay focused, explain what changed and why, note which simulator/backend was exercised, and include screenshots or short clips when UI, visualization, or rendered behavior changes.
