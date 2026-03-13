# Repository Guidelines

## Project Structure & Module Organization
`protomotions/` contains the Python package: `agents/` for RL algorithms, `envs/` and `components/` for task logic, `simulator/` for backend adapters (`isaacgym`, `isaaclab`, `newton`, `genesis`), `robot_configs/` for robot definitions, and `utils/` for shared helpers. Unless a task explicitly targets another backend, treat `newton` as the default simulator for local setup, commands, and validation. Tests live in `protomotions/tests/`. Experiment configs live under `examples/experiments/` by task family (`mimic/`, `masked_mimic/`, `amp/`, etc.). Static assets, pretrained checkpoints, and YAML motion configs are under `data/` and `protomotions/data/assets/`. Documentation sources are in `docs/source/`.

## Build, Test, and Development Commands
Install the package in editable mode inside a `newton` environment unless the task specifically requires another simulator:

```bash
pip install -e .
pip install -r requirements_newton.txt
```

Use the matching requirements file only when you are intentionally working on another backend (`requirements_isaacgym.txt`, `requirements_isaaclab.txt`, `requirements_genesis.txt`). Common workflows should default to `newton`:

```bash
python protomotions/train_agent.py --robot-name smpl --simulator newton --experiment-path examples/experiments/mimic/mlp.py --experiment-name my_run --motion-file path/to/motions.pt
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
Add or update `pytest` coverage when changing behavior, especially under `protomotions/simulator/`, `envs/`, or `robot_configs/`. Prefer `newton` for local verification unless the change is backend-specific, and keep non-`newton` assumptions explicit in the test name or docstring. There is no visible global coverage gate in this repo, so prefer targeted regression tests that prove the exact fix.

## Commit & Pull Request Guidelines
Recent history follows short conventional-style subjects such as `fix: ...`, `fix(data): ...`, and `docs(tutorial): ...`. Keep that format, and sign every commit with `git commit -s -m "fix: describe change"`. Pull requests should stay focused, explain what changed and why, note whether validation used `newton` or another simulator/backend, and include screenshots or short clips when UI, visualization, or rendered behavior changes.
