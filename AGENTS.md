# Repository guidance

## Python tooling

Use `uv` throughout this repository. Manage the project environment and
dependencies with `uv sync`, and run Python scripts, tests, and the CLI with
`uv run` (for example, `uv run python convert.py`, `uv run pytest`, and
`uv run mmlx_whisper`). Keep documentation and generated model-card examples
consistent with this convention. For installation outside a project checkout,
use `uv venv` and `uv pip install`.

Package metadata lives in `pyproject.toml`; `setup.py` is a compatibility shim.
Keep `uv.lock` in sync when changing dependencies.

The console command is `mmlx_whisper`. The Python distribution and import
remain `mlx-whisper` and `mlx_whisper`, respectively.

## Git publishing

When the current checkout is on `main` and Maciej asks to push changes, commit
the requested changes and push directly to `origin/main`. Do not create a branch
or pull request unless Maciej explicitly asks for one. After pushing, verify
that local `main` and `origin/main` point to the same commit.
