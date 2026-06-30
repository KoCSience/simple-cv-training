# simple_cv_framework migration workspace

CNN/ViT学習コードを、Linux前提の `simple_cv_framework` へ段階的にリファクタリングしているリポジトリです。

## Quick start

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run scv train --cfg job
uv run scv doctor
```

GPUで最小学習を試す場合:

```bash
uv run scv train data=zero_images trainer=smoke
```

## Documentation

- Linux setup: [docs/getting-started/linux_setup.md](docs/getting-started/linux_setup.md)
- Run training: [docs/how-to/run_training.md](docs/how-to/run_training.md)
- Configure experiments: [docs/how-to/configure_experiment.md](docs/how-to/configure_experiment.md)
- Hydra usage: [docs/how-to/use_hydra.md](docs/how-to/use_hydra.md)
- Run tests: [docs/how-to/run_tests.md](docs/how-to/run_tests.md)
- Static analysis migration: [docs/reference/static_analysis_migration.md](docs/reference/static_analysis_migration.md)
- Add a model: [docs/how-to/add_model.md](docs/how-to/add_model.md)
- Add a dataset: [docs/how-to/add_dataset.md](docs/how-to/add_dataset.md)
- Project structure: [docs/explanation/project_structure.md](docs/explanation/project_structure.md)

## Safety rules

- Do not read or commit `.env` files.
- Do not write API keys in code, docs, logs, snapshots, or commits.
- Use `uv lock` for dependency lock updates.
- Use `git mv` when moving tracked files.
