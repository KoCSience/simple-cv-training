# Run tests

テストは「CPUだけで常に確認するもの」と「GPUがある時だけ確認するもの」に分けます。

## CPUテスト

```bash
uv run pytest
```

通常のCIではこの範囲を実行します。

## GPUテスト

```bash
uv run pytest --run-gpu
```

GPUテストには `gpu` marker が付きます。モデルのforward、checkpointのDataParallel変換、optimizer/schedulerのGPUモデル利用などが対象です。

## 外部データセットテスト

NASやローカルの実データセットを必要とするテストは通常実行から外します。

```bash
uv run pytest --run-external-data
```

GPUも外部データセットも使う場合:

```bash
uv run pytest --run-gpu --run-external-data
```

## marker指定

GPUテストだけを選ぶ場合:

```bash
uv run pytest -m gpu --run-gpu
```

GPUテストと外部データセットテストを除外する場合:

```bash
uv run pytest -m "not gpu and not external_data"
```

## 失敗時の見方

- 設定値の失敗: `configs/` と Pydantic schema を確認する。
- GPU関連の失敗: `nvidia-smi` と `torch.cuda.is_available()` を確認する。
- checkpoint関連の失敗: 保存先、resume元、DataParallel有無を確認する。

## doctor

リポジトリの基本状態を確認するには次を使います。

```bash
uv run scv doctor
```

## 静的解析

現段階のCIでは、基本的なlint、format、import order、bugbear系ルールを `ruff` で確認し、型チェックは `ty` で確認します。
`ty` の対象は新しく追加したフレームワーク層を中心に限定しています。
既存の練習用コード全体はまだlint/type cleanではないため、テストで互換性を守りながら段階的に対象を広げます。

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

旧 `pep8` / `flake8` / `pylint` / `mypy` 設定からの移行差分は [static_analysis_migration.md](../reference/static_analysis_migration.md) に記録します。
