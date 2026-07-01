# Run tests

テストは「CPUだけで常に確認するもの」と「GPUがある時だけ確認するもの」に分けます。

## CPUテスト

```bash
uv run pytest
```

通常のCIではこの範囲を実行します。

`pytest` のentrypoint解決で問題がある場合は、同じ環境で次のようにmoduleとして実行できます。

```bash
uv run python -m pytest
```

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
