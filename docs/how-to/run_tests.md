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

### VideoFolderのmultiprocessingテスト

`LimitDataset` のworker起動前の状態、複数epoch、2 passの補完処理は、外部動画を使わない単体テストで確認できます。

```bash
uv run python -m pytest test/dataset/test_limit_dataset.py
```

このテストは、利用可能な環境では `spawn` と `forkserver` の両方で `num_workers > 0` のDataLoaderを起動します。Python 3.14環境を明示して確認する場合は、次を実行します。

```bash
uv run --python 3.14 python -m pytest test/dataset/test_limit_dataset.py
```

実際の動画decode、transform、batch shapeまで含めた結合確認には、外部データセットテストを実行します。

```bash
uv run python -m pytest test/dataset/test_video_folder.py --run-external-data
```

`num_workers=0` は、workerへのDatasetのserializeを行わないため、エラー箇所を調べるデバッグ設定としては有効ですが、multiprocessingのpickle回帰を修正できたことの確認にはなりません。

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
