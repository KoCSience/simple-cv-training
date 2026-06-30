# Linux setup

このプロジェクトは Linux 上での実行を前提にします。GPUを使う場合は、NVIDIA driver と CUDA 対応の PyTorch が利用できる状態にしてください。

## 1. 環境を同期する

```bash
uv sync
```

lockfileと完全に一致させる確認では次を使います。

```bash
uv sync --locked
```

## 2. GPUを確認する

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

`False` が出る場合、GPU必須テストやGPU学習は実行できません。その場合でもCPU向けの単体テストは実行できます。

## 3. 最小テストを実行する

```bash
uv run pytest
```

GPU必須テストも実行する場合:

```bash
uv run pytest --run-gpu
```

## 4. 最小学習を実行する

現在の互換入口では、Cometを無効にして次のように実行できます。

```bash
uv run python main_pl.py dataset.dataset_name=ZeroImages training.num_epochs=1 disable_comet=true
```

リファクタリング後は `scv train` を正式入口にします。移行中は互換のため `main.py` と `main_pl.py` を残します。
