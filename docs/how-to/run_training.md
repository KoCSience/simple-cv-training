# Run training

移行中は `main_pl.py` をHydra入口として使います。最終的には `scv train` に統一します。

## Cometを無効にして実行する

```bash
uv run python main_pl.py dataset.dataset_name=ZeroImages training.num_epochs=1 disable_comet=true
```

## GPUを指定する

Linuxでは `CUDA_VISIBLE_DEVICES` で見せるGPUを制限します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python main_pl.py dataset.dataset_name=ZeroImages training.num_epochs=1 disable_comet=true
```

複数GPUを使う場合:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python main_pl.py GPU.devices=2 dataset.dataset_name=ZeroImages disable_comet=true
```

## checkpointから再開する

```bash
uv run python main_pl.py checkpoint_file.checkpoint_to_resume=/path/to/checkpoint.ckpt disable_comet=true
```

checkpointのパスは存在するファイルを指定してください。存在しない場合は実行前にエラーにします。
