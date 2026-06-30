# Run training

正式入口は `scv train` です。移行中の互換入口として `main_pl.py` も残します。

## Cometを無効にして実行する

```bash
uv run scv train data=zero_images trainer=smoke
```

## GPUを指定する

Linuxでは `CUDA_VISIBLE_DEVICES` で見せるGPUを制限します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run scv train data=zero_images trainer=smoke
```

複数GPUを使う場合:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run scv train data=zero_images trainer=smoke GPU.devices=2
```

## checkpointから再開する

```bash
uv run scv train checkpoint_file.checkpoint_to_resume=/path/to/checkpoint.ckpt disable_comet=true
```

checkpointのパスは存在するファイルを指定してください。存在しない場合は実行前にエラーにします。
