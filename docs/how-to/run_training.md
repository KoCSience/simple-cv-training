# Run training

`main.py` と `main_pl.py` はどちらも Hydra 入口です。`main.py` は手動 PyTorch loop、`main_pl.py` は PyTorch Lightning loop を実行します。最終的には `scv train` に統一します。

## Cometを無効にして実行する

```bash
uv run python main.py data=zero_images trainer=smoke disable_comet=true
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true
```

## GPUを指定する

Linuxでは `CUDA_VISIBLE_DEVICES` で見せるGPUを制限します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python main.py data=zero_images trainer=smoke disable_comet=true
CUDA_VISIBLE_DEVICES=0 uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true
```

複数GPUを使う場合:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python main.py data=zero_images trainer=smoke GPU.use_dp=true disable_comet=true
CUDA_VISIBLE_DEVICES=0,1 uv run python main_pl.py data=zero_images trainer=smoke GPU.devices=2 disable_comet=true
```

## checkpointから再開する

```bash
uv run python main.py checkpoint_file.checkpoint_to_resume=/path/to/checkpoint.pt disable_comet=true
uv run python main_pl.py checkpoint_file.checkpoint_to_resume=/path/to/checkpoint.ckpt disable_comet=true
```

`main.py` は独自の `.pt` checkpoint、`main_pl.py` は Lightning の `.ckpt` checkpoint を使います。checkpointのパスは存在するファイルを指定してください。存在しない場合は実行前にエラーにします。

## 最適化を試す

既定では baseline の再現性を優先し、AMP と `torch.compile` は有効化しません。Lightning 経路で高速化を比較したい場合は、明示的に有効化します。

```bash
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true optimization.amp.enabled=true
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true optimization.compile.enabled=true
```

詳しくは [Use optimization options](use_optimization.md) を参照してください。
