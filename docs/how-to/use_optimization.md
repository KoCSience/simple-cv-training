# Use optimization options

`optimization` は、高速化を比較実験として有効化するための設定グループです。初期状態では baseline の理解と再現性を優先するため、AMP と `torch.compile` は有効化しません。

## Baselineを実行する

何も指定しない場合、最適化は実行されません。

```bash
uv run python main.py data=zero_images trainer=smoke disable_comet=true
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true
```

合成後の設定は次で確認できます。

```bash
uv run python main.py --cfg job
uv run python main_pl.py --cfg job
```

## bf16 AMPを使う

bf16 mixed precision は Lightning の `Trainer.precision` に `bf16-mixed` を渡して有効化します。

```bash
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true optimization.amp.enabled=true
```

`optimization.amp.precision` の既定値は `bf16-mixed` です。bf16 は Turing 世代 GPU では対応していません。Ampere 世代以降の対応 GPU を想定しますが、実装では GPU 世代名ではなく PyTorch の `torch.cuda.is_bf16_supported()` の結果を優先します。

bf16 未対応環境で `optimization.amp.enabled=true` を指定すると、既定では実行前に分かりやすく失敗します。Lightning 側の挙動に委ねたい場合だけ、次のように明示します。

```bash
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true optimization.amp.enabled=true optimization.amp.require_bf16_supported=false
```

fp16 を試す場合:

```bash
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true optimization.amp.enabled=true optimization.amp.precision=16-mixed
```

## main.pyとmain_pl.pyの対応範囲

`main_pl.py` は PyTorch Lightning の入口であり、AMP と `torch.compile` の標準 opt-in 経路です。Lightning が precision 管理を持っているため、`optimization.amp.enabled=true` は `main_pl.py` で使う想定です。

`main.py` は手動 PyTorch loop の教材用入口です。ここに AMP を入れると、`train.py` 側に `autocast`、loss/backward、必要に応じて `GradScaler` の責務が入り、初学者向け loop の見通しが悪くなります。そのため、manual runner では AMP を標準対応にしません。

一方で `torch.compile` は model adapter として `prepare_manual_training()` のモデル生成後に薄く差し込めます。`main.py` では、`torch.compile` のみ opt-in で対応します。

`main.py` の手動 training 経路は次の順番です。

```text
main.py
  -> run_manual_training(cfg)
    -> prepare_manual_training(cfg)
      1. validate config
      2. configure logger
      3. configure dataloaders
      4. require CUDA
      5. create base model
      6. move base model to device
      7. create optimizer / scheduler from base model parameters
      8. optional checkpoint load into uncompiled base model
      9. optional torch.compile
     10. optional DataParallel
    -> train(...)
    -> validation(...)
    -> save checkpoint
```

checkpoint resume は通常の未 compile model に対して行います。これにより、既存の manual `.pt` checkpoint と互換性を保ちやすくなります。その後で `torch.compile` を薄い adapter として差し込み、最後に manual runner 専用の外側 wrapper として `DataParallel` を適用します。

## torch.compileを使う

`main_pl.py` では、`torch.compile` は LightningModule を `trainer.fit()` に渡す前に適用します。

```bash
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true optimization.compile.enabled=true
```

`main.py` では、checkpoint load 後、DataParallel 前に `torch.compile` を適用します。

```bash
uv run python main.py data=zero_images trainer=smoke disable_comet=true optimization.compile.enabled=true
```

mode を変更する例:

```bash
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true optimization.compile.enabled=true optimization.compile.mode=reduce-overhead
```

`torch.compile` は初回実行時にコンパイル時間がかかります。入力 shape が頻繁に変わる場合は再コンパイルが起き、速度が安定しないことがあります。

## 比較する

同じデータセット、モデル、batch size で条件だけを変えて比較します。

```bash
# baseline
uv run python main.py data=image_folder dataset.root=/path/to/Tiny-ImageNet trainer=smoke disable_comet=true
uv run python main_pl.py data=image_folder dataset.root=/path/to/Tiny-ImageNet trainer=smoke GPU.devices=1 disable_comet=true

# bf16 AMP
uv run python main_pl.py data=image_folder dataset.root=/path/to/Tiny-ImageNet trainer=smoke GPU.devices=1 disable_comet=true optimization.amp.enabled=true

# manual torch.compile
uv run python main.py data=image_folder dataset.root=/path/to/Tiny-ImageNet trainer=smoke disable_comet=true optimization.compile.enabled=true

# bf16 AMP + torch.compile
uv run python main_pl.py data=image_folder dataset.root=/path/to/Tiny-ImageNet trainer=smoke GPU.devices=1 disable_comet=true optimization.amp.enabled=true optimization.compile.enabled=true
```

高速化の効果は GPU、モデル、batch size、入力 shape、DataLoader 設定に依存します。速度だけでなく、loss と validation metric が baseline と大きくずれていないかも確認してください。
