# Use optimization options

`optimization` は、高速化を比較実験として有効化するための設定グループです。初期状態では baseline の理解と再現性を優先するため、AMP と `torch.compile` は有効化しません。

## Baselineを実行する

何も指定しない場合、最適化は実行されません。

```bash
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true
```

合成後の設定は次で確認できます。

```bash
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

## torch.compileを使う

`torch.compile` は LightningModule を `trainer.fit()` に渡す前に適用します。

```bash
uv run python main_pl.py data=zero_images trainer=smoke disable_comet=true optimization.compile.enabled=true
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
uv run python main_pl.py data=image_folder dataset.root=/path/to/Tiny-ImageNet trainer=smoke GPU.devices=1 disable_comet=true

# bf16 AMP
uv run python main_pl.py data=image_folder dataset.root=/path/to/Tiny-ImageNet trainer=smoke GPU.devices=1 disable_comet=true optimization.amp.enabled=true

# bf16 AMP + torch.compile
uv run python main_pl.py data=image_folder dataset.root=/path/to/Tiny-ImageNet trainer=smoke GPU.devices=1 disable_comet=true optimization.amp.enabled=true optimization.compile.enabled=true
```

高速化の効果は GPU、モデル、batch size、入力 shape、DataLoader 設定に依存します。速度だけでなく、loss と validation metric が baseline と大きくずれていないかも確認してください。
