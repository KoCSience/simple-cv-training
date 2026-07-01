# Config schema

設定はHydraで合成し、`src/simple_cv_training/config/schema.py` のPydantic schemaで型と値を検証します。

## 主なグループ

- `dataset`: データセット名、root、train/valディレクトリ
- `model`: モデル名、事前学習重み、重み保存先
- `video`: 動画clip設定
- `training`: batch size、worker数、epoch数、ログ間隔
- `optimizer`: optimizer名、learning rate、weight decay、scheduler
- `optimization`: AMP、`torch.compile`、benchmark、環境情報ログ
- `GPU`: DataParallel利用有無、Lightning devices
- `log_dirs`: Comet、TensorBoardの保存先
- `checkpoint_file`: checkpoint保存先、resume元
- `mode`: beginner、researcher、reproduce、framework-dev

## 代表的な制約

- `training.batch_size`: 1以上
- `training.num_workers`: 0以上
- `training.num_epochs`: 1以上
- `optimizer.lr`: 0より大きい
- `optimizer.grad_accum`: 1以上
- `video.frames_per_clip`: 1以上
- `checkpoint_file.checkpoint_to_resume`: `null`、`experiment:` で始まるComet参照、または存在するファイル
- `optimization.amp.precision`: `32-true`、`16-mixed`、`bf16-mixed`
- `optimization.compile.mode`: `default`、`reduce-overhead`、`max-autotune`、`max-autotune-no-cudagraphs`

## optimization

`optimization` は常に compose されますが、初期状態では高速化を有効化しません。初学者がまず baseline を動かし、変更した設定と結果の差分を理解できるようにするためです。

既定値:

```yaml
optimization:
  amp:
    enabled: false
    precision: bf16-mixed
    require_bf16_supported: true
  compile:
    enabled: false
    mode: default
    fullgraph: false
    backend: null
    dynamic: null
  benchmark:
    enabled: true
  environment:
    log: true
```

`optimization.amp.enabled=true` のときだけ Lightning Trainer に precision を渡します。`bf16-mixed` は Turing 世代 GPU では対応していません。Ampere 世代以降の対応 GPU を想定しますが、実行時の判定は PyTorch の `torch.cuda.is_bf16_supported()` を優先します。

`optimization.compile.enabled=true` のときだけ、LightningModule を `trainer.fit()` に渡す前に `torch.compile` で包みます。

beginner mode では baseline を優先し、researcher mode や `configs/experiment/` では必要な最適化だけを明示的に有効化してください。

## 検証方針

- batch size、epoch、worker数は0以上または1以上の範囲を検証する。
- optimizer名とmodel名は許可リストで検証する。
- checkpoint resume元は存在するファイルか確認する。
- 秘密情報は設定schemaに含めない。
