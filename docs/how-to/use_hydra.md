# Use Hydra

Hydraは、複数のYAML設定を合成して1つの実験設定を作るために使います。

## 基本

```bash
uv run scv train training.num_epochs=1 disable_comet=true
```

`training.num_epochs=1` のように、ドットで階層を指定して値を変更します。

## defaults list

`configs/train.yaml` に defaults list を置きます。

```yaml
defaults:
  - _self_
  - data: cifar10
  - model: resnet18
  - optimizer: sgd
  - trainer: default
  - mode: beginner
```

## modeの使い分け

- `beginner`: 最小構成。ログや外部連携を抑えて動作確認しやすくする。
- `researcher`: 通常研究用。GPU、logger、callbackを使う。
- `reproduce`: 再現用。seedやdeterministic設定を重視する。
- `framework-dev`: フレームワーク開発用。path dependencyを許可する。

## experimentファイル

繰り返し使う設定は `configs/experiment/*.yaml` に保存します。

```bash
uv run scv train experiment=cls_resnet50_cifar10
```

設定の合成結果だけを見る場合:

```bash
uv run scv train --cfg job
```
