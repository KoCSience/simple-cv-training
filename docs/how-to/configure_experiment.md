# Configure experiments

実験設定はコード中の分岐ではなく、設定ファイルとHydra overrideで変更します。

## よく変更する項目

データセット:

```bash
uv run python main_pl.py dataset.dataset_name=ImageFolder dataset.root=/data/tiny-imagenet
```

モデル:

```bash
uv run python main_pl.py model.model_name=resnet50 model.use_pretrained=false
```

optimizer:

```bash
uv run python main_pl.py optimizer.optimizer_name=Adam optimizer.lr=1e-4
```

学習回数:

```bash
uv run python main_pl.py training.num_epochs=50
```

## 設定変更の考え方

- 何度も使う実験は `configs/experiment/` に保存する。
- その場だけの変更はコマンドラインoverrideで指定する。
- 型や範囲の検証は Pydantic schema 側で行う。
