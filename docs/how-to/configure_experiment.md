# Configure experiments

実験設定はコード中の分岐ではなく、`configs/` 以下のYAMLとHydra overrideで変更します。

主な入口は `configs/train.yaml` です。

```bash
uv run scv train --cfg job
```

## よく変更する項目

データセット:

```bash
uv run scv train dataset.dataset_name=ImageFolder dataset.root=/data/tiny-imagenet
```

config groupで切り替える場合:

```bash
uv run scv train data=image_folder dataset.root=/data/tiny-imagenet
```

モデル:

```bash
uv run scv train model.model_name=resnet50 model.use_pretrained=false
```

config groupで切り替える場合:

```bash
uv run scv train model=resnet50 model.use_pretrained=false
```

optimizer:

```bash
uv run scv train optimizer.optimizer_name=Adam optimizer.lr=1e-4
```

config groupで切り替える場合:

```bash
uv run scv train optimizer=adam optimizer.lr=1e-4
```

学習回数:

```bash
uv run scv train training.num_epochs=50
```

## 設定変更の考え方

- 何度も使う実験は `configs/experiment/` に保存する。
- その場だけの変更はコマンドラインoverrideで指定する。
- 型や範囲の検証は Pydantic schema 側で行う。
- `checkpoint_file.checkpoint_to_resume` に通常ファイルを指定した場合、存在しないパスは実行前にエラーになる。
