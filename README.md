# A simple CNN/ViT training code

CNN/ViT を使って学習する単純な練習用コードです．

## 準備

uvを使うことを推奨しています。

インストール方法 (installation)

- 仮想環境の作成
  - `uv venv`
- パッケージ依存関係と仮想環境を同期する (パッケージインストール)
  - `uv sync`
- パッケージ依存関係を記述したlockfileを生成する
  - `uv lock`
- (仮想環境上にて) スクリプトを実行する。\
  → 毎回venv環境に入らなくて良い
  - `uv run <command>`

### 実行例

```shell
uv run python3 main.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ trainer=smoke disable_comet=true
uv run python3 main_pl.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ trainer=smoke GPU.devices=1 disable_comet=true
```

高速化は初期状態では有効化しません。baseline と比較したい場合だけ、明示的に指定します。

```shell
uv run python3 main.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ trainer=smoke disable_comet=true optimization.compile.enabled=true
uv run python3 main_pl.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ trainer=smoke GPU.devices=1 disable_comet=true optimization.amp.enabled=true
uv run python3 main_pl.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ trainer=smoke GPU.devices=1 disable_comet=true optimization.compile.enabled=true
```

`main.py` は手動 PyTorch loop の構造を学ぶ入口で、`torch.compile` のみ opt-in で使えます。AMP は `main_pl.py` の Lightning 経路で使います。詳しくは [Use optimization options](docs/how-to/use_optimization.md) を参照してください。

### 旧

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.pytorch.txt
pip install -r requirements.txt
```

## 使い方

```bash
uv run python3 main.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ training.num_workers=24 training.batch_size=8 training.num_epochs=5 GPU.use_dp=true
uv run python3 main.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ training.num_workers=24 training.batch_size=8 training.num_epochs=5 GPU.use_dp=true disable_comet=true
uv run python3 main_pl.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ training.num_workers=24 training.batch_size=8 training.num_epochs=5 GPU.devices=3
uv run python3 main_pl.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ training.num_workers=24 training.batch_size=8 training.num_epochs=5 GPU.devices=3 disable_comet=true
```

`main.py` は Hydra + 手動 PyTorch loop の教材用入口です。`main_pl.py` は Hydra + PyTorch Lightning の入口です。

checkpoint 形式は入口ごとに異なります。

- `main.py`: `utils.save_to_checkpoint()` が保存する `.pt` 形式
- `main_pl.py`: Lightning `Trainer` が保存する `.ckpt` 形式

### multi-GPU 学習

GPU の指定には`CUDA_VISIBLE_DEVICES`を使用すること．

- dp (data parallel) は`main.py`で利用可能
- ddp (distributed data parallel)は lightning の`main_pl.py`で利用可能
  - 注意：複数 GPU を用いる dp や ddp が動作しなくなるため，コード内で GPU 番号を指定するような`torch.device("cuda:0")`は**使わない**．dp や ddp のために，コード内では`torch.device("cuda")`としておく．

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python3 main.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ training.num_workers=24 training.batch_size=8 training.num_epochs=5 GPU.use_dp=true
CUDA_VISIBLE_DEVICES=0,1 uv run python3 main_pl.py data=image_folder dataset.root=/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ training.num_workers=24 training.batch_size=8 training.num_epochs=5 GPU.devices=2
```

デバッグ用には [launch.json](.vscode/launch.json) を以下のように設定する．

```json
            "env": {
                "CUDA_VISIBLE_DEVICES": "0,1",
            },
```

task 用には[tasks.json](.vscode/tasks.json)に次のように設定する．

```json
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": "0,1",
                },
            },
```

### config override

詳しくは `configs/` と [docs/how-to/configure_experiment.md](docs/how-to/configure_experiment.md) を参照．主な override は以下の通り．

- `dataset.root`：データセットの root フォルダ
- `training.batch_size`：バッチサイズ
- `training.num_workers`：データローダーのワーカー数
- `training.num_epochs`：エポック数
- `data` または `dataset.dataset_name`：データセット
  - `CIFAR10`：[torchvision の CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
  - `ImageFolder`：`dataset.root`で指定したフォルダ以下に`train/`と`val/`のディレクトリがあり，それ以下はカテゴリ名のサブディレクトリに分かれて保存されている画像データセット（[torchvision の ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)）
- `GPU.use_dp=true`：`main.py` の手動 loop で dp (Data Parallel)を使用する
- `GPU.devices`: `main_pl.py` の Lightning で使用する GPU 数または GPU 番号（`-1` は全 GPU）
- `optimization.amp.enabled=true`: `main_pl.py` の Lightning で AMP を有効化する。既定の precision は `bf16-mixed`
- `optimization.compile.enabled=true`: `main.py` の手動 model または `main_pl.py` の LightningModule に `torch.compile` を適用する
- `disable_comet=true`: cometを無効化して実行する

#### help

```bash
uv run python3 main.py --help
uv run python3 main_pl.py --help
```

Hydra で合成された設定は `uv run python3 main.py --cfg job` または `uv run python3 main_pl.py --cfg job` で確認できます。

## Comet の設定

comet の設定は，

- このディレクトリの`./.comet.config`と，
- ホームの`~/.comet.config`

の 2 つのファイルを利用する．詳しくは[comet のドキュメント](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/advanced/configuration/)を参照．コード中には API キーなどは書かないこと（[logger.py](./src/simple_cv_training/logger/logger.py)参照）．

### ホームでの全体設定

- `~/.comet.config`：すべてに共通する設定を書く．
  - comet の API キー，デフォルトの comet workspace を設定．
  - `hide_api_key`は True にすること（しないとログに API キーが残ってしまう）

```ini
[comet]
api_key=XXXXXHereIsYourAPIKeyXXXXXXXX
workspace=tttamaki

[comet_logging]
hide_api_key=True
```

### フォルダごとの設定

- このディレクトリの`./.comet.config`：このディレクトリで使用する設定を書く．
  - comet project name を設定．
  - （ここで設定する内容はホームの`~/.comet.config`よりも優先されて，上書きされる）

```ini
[comet]
project_name=simple_cnn_20230309

[comet_logging]
display_summary_level=0
file=comet_logs/comet_{project}_{datetime}.log

[comet_auto_log]
env_details=True
env_gpu=True
env_host=True
env_cpu=True
cli_arguments=True
```

### コード内での設定

- コード
  - `Experiment`オブジェクトに comet experiment name を設定．必要なら tag を設定する．
  - コード中には **API キーなどは書かない**．
  - （コード中で設定する内容は，ディレクトリごとの`./.comet.config`よりも優先される）

```python
    experiment = Experiment()  # ここでは何も設定しない

    exp_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')  # これは日時をexperiment nameに設定する例．
    experiment.set_name(exp_name)
    experiment.add_tag(args.model)  # これはモデル名をタグに設定する例．
```

---

## 管理ファイル

- `.python-version`: Pythonのバージョンの指定
- `pyproject.toml`: プロジェクトのメタデータ、パッケージ依存関係を記述 \
  [PEP 621 – Storing project metadata in pyproject.toml | peps.python.org](https://peps.python.org/pep-0621/)
- `uv.lock`: パッケージのバージョンの指定
- `configs/train.yaml`: パラメーターツールHydraを使っています。引数設定はここを触る。

## テスト環境

PyTestを利用してください。

```shell
uv run pytest test/dataset
```
