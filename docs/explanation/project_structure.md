# Project structure

このリポジトリは、練習用学習コードから `simple_cv_framework` へ段階的に移行します。

## 移行前

- `main.py`: Hydra + 手動 PyTorch loop の学習入口
- `main_pl.py`: Hydra + Lightning の学習入口
- `runner/`: 手動 loop と Lightning loop の実行本体
- `dataset/`: データセットとtransform
- `model/`: モデル本体とfactory
- `setup/`: optimizerとscheduler
- `utils/`: 手動 loop 用 checkpoint、accuracy、meter

## 入口と実行本体を分ける理由

このリポジトリでは、入口と学習処理を分けます。入口は起動と設定合成を担当し、runnerは学習処理を担当します。

- `main.py`: Hydraで設定を合成し、手動 PyTorch loop のrunnerを呼ぶ入口
- `main_pl.py`: Hydraで設定を合成し、Lightning runnerを呼ぶ入口
- `runner/manual_runner.py`: data/model/optimizer/schedulerを組み立て、手動のtrain/validation/checkpoint処理を実行する本体
- `runner/lightning_runner.py`: Lightning `Trainer`、`LightningModule`、`DataModule`を組み立てて実行する本体

この分離により、設定方式はHydraに統一しつつ、実行方式は手動 PyTorch loop と Lightning loop の2つを教材として比較できます。`main.py`と`main_pl.py`を統合しないのは、両者の違いを入口名ではなく実行方式として明確に残すためです。

入口を薄くしておくと、将来 `scv train` のようなCLIに移行するときも、CLI側は設定を作ってrunnerを呼ぶだけで済みます。また、テストではHydraのCLIを経由せずrunnerを直接呼べるため、設定の参照、checkpoint形式、multi-GPU設定などの挙動を小さく確認できます。

checkpoint形式やmulti-GPUの扱いもrunner側の責務として分けます。`main.py`側は手動loop用の`.pt` checkpointと`GPU.use_dp`を扱い、`main_pl.py`側はLightningの`.ckpt` checkpointと`GPU.devices`を扱います。

## 移行後

- `packages/core`: CLI、設定schema、runner、snapshot、doctor
- `packages/vision`: 標準モデル、標準データセット、transform
- `template`: Copierで生成する学生repoのひな形
- `docs`: 学部生が実行・設定・拡張を理解するための文書

## 学生repoとの関係

上流repoは共通機能を提供し、学生repoは研究ごとのコードと設定を持ちます。学生repoはCopierで生成し、上流の更新は `copier update` で取り込みます。
