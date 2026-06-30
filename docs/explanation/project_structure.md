# Project structure

このリポジトリは、練習用学習コードから `simple_cv_framework` へ段階的に移行します。

## 移行前

- `main.py`: argparseベースの手動学習入口
- `main_pl.py`: Hydra + Lightning の学習入口
- `dataset/`: データセットとtransform
- `model/`: モデル本体とfactory
- `setup/`: optimizerとscheduler
- `utils/`: checkpoint、accuracy、meter

## 移行後

- `packages/core`: CLI、設定schema、runner、snapshot、doctor
- `packages/vision`: 標準モデル、標準データセット、transform
- `template`: Copierで生成する学生repoのひな形
- `docs`: 学部生が実行・設定・拡張を理解するための文書

## 学生repoとの関係

上流repoは共通機能を提供し、学生repoは研究ごとのコードと設定を持ちます。学生repoはCopierで生成し、上流の更新は `copier update` で取り込みます。
