# Project structure

このリポジトリは、練習用学習コードから `simple_cv_framework` へ段階的に移行します。

## 現在の構造

- `main.py`: Hydra + 手動 PyTorch loop の入口
- `main_pl.py`: Hydra + Lightning の入口
- `configs/`: Hydraで合成する実験設定
- `src/simple_cv_training/`: Python packageとしてimportされる実装本体
- `src/simple_cv_training/callback/`: Lightning `Trainer` に渡すcallback factory
- `src/simple_cv_training/config/`: Hydra設定のschema検証とPython 3.14向け互換処理
- `src/simple_cv_training/runner/`: 手動 loop と Lightning loop の実行本体
- `src/simple_cv_training/dataset/`: データセットとtransform
- `src/simple_cv_training/logger/`: 手動 loop 用とLightning用のComet logger factory
- `src/simple_cv_training/model/`: モデル本体とfactory
- `src/simple_cv_training/setup/`: optimizerとscheduler
- `src/simple_cv_training/train.py`: 手動 PyTorch loop の1 epoch training処理
- `src/simple_cv_training/val.py`: 手動 PyTorch loop のvalidation処理
- `src/simple_cv_training/utils/`: 手動 loop 用 checkpoint、accuracy、meter

`main.py` / `main_pl.py` は、repo root から `python3 main.py ...` / `python3 main_pl.py ...` と起動する互換入口として残します。一方で、runner、dataset、model、logger、configなどのimportされる実装本体は `src/simple_cv_training/` に集約します。

```text
main.py / main_pl.py
  = repo root から起動する互換入口

src/simple_cv_training/
  = import される実装本体
```

## 入口と実行本体を分ける理由

このリポジトリでは、入口と学習処理を分けます。入口は起動と設定合成を担当し、runnerは学習処理を担当します。

- `main.py`: Hydraで設定を合成し、手動 PyTorch loop のrunnerを呼ぶ入口
- `main_pl.py`: Hydraで設定を合成し、Lightning runnerを呼ぶ入口
- `src/simple_cv_training/runner/manual_runner.py`: data/model/optimizer/schedulerを組み立て、手動のtrain/validation/checkpoint処理を実行する本体
- `src/simple_cv_training/runner/lightning_runner.py`: Lightning `Trainer`、`LightningModule`、`DataModule`を組み立てて実行する本体

この分離により、設定方式はHydraに統一しつつ、実行方式は手動 PyTorch loop と Lightning loop の2つを教材として比較できます。`main.py`と`main_pl.py`を統合しないのは、両者の違いを入口名ではなく実行方式として明確に残すためです。

入口を薄くしておくと、将来 `scv train` のようなCLIに移行するときも、CLI側は設定を作ってrunnerを呼ぶだけで済みます。また、テストではHydraのCLIを経由せずrunnerを直接呼べるため、設定の参照、checkpoint形式、multi-GPU設定などの挙動を小さく確認できます。

checkpoint形式やmulti-GPUの扱いもrunner側の責務として分けます。`main.py`側は手動loop用の`.pt` checkpointと`GPU.use_dp`を扱い、`main_pl.py`側はLightningの`.ckpt` checkpointと`GPU.devices`を扱います。

## パッケージ名と配布名の使い分け

`src/` layout では、Python packaging の慣習に合わせて、import用の名前と配布・リポジトリ用の名前を分けます。

- Python package directory: `src/simple_cv_training/`
- import name: `simple_cv_training`
- project / distribution name: `simple-cv-training`
- repository name: `simple-cv-training`

Python の `import` では hyphen `-` を使えないため、package directory と import name には underscore `_` を使います。一方で、`pyproject.toml` の `[project].name` や `pip install` で使う distribution name では hyphen `-` が一般的です。リポジトリ名も既存の `simple-cv-training` に合わせます。

```python
from simple_cv_training.runner import run_manual_training
```

```toml
[project]
name = "simple-cv-training"
```

実装本体は `src/simple_cv_training/` に集約し、root直下の `main.py` と `main_pl.py` は互換入口として残します。

## 移行後

- `packages/core`: CLI、設定schema、runner、snapshot、doctor
- `packages/vision`: 標準モデル、標準データセット、transform
- `template`: Copierで生成する学生repoのひな形
- `docs`: 学部生が実行・設定・拡張を理解するための文書

## 学生repoとの関係

上流repoは共通機能を提供し、学生repoは研究ごとのコードと設定を持ちます。学生repoはCopierで生成し、上流の更新は `copier update` で取り込みます。
