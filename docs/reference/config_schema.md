# Config schema

設定はHydraで合成し、`src/simple_cv_training/config/schema.py` のPydantic schemaで型と値を検証します。

## 主なグループ

- `dataset`: データセット名、root、train/valディレクトリ
- `model`: モデル名、事前学習重み、重み保存先
- `video`: 動画clip設定
- `training`: batch size、worker数、epoch数、ログ間隔
- `optimizer`: optimizer名、learning rate、weight decay、scheduler
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

## 検証方針

- batch size、epoch、worker数は0以上または1以上の範囲を検証する。
- optimizer名とmodel名は許可リストで検証する。
- checkpoint resume元は存在するファイルか確認する。
- 秘密情報は設定schemaに含めない。
