# Static analysis migration

このリポジトリでは、旧 `pep8` / `flake8` / `pylint` / `mypy` の完全再現ではなく、現在運用している基本的な品質チェックを `ruff` と `ty` に集約します。

## 実行コマンド

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## ruff へ移したもの

| 旧設定 | 新しい扱い |
| --- | --- |
| pycodestyle / flake8 の基本エラー | `ruff` の `E` / `F` |
| import order | `ruff` の `I` |
| flake8-bugbear | `ruff` の `B` |
| pyupgrade | `ruff` の `UP` |
| 行長120文字 | `tool.ruff.line-length = 120` |
| formatter | `ruff format` |

## ruff へ移さないもの

次の項目は `ruff` で完全には再現しません。必要になった場合は、将来 `pre-commit` の独自チェックや小さな専用スクリプトとして別管理します。

| 旧項目 | 内容 | 今回の扱い |
| --- | --- | --- |
| `VNE001` | 1文字変数名を禁止する `flake8-variables-names` のルール | 移行対象外 |
| `flake8-variables-names` | 変数名の長さや命名方針を検査する plugin | 移行対象外 |
| `flake8-functions-names` | 関数名の意味と戻り値・副作用の関係を検査する plugin | 移行対象外 |
| `W503` | 二項演算子の前で改行している場合の pycodestyle 警告 | `ruff format` に任せる |
| `pylint arguments-differ` | override したメソッドの引数差分を検査する警告 | 移行対象外 |
| `.pep8` の `aggressive = 2` | `autopep8` の強い自動修正 | 移行対象外 |

## ty へ移したもの

`ty` は `mypy` / `basedpyright` の完全再現ではなく、現行の型チェック範囲に限定して段階導入します。

初期対象は `config`、`packages/core/src/simple_cv_core`、および新しいテスト群です。既存の練習用コード全体は後続フェーズで段階的に広げます。

## mypy から変わるもの

| mypy設定 | tyでの扱い |
| --- | --- |
| `check_untyped_defs = True` | `ty check` の通常実行で補います |
| `warn_return_any = True` | 完全な1対1対応はありません |
| `warn_unused_configs = True` | `pyproject.toml` に設定を集約してCIで検出します |
| `ignore_missing_imports = True` | 広域には再現しません |

`warn_return_any` の代替として、戻り値型の明示、境界APIへの型注釈追加、`Unknown` が外へ漏れる箇所の修正を優先します。必要になった場合だけ、移行初期の比較用・補助チェックとして `mypy` を一時的に併用します。

## unresolved import の扱い

`ignore_missing_imports = True` は `ty` で広域再現しません。

1. まず `uv sync --locked` で依存関係を解決します。
2. 型スタブが必要な場合は dependency または dev dependency に追加します。
3. それでも解決できない optional dependency や型スタブ未提供の外部 module だけ、`allowed-unresolved-imports` に明示的に列挙します。

`allowed-unresolved-imports` に含めた module 由来の型は `Unknown` になり得ます。これは型の穴が完全になくなるという意味ではなく、型情報が欠落し得る範囲を設定上で追跡できるようにするための最後の手段です。
