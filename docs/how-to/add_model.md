# Add a model

新しいモデルは、モデル本体、registry登録、テストの3点をセットで追加します。

## 1. モデルを実装する

モデルは `ClassificationBaseModel` と同じ入出力規約に合わせます。

- 入力: imageなら `BCHW`、videoなら `BCTHW`
- 出力: `ModelOutput`
- loss計算: labelが渡された時だけ行う

## 2. registryに登録する

`register_model("name")` を使います。上流の標準モデルは `model/model_factory.py` で登録しています。

```python
from simple_cv_core.registry import register_model


@register_model("my_model")
class MyModel:
    ...
```

既存のfactory互換APIから使う場合、登録した名前を `model.model_name` に指定します。

## 3. テストを追加する

最低限確認すること:

- factoryまたはregistryから取得できる
- forwardで期待shapeのlogitsが返る
- labelsありでlossが返る
- labelsなしでlossがNoneになる
