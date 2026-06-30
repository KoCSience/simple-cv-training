# Add a model

新しいモデルは、モデル本体、registry登録、テストの3点をセットで追加します。

## 1. モデルを実装する

モデルは `ClassificationBaseModel` と同じ入出力規約に合わせます。

- 入力: imageなら `BCHW`、videoなら `BCTHW`
- 出力: `ModelOutput`
- loss計算: labelが渡された時だけ行う

## 2. registryに登録する

リファクタリング後は `register_model("name")` を使います。

```python
from simple_cv_core.registry import register_model


@register_model("my_model")
class MyModel:
    ...
```

## 3. テストを追加する

最低限確認すること:

- factoryまたはregistryから取得できる
- forwardで期待shapeのlogitsが返る
- labelsありでlossが返る
- labelsなしでlossがNoneになる
