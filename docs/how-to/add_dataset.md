# Add a dataset

新しいデータセットは、catalog、DataLoader/DataModule、テストをセットで追加します。

## 1. catalogを書く

リファクタリング後は `datasets/catalog/*.yaml` にデータセット情報を書きます。

含める情報:

- name
- version
- root
- splits
- num_classes
- checksum
- license

## 2. DataLoaderまたはDataModuleを実装する

入力パスの存在確認を行い、train/valのクラス数が一致することを確認します。

## 3. テストを追加する

最低限確認すること:

- 1 batchを取得できる
- tensor shapeが想定通り
- label shapeが想定通り
- class数が正しい
- 存在しないパスで分かりやすく失敗する
