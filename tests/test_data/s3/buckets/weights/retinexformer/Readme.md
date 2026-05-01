# Retinexformer / MST++ weight

明るさ調整で使用するMST++のNTIRE向け重みを配置します。

## 配置ファイル

```text
MST_Plus_Plus_8x1150.pth
```

## 取得方法

1. Retinexformer公式リポジトリを開きます。
   - https://github.com/caiyuanhao1998/Retinexformer
2. READMEの `Testing` セクションにある `Download our models` から事前学習済みモデルを取得します。
   - Baidu Disk: README記載のリンク、code `cyh2`
   - Google Drive: README記載のリンク
3. 取得したファイル群から `MST_Plus_Plus_8x1150.pth` を取り出し、このディレクトリへ配置します。

公式READMEでは、以下のテストコマンドでこの重みが利用されています。

```bash
python3 Enhancement/test_from_dataset.py --opt Options/MST_Plus_Plus_NTIRE_8x1150.yml --weights pretrained_weights/MST_Plus_Plus_8x1150.pth --dataset NTIRE --self_ensemble
```

## SSM設定

[tests/test_data/ssm/face_api_setting.json](../../../../ssm/face_api_setting.json) では以下のキーで参照されます。

```json
{
  "retinexformer_weight": "retinexformer/MST_Plus_Plus_8x1150.pth"
}
```
