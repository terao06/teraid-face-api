# SCRFD weight

顔検出で使用するSCRFDのONNXモデルを配置します。

## 配置ファイル

```text
scrfd.onnx
```

## 取得方法

1. cospectrum/scrfd リポジトリの `models/scrfd.onnx` を取得します。
   - https://github.com/cospectrum/scrfd/tree/main/models
2. ダウンロードしたONNXファイルを `scrfd.onnx` という名前でこのディレクトリへ配置します。

PowerShellで直接取得する場合:

```powershell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/cospectrum/scrfd/main/models/scrfd.onnx" `
  -OutFile "scrfd.onnx"
```

## SSM設定

[tests/test_data/ssm/face_api_setting.json](../../../../ssm/face_api_setting.json) では以下のキーで参照されます。

```json
{
  "scrfd_weight": "scrfd/scrfd.onnx"
}
```
