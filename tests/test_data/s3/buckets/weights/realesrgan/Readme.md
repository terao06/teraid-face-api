# RealESRGAN weight

解像度補正で使用するRealESRGAN x2モデルの重みを配置します。

## 配置ファイル

```text
RealESRGAN_x2plus.pth
```

## 取得方法

Real-ESRGAN公式リリースから `RealESRGAN_x2plus.pth` を取得します。

```text
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

PowerShell例:

```powershell
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" -OutFile "RealESRGAN_x2plus.pth"
```

## SSM設定

[tests/test_data/ssm/face_api_setting.json](../../../../ssm/face_api_setting.json) では以下のキーで参照されます。

```json
{
  "realesrgan_weight": "realesrgan/RealESRGAN_x2plus.pth"
}
```
