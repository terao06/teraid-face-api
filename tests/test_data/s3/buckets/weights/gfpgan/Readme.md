# GFPGAN weights

顔補正で使用するGFPGAN本体、顔検出、顔パース用の重みを配置します。

## 配置ファイル

```text
GFPGANv.pth
detection_Resnet50_Final.pth
parsing_parsenet.pth
```

## 取得方法

公式リリースから以下のファイルを取得します。

```text
https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/detection_Resnet50_Final.pth
https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/parsing_parsenet.pth
```

取得後、`GFPGANv1.4.pth` はこのリポジトリの設定に合わせて `GFPGANv.pth` にリネームしてください。

PowerShell例:

```powershell
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" -OutFile "GFPGANv.pth"
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/detection_Resnet50_Final.pth" -OutFile "detection_Resnet50_Final.pth"
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/parsing_parsenet.pth" -OutFile "parsing_parsenet.pth"
```

## SSM設定

[tests/test_data/ssm/face_api_setting.json](../../../../ssm/face_api_setting.json) では以下のキーで参照されます。

```json
{
  "gfpgan_nv_weight": "gfpgan/GFPGANv.pth",
  "gfpgan_resnet_weight": "gfpgan/detection_Resnet50_Final.pth",
  "gfpgan_parsenet_weight": "gfpgan/parsing_parsenet.pth"
}
```
