# Face Landmarker weight

顔向き補正で使用するMediaPipe Face Landmarkerのtaskモデルを配置します。

## 配置ファイル

```text
face_landmarker.task
```

## 取得方法

MediaPipe公式のモデル配布先から `face_landmarker.task` を取得します。

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

PowerShell例:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "face_landmarker.task"
```

## SSM設定

[tests/test_data/ssm/face_api_setting.json](../../../../ssm/face_api_setting.json) では以下のキーで参照されます。

```json
{
  "face_alignment_weight": "facealignment/face_landmarker.task"
}
```
