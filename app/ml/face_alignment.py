from __future__ import annotations

import math
from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np

from app.core.exceptions import FaceAlignmentError


class FaceAlignment:
    def __init__(self, weight_bytes: BytesIO) -> None:
        self.weight_bytes = weight_bytes

        self.left_eye_indices = [33, 133]
        self.right_eye_indices = [362, 263]
        self.nose_tip_index = 1
        self.mouth_center_indices = [13, 14]

        self.max_roll_deg = 25.0
        self.max_yaw_ratio = 0.30
        self.max_pitch_ratio = 0.20

    def _create_face_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_buffer=self.weight_bytes.getvalue()
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
        )
        return FaceLandmarker.create_from_options(options)

    def _reverse_color_channels(self, image_np: np.ndarray) -> np.ndarray:
        return image_np[:, :, ::-1]

    def _detect_landmarks_bgr(self, landmarker, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise FaceAlignmentError("顔が検出できませんでした")

        if len(result.face_landmarks) > 1:
            raise FaceAlignmentError("顔が複数検出されました")

        return result.face_landmarks[0]

    def _to_pixel(self, landmark, width: int, height: int) -> tuple[float, float]:
        return landmark.x * width, landmark.y * height

    def _average_points(
        self,
        points: list[tuple[float, float]],
    ) -> tuple[float, float]:
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )

    def _get_average_landmark_point(
        self,
        landmarks,
        indices: list[int],
        width: int,
        height: int,
    ) -> tuple[float, float]:
        return self._average_points([
            self._to_pixel(landmarks[i], width, height)
            for i in indices
        ])

    def _calculate_roll_angle_deg(
        self,
        left_eye: tuple[float, float],
        right_eye: tuple[float, float],
    ) -> float:
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        return math.degrees(math.atan2(dy, dx))

    def _validate_face_pose(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> tuple[float, float, float]:
        left_eye = self._get_average_landmark_point(
            landmarks, self.left_eye_indices, width, height
        )
        right_eye = self._get_average_landmark_point(
            landmarks, self.right_eye_indices, width, height
        )
        nose = self._to_pixel(landmarks[self.nose_tip_index], width, height)
        mouth = self._get_average_landmark_point(
            landmarks, self.mouth_center_indices, width, height
        )

        eye_center = self._average_points([left_eye, right_eye])
        eye_distance = abs(right_eye[0] - left_eye[0])

        if eye_distance <= 0:
            raise FaceAlignmentError("目の位置を正しく取得できませんでした")

        roll_deg = self._calculate_roll_angle_deg(left_eye, right_eye)

        yaw_ratio = abs(nose[0] - eye_center[0]) / eye_distance

        eye_to_mouth_distance = abs(mouth[1] - eye_center[1])
        if eye_to_mouth_distance <= 0:
            raise FaceAlignmentError("顔の上下位置を正しく取得できませんでした")

        expected_nose_y = eye_center[1] + eye_to_mouth_distance * 0.45
        pitch_ratio = abs(nose[1] - expected_nose_y) / eye_to_mouth_distance

        if abs(roll_deg) > self.max_roll_deg:
            raise FaceAlignmentError(
                "顔の傾きが大きすぎます。もう少しまっすぐ向いて撮影してください。"
                f"roll={roll_deg:.2f}"
            )

        if yaw_ratio > self.max_yaw_ratio:
            raise FaceAlignmentError(
                "横向きすぎる可能性があります。正面を向いて撮影してください。"
                f"yaw_ratio={yaw_ratio:.3f}"
            )

        if pitch_ratio > self.max_pitch_ratio:
            raise FaceAlignmentError(
                "上下を向きすぎている可能性があります。正面を向いて撮影してください。"
                f"pitch_ratio={pitch_ratio:.3f}"
            )

        return roll_deg, yaw_ratio, pitch_ratio

    def _rotate_image_keep_size(
        self,
        image_bgr: np.ndarray,
        angle_deg: float,
    ) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        center = (width / 2, height / 2)

        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        return cv2.warpAffine(
            image_bgr,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _align_face_for_registration(
        self,
        image_bgr: np.ndarray,
        landmarker,
    ) -> tuple[np.ndarray, dict]:
        height, width = image_bgr.shape[:2]

        landmarks = self._detect_landmarks_bgr(landmarker, image_bgr)

        roll_deg, yaw_ratio, pitch_ratio = self._validate_face_pose(
            landmarks=landmarks,
            width=width,
            height=height,
        )

        aligned = self._rotate_image_keep_size(
            image_bgr=image_bgr,
            angle_deg=roll_deg,
        )

        metadata = {
            "roll_deg": roll_deg,
            "yaw_ratio": yaw_ratio,
            "pitch_ratio": pitch_ratio,
            "max_roll_deg": self.max_roll_deg,
            "max_yaw_ratio": self.max_yaw_ratio,
            "max_pitch_ratio": self.max_pitch_ratio,
        }

        return aligned, metadata

    def processing(
        self,
        image_np: np.ndarray,
    ) -> np.ndarray:
        bgr_image_np = self._reverse_color_channels(image_np=image_np)

        with self._create_face_landmarker() as landmarker:
            aligned_bgr_image_np, _ = self._align_face_for_registration(
                image_bgr=bgr_image_np,
                landmarker=landmarker,
            )

        return self._reverse_color_channels(image_np=aligned_bgr_image_np)
