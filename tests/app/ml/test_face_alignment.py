from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest

from app.ml import face_alignment as face_alignment_module
from app.core.exceptions import FaceAlignmentError
from app.ml.face_alignment import FaceAlignment


TEST_IMAGE_PATH = Path("tests/test_data/images/facealignment/test_face.png")
RESULT_IMAGE_PATH = Path("tests/test_data/images/facealignment/result_face.png")
WEIGHT_PATH = Path("tests/test_data/s3/buckets/weights/facealignment/face_landmarker.task")


class TestFaceAlignment:
    @staticmethod
    def _dummy_weight_bytes() -> BytesIO:
        return BytesIO(b"dummy-weight-bytes")

    @staticmethod
    def _load_weight_bytes(path: Path) -> BytesIO:
        return BytesIO(path.read_bytes())

    @staticmethod
    def _landmarks(
        *,
        left_eye: tuple[float, float] = (0.30, 0.40),
        right_eye: tuple[float, float] = (0.70, 0.40),
        nose: tuple[float, float] = (0.50, 0.58),
        mouth: tuple[float, float] = (0.50, 0.80),
    ) -> list[SimpleNamespace]:
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(468)]

        for index in [33, 133]:
            landmarks[index] = SimpleNamespace(x=left_eye[0], y=left_eye[1])
        for index in [362, 263]:
            landmarks[index] = SimpleNamespace(x=right_eye[0], y=right_eye[1])
        landmarks[1] = SimpleNamespace(x=nose[0], y=nose[1])
        for index in [13, 14]:
            landmarks[index] = SimpleNamespace(x=mouth[0], y=mouth[1])

        return landmarks

    def test_reverse_color_channels_reverses_rgb_and_bgr_order(self) -> None:
        alignment = FaceAlignment(weight_bytes=self._dummy_weight_bytes())
        image_np = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )

        output = alignment._reverse_color_channels(image_np)

        assert np.array_equal(output, image_np[:, :, ::-1])

    @patch.object(face_alignment_module.mp.tasks.vision.FaceLandmarker, "create_from_options")
    @patch.object(face_alignment_module.mp.tasks.vision, "FaceLandmarkerOptions")
    @patch.object(face_alignment_module.mp.tasks, "BaseOptions")
    def test_create_face_landmarker_builds_mediapipe_landmarker_with_expected_options(
        self,
        mock_base_options: MagicMock,
        mock_face_landmarker_options: MagicMock,
        mock_create_from_options: MagicMock,
    ) -> None:
        weight_bytes = BytesIO(b"dummy-weight-bytes")
        alignment = FaceAlignment(weight_bytes=weight_bytes)
        base_options = object()
        options = object()
        landmarker = object()
        mock_base_options.return_value = base_options
        mock_face_landmarker_options.return_value = options
        mock_create_from_options.return_value = landmarker

        output = alignment._create_face_landmarker()

        assert output is landmarker
        mock_base_options.assert_called_once_with(
            model_asset_buffer=b"dummy-weight-bytes",
        )
        mock_face_landmarker_options.assert_called_once_with(
            base_options=base_options,
            running_mode=face_alignment_module.mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        mock_create_from_options.assert_called_once_with(options)

    @pytest.mark.parametrize(
        ("face_landmarks", "match"),
        [
            ([], "顔が検出できませんでした"),
            ([object(), object()], "顔が複数検出されました"),
        ],
    )
    def test_detect_landmarks_bgr_rejects_invalid_face_count(
        self,
        face_landmarks: list[object],
        match: str,
    ) -> None:
        alignment = FaceAlignment(weight_bytes=self._dummy_weight_bytes())
        image_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        landmarker = MagicMock()
        landmarker.detect.return_value = SimpleNamespace(face_landmarks=face_landmarks)

        with pytest.raises(FaceAlignmentError, match=match):
            alignment._detect_landmarks_bgr(landmarker=landmarker, image_bgr=image_bgr)

    def test_detect_landmarks_bgr_returns_single_face_landmarks(self) -> None:
        alignment = FaceAlignment(weight_bytes=self._dummy_weight_bytes())
        image_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        expected_landmarks = self._landmarks()
        landmarker = MagicMock()
        landmarker.detect.return_value = SimpleNamespace(face_landmarks=[expected_landmarks])

        landmarks = alignment._detect_landmarks_bgr(
            landmarker=landmarker,
            image_bgr=image_bgr,
        )

        assert landmarks is expected_landmarks
        landmarker.detect.assert_called_once()

    def test_validate_face_pose_returns_pose_metadata_for_front_face(self) -> None:
        alignment = FaceAlignment(weight_bytes=self._dummy_weight_bytes())

        roll_deg, yaw_ratio, pitch_ratio = alignment._validate_face_pose(
            landmarks=self._landmarks(),
            width=100,
            height=100,
        )

        assert roll_deg == pytest.approx(0.0)
        assert yaw_ratio == pytest.approx(0.0)
        assert pitch_ratio == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("landmarks", "match"),
        [
            (
                _landmarks(left_eye=(0.30, 0.40), right_eye=(0.30, 0.40)),
                "目の位置を正しく取得できませんでした",
            ),
            (
                _landmarks(mouth=(0.50, 0.40)),
                "顔の上下位置を正しく取得できませんでした",
            ),
            (
                _landmarks(right_eye=(0.70, 0.70)),
                "顔の傾きが大きすぎます",
            ),
            (
                _landmarks(nose=(0.90, 0.58)),
                "横向きすぎる可能性があります",
            ),
            (
                _landmarks(nose=(0.50, 0.40)),
                "上下を向きすぎている可能性があります",
            ),
        ],
    )
    def test_validate_face_pose_rejects_invalid_pose(
        self,
        landmarks: list[SimpleNamespace],
        match: str,
    ) -> None:
        alignment = FaceAlignment(weight_bytes=self._dummy_weight_bytes())

        with pytest.raises(FaceAlignmentError, match=match):
            alignment._validate_face_pose(
                landmarks=landmarks,
                width=100,
                height=100,
            )

    def test_rotate_image_keep_size_keeps_shape_and_uses_replicate_border(self) -> None:
        alignment = FaceAlignment(weight_bytes=self._dummy_weight_bytes())
        image_bgr = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)

        output = alignment._rotate_image_keep_size(
            image_bgr=image_bgr,
            angle_deg=15.0,
        )

        assert isinstance(output, np.ndarray)
        assert output.shape == image_bgr.shape
        assert output.dtype == image_bgr.dtype

    @patch.object(FaceAlignment, "_rotate_image_keep_size")
    @patch.object(FaceAlignment, "_validate_face_pose")
    @patch.object(FaceAlignment, "_detect_landmarks_bgr")
    def test_align_face_for_registration_returns_aligned_image_and_metadata(
        self,
        mock_detect_landmarks_bgr: MagicMock,
        mock_validate_face_pose: MagicMock,
        mock_rotate_image_keep_size: MagicMock,
    ) -> None:
        alignment = FaceAlignment(weight_bytes=self._dummy_weight_bytes())
        image_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        aligned_bgr = np.full((2, 3, 3), 7, dtype=np.uint8)
        landmarks = self._landmarks()
        landmarker = object()
        mock_detect_landmarks_bgr.return_value = landmarks
        mock_validate_face_pose.return_value = (3.5, 0.12, 0.08)
        mock_rotate_image_keep_size.return_value = aligned_bgr

        output, metadata = alignment._align_face_for_registration(
            image_bgr=image_bgr,
            landmarker=landmarker,
        )

        assert output is aligned_bgr
        assert metadata == {
            "roll_deg": 3.5,
            "yaw_ratio": 0.12,
            "pitch_ratio": 0.08,
            "max_roll_deg": alignment.max_roll_deg,
            "max_yaw_ratio": alignment.max_yaw_ratio,
            "max_pitch_ratio": alignment.max_pitch_ratio,
        }
        mock_detect_landmarks_bgr.assert_called_once_with(landmarker, image_bgr)
        mock_validate_face_pose.assert_called_once_with(
            landmarks=landmarks,
            width=3,
            height=2,
        )
        mock_rotate_image_keep_size.assert_called_once_with(
            image_bgr=image_bgr,
            angle_deg=3.5,
        )

    @patch.object(FaceAlignment, "_align_face_for_registration")
    @patch.object(FaceAlignment, "_create_face_landmarker")
    def test_processing_creates_landmarker_aligns_bgr_and_returns_rgb(
        self,
        mock_create_face_landmarker: MagicMock,
        mock_align_face_for_registration: MagicMock,
    ) -> None:
        alignment = FaceAlignment(weight_bytes=self._dummy_weight_bytes())
        image_np = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        aligned_bgr = np.array(
            [
                [[30, 20, 10], [60, 50, 40]],
                [[90, 80, 70], [120, 110, 100]],
            ],
            dtype=np.uint8,
        )
        landmarker = MagicMock()
        context_manager = MagicMock()
        context_manager.__enter__.return_value = landmarker
        context_manager.__exit__.return_value = None
        mock_create_face_landmarker.return_value = context_manager
        mock_align_face_for_registration.return_value = (aligned_bgr, {"roll_deg": 0.0})

        output = alignment.processing(image_np=image_np)

        assert np.array_equal(output, aligned_bgr[:, :, ::-1])
        mock_create_face_landmarker.assert_called_once_with()
        mock_align_face_for_registration.assert_called_once()
        call_kwargs = mock_align_face_for_registration.call_args.kwargs
        assert call_kwargs["landmarker"] is landmarker
        assert np.array_equal(call_kwargs["image_bgr"], image_np[:, :, ::-1])

    def test_processing(self) -> None:
        test_image_np = np.asarray(Image.open(TEST_IMAGE_PATH).convert("RGB"))
        expected = np.asarray(Image.open(RESULT_IMAGE_PATH).convert("RGB"))

        alignment = FaceAlignment(weight_bytes=self._load_weight_bytes(WEIGHT_PATH))
        image_np = alignment.processing(image_np=test_image_np)
        diff = np.abs(image_np.astype(np.int16) - expected.astype(np.int16))

        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == expected.shape
        assert image_np.dtype == np.uint8
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
