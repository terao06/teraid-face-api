from io import BytesIO
from pathlib import Path

from onnxruntime import InferenceSession
from PIL import Image
from scrfd import SCRFD, Threshold


ML_ROOT = Path(__file__).resolve().parent

class Scrfd:
    def __init__(self, weight_bytes: BytesIO):
        self.weight_bytes = weight_bytes

    def get_face_counts(self, image: Image.Image) -> int:
        self.weight_bytes.seek(0)
        session = InferenceSession(self.weight_bytes.read())
        model = SCRFD.from_session(session)
        faces = model.detect(image=image, threshold=Threshold(probability=0.4))
        return len(faces)
