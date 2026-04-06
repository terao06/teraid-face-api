import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from scrfd import SCRFD, Threshold


ML_ROOT = Path(__file__).resolve().parent

class Scrfd:
    def __init__(self):
        self.weight_path = ML_ROOT / "weights" / "scrfd" / "scrfd.onnx"

    def get_face_counts(self, image: Image.Image) -> int:
        model = SCRFD.from_path(self.weight_path)
        faces = model.detect(image=image, threshold=Threshold(probability=0.4))
        return len(faces)
