import base64
from pathlib import Path


def detect_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def main() -> None:
    input_path = Path("sample.txt")
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} was not found")

    encoded = input_path.read_text(encoding="utf-8").strip()
    if not encoded:
        raise ValueError("sample.txt is empty")

    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]

    image_bytes = base64.b64decode(encoded)
    extension = detect_extension(image_bytes)
    output_path = Path(f"sample_copy{extension}")
    output_path.write_bytes(image_bytes)

    print(f"created: {output_path}")


if __name__ == "__main__":
    main()
