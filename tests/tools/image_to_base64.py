import argparse
import base64
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
IMAGES_DIR = TOOLS_DIR / "images"
OUTPUT_PATH = TOOLS_DIR / "output.txt"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def get_image_files() -> list[Path]:
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"images directory not found: {IMAGES_DIR}")

    image_files = [
        path for path in sorted(IMAGES_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not image_files:
        raise FileNotFoundError(f"no image files found in: {IMAGES_DIR}")
    return image_files


def select_image(image_files: list[Path], file_name: str | None) -> Path:
    if file_name:
        target = IMAGES_DIR / file_name
        if not target.is_file():
            raise FileNotFoundError(f"image file not found: {target}")
        if target.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported file type: {target.suffix}")
        return target

    print("Select an image file:")
    for index, image_path in enumerate(image_files, start=1):
        print(f"{index}. {image_path.name}")

    selection = input("Enter number: ").strip()
    if not selection.isdigit():
        raise ValueError("selection must be a number")

    selected_index = int(selection)
    if not 1 <= selected_index <= len(image_files):
        raise ValueError("selection is out of range")

    return image_files[selected_index - 1]


def encode_file_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def write_output(encoded_text: str) -> None:
    OUTPUT_PATH.write_text(encoded_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image in tests/tools/images to Base64 and write it to output.txt."
    )
    parser.add_argument(
        "file_name",
        nargs="?",
        help="Image file name in tests/tools/images. If omitted, you can select interactively.",
    )
    parser.add_argument(
        "--file-name",
        dest="file_name_option",
        help="Image file name in tests/tools/images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_files = get_image_files()
    file_name = args.file_name_option or args.file_name
    image_path = select_image(image_files=image_files, file_name=file_name)
    encoded_text = encode_file_to_base64(image_path)
    write_output(encoded_text)

    print(f"Selected file: {image_path.name}")
    print(f"Output file: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    # Usage:
    # python .\image_to_base64.py
    # python .\image_to_base64.py one_face.png
    # python .\image_to_base64.py --file-name not_has_faces.jpeg
    raise SystemExit(main())
