import requests
import base64
from pathlib import Path
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = "./path/to/the/image/"

SCRIPT_DIR = Path(__file__).resolve().parent


def pretty_print(title: str, response: requests.Response, extra: str = ""):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"Status: {response.status_code}")
    if extra:
        print(extra)
        
    else:
        try:
            print("Response:", response.json())
        except Exception:
            print("Response (raw):", response.text[:300])
    print('='*60)


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    pretty_print("GET /health", r)


def test_classify():
    if not IMAGE_PATH or not Path(IMAGE_PATH).exists():
        print("\n[!] IMAGE_PATH is not set or file not found — skipping /classify")
        return
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (Path(IMAGE_PATH).name, f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/classify", files=files)
    pretty_print("POST /classify", r)


def test_find_nearest():
    if not IMAGE_PATH or not Path(IMAGE_PATH).exists():
        print("\n[!] IMAGE_PATH is not set or file not found — skipping /find_nearest")
        return
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (Path(IMAGE_PATH).name, f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/find_nearest", files=files)

    if r.status_code != 200:
        pretty_print("POST /find_nearest", r)
        return

    data = r.json()
    similarity = data.get("similarity")
    img_b64 = data.get("image", "")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SCRIPT_DIR / f"nearest_result_{timestamp}.jpg"

    try:
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
            
        img_bytes = base64.b64decode(img_b64)
        output_path.write_bytes(img_bytes)
        extra = (
            f"similarity: {similarity}\n"
            f"Image saved to: {output_path}"
        )
        pretty_print("POST /find_nearest", r, extra=extra)
        
    except Exception as e:
        pretty_print("POST /find_nearest", r, extra=f"Failed to save image: {e}")


if __name__ == "__main__":
    print(f"testing service: {BASE_URL}")
    print(f"image: {IMAGE_PATH or '(not set)'}")
    print(f"results will be saved to: {SCRIPT_DIR}")

    test_health()
    test_classify()
    test_find_nearest()