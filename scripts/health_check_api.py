#!/usr/bin/env python
"""
Quick API health check for the FastAPI server.

Checks:
 - GET /
 - POST /predict with a generated 224x224 RGB image

Usage:
  python scripts/health_check_api.py --url http://127.0.0.1:8000/predict
"""
import argparse
import io
import sys
from urllib.parse import urlparse

import requests
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()

    predict_url = args.url.rstrip("/")
    base = urlparse(predict_url)
    root_url = f"{base.scheme}://{base.hostname}:{base.port or 80}/"

    # GET /
    try:
        r = requests.get(root_url, timeout=5)
        print("GET /:", r.status_code, r.text[:200])
    except Exception as e:
        print("GET / failed:", e)
        # do not exit; continue to POST

    # Build an in-memory image (solid color)
    img = Image.new("RGB", (224, 224), color=(120, 180, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    try:
        r = requests.post(predict_url, files=files, timeout=10)
        print("POST /predict:", r.status_code, r.text[:200])
        r.raise_for_status()
        data = r.json()
        # minimal contract validation
        if not isinstance(data, dict):
            print("Unexpected response type:", type(data))
            sys.exit(1)
        if "species" in data and "confidence" in data:
            print("OK ->", data)
            return
        if isinstance(data.get("prediction"), dict):
            print("Legacy shape ->", data)
            return
        print("Unknown response shape ->", data)
        sys.exit(1)
    except Exception as e:
        print("POST /predict failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

