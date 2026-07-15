#!/usr/bin/env python3
"""Smoke test for the Healix medical image analyzer.

Generates a synthetic grayscale image (no real patient data) and runs it
through the analyzer to verify the model loads, weights download, and the
inference pipeline produces structured findings.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
from PIL import Image

from services.vision import analyze_image


def main():
    # Synthetic "radiograph-like" grayscale image (low saturation -> passes the
    # x-ray heuristic). Content is meaningless; this only exercises the pipeline.
    rng = np.random.default_rng(0)
    base = np.linspace(20, 220, 256, dtype=np.float32)
    img = np.tile(base, (256, 1))
    img = (img + rng.normal(0, 12, img.shape)).clip(0, 255).astype("uint8")
    pil = Image.fromarray(img, mode="L")

    print("Running analyzer (first run downloads model weights)...")
    result = analyze_image(pil)

    print(f"available : {result['available']}")
    print(f"modality  : {result['modality']}")
    if result.get("error"):
        print(f"error     : {result['error']}")
    if result["available"] and result["modality"] == "chest_xray":
        print(f"device    : {result.get('device')}")
        print(f"#findings : {len(result['findings'])}")
        print("top 5 by probability:")
        for f in result["findings"][:5]:
            print(f"   {f['label']:<28} {f['probability']:.3f}")
        print("\nprompt_block:\n   " + result["prompt_block"])
        print("\nSMOKE OK")
    else:
        print("summary   : " + result["summary"])
        print("\nSMOKE: analyzer reachable but not in chest_xray mode")


if __name__ == "__main__":
    main()
