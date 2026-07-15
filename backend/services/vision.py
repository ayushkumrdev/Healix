#!/usr/bin/env python3
"""
Medical image analysis for Healix (local-first computer vision).

Multi-modality, routed automatically from the image itself:

  - Chest radiographs (X-rays) -> TorchXRayVision DenseNet (18 thoracic pathologies)
  - Skin lesions / dermatoscopic photos -> a Hugging Face image-classification
    model (skin-cancer classes)

The vision models perform PERCEPTION (what is visible). They do not diagnose.
The findings are handed to the RAG + LLM pipeline, which explains them in plain
language. Everything degrades gracefully: if a model or its weights are missing,
analysis returns an informative result instead of raising, so the app never
breaks. Runs offline once weights are cached.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_FINDING_THRESHOLD = float(os.getenv("HEALIX_VISION_THRESHOLD", "0.5"))
SKIN_MODEL = os.getenv("HEALIX_SKIN_MODEL", "Anwarkh1/Skin_Cancer-Image_Classification")

_BACKEND: Optional[Dict[str, Any]] = None
_BACKEND_ERROR: Optional[str] = None


def _load_backend() -> Optional[Dict[str, Any]]:
    global _BACKEND, _BACKEND_ERROR
    if _BACKEND is not None:
        return _BACKEND
    if _BACKEND_ERROR is not None:
        return None
    try:
        import torch
        import torchvision
        import torchxrayvision as xrv
        _BACKEND = {"torch": torch, "torchvision": torchvision, "xrv": xrv}
        return _BACKEND
    except Exception as e:  # pragma: no cover
        _BACKEND_ERROR = f"{type(e).__name__}: {e}"
        return None


_PRETTY = {
    "Effusion": "Pleural effusion",
    "Lung Opacity": "Lung opacity",
    "Lung Lesion": "Lung lesion",
    "Pleural_Thickening": "Pleural thickening",
    "Enlarged Cardiomediastinum": "Enlarged cardiomediastinum",
}


def _pretty(label: str) -> str:
    return _PRETTY.get(label, str(label).replace("_", " ").strip().title())


class MedicalImageAnalyzer:
    """Routes a medical image to the right local model and returns findings."""

    def __init__(self, xray_weights: str = "densenet121-res224-all"):
        self.xray_weights = xray_weights
        self._xray_model = None
        self._skin_pipe = None
        self._skin_error: Optional[str] = None
        self._device = "cpu"

    # ----- model lifecycle ---------------------------------------------
    def _ensure_xray(self) -> bool:
        if self._xray_model is not None:
            return True
        be = _load_backend()
        if be is None:
            return False
        try:
            import contextlib
            import io as _io
            torch = be["torch"]
            xrv = be["xrv"]
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            with contextlib.redirect_stdout(_io.StringIO()), \
                    contextlib.redirect_stderr(_io.StringIO()):
                model = xrv.models.DenseNet(weights=self.xray_weights)
            model.eval().to(self._device)
            self._xray_model = model
            return True
        except Exception as e:  # pragma: no cover
            global _BACKEND_ERROR
            _BACKEND_ERROR = f"xray model load failed: {type(e).__name__}: {e}"
            return False

    def _ensure_skin(self) -> bool:
        if self._skin_pipe is not None:
            return True
        if self._skin_error is not None:
            return False
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self._device = "cuda" if device == 0 else "cpu"
            self._skin_pipe = pipeline("image-classification", model=SKIN_MODEL, device=device)
            return True
        except Exception as e:  # pragma: no cover
            self._skin_error = f"{type(e).__name__}: {e}"
            return False

    # ----- helpers ------------------------------------------------------
    @staticmethod
    def _to_gray_array(image) -> np.ndarray:
        from PIL import Image
        if isinstance(image, Image.Image):
            return np.array(image.convert("L"), dtype=np.float32)
        arr = np.asarray(image, dtype=np.float32)
        return arr.mean(axis=2) if arr.ndim == 3 else arr

    @staticmethod
    def _to_pil(image):
        from PIL import Image
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")

    @staticmethod
    def _mean_saturation(image) -> float:
        try:
            from PIL import Image
            pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype("uint8"))
            hsv = np.asarray(pil.convert("HSV"), dtype=np.float32)
            return float(hsv[..., 1].mean())
        except Exception:
            return 0.0

    def _route(self, image, modality: str) -> str:
        if modality and modality != "auto":
            return modality
        # Grayscale -> radiograph; colorful -> skin/derm photo
        return "chest_xray" if self._mean_saturation(image) < 25.0 else "skin"

    # ----- public -------------------------------------------------------
    def analyze(self, image, modality: str = "auto",
                threshold: float = DEFAULT_FINDING_THRESHOLD) -> Dict[str, Any]:
        route = self._route(image, modality)
        if route == "skin":
            return self._analyze_skin(image, threshold)
        return self._analyze_xray(image, threshold)

    def _analyze_xray(self, image, threshold: float) -> Dict[str, Any]:
        be = _load_backend()
        if be is None or not self._ensure_xray():
            return self._unavailable("chest_xray")
        try:
            torch, xrv, tv = be["torch"], be["xrv"], be["torchvision"]
            img = self._to_gray_array(image)
            img = xrv.datasets.normalize(img, 255)[None, ...]
            img = tv.transforms.Compose([
                xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])(img)
            tensor = torch.from_numpy(img)[None, ...].to(self._device)
            with torch.no_grad():
                out = self._xray_model(tensor).cpu().numpy()[0]
            findings = [{"label": _pretty(p), "probability": round(float(v), 4)}
                        for p, v in zip(self._xray_model.pathologies, out) if p]
            findings.sort(key=lambda f: f["probability"], reverse=True)
            top = [f for f in findings if f["probability"] >= threshold]
            return {
                "available": True, "modality": "chest_xray", "modality_label": "Chest X-ray",
                "device": self._device, "findings": findings, "top_findings": top,
                "summary": self._summary(top, findings, "chest radiograph"),
                "prompt_block": self._block(top, findings, "chest radiograph"),
            }
        except Exception as e:  # pragma: no cover
            return self._unavailable("chest_xray", str(e))

    def _analyze_skin(self, image, threshold: float) -> Dict[str, Any]:
        if not self._ensure_skin():
            return self._unavailable("skin", self._skin_error)
        try:
            preds = self._skin_pipe(self._to_pil(image), top_k=8)
            findings = [{"label": _pretty(p["label"]), "probability": round(float(p["score"]), 4)}
                        for p in preds]
            findings.sort(key=lambda f: f["probability"], reverse=True)
            top = findings[:1]  # classification: surface the leading class
            return {
                "available": True, "modality": "skin", "modality_label": "Skin image",
                "device": self._device, "findings": findings, "top_findings": top,
                "summary": self._summary(top, findings, "skin image"),
                "prompt_block": self._block(top, findings, "skin image"),
            }
        except Exception as e:  # pragma: no cover
            return self._unavailable("skin", str(e))

    # ----- formatting ---------------------------------------------------
    @staticmethod
    def _summary(top: List[Dict], findings: List[Dict], kind: str) -> str:
        if not findings:
            return "No findings could be computed."
        if not top:
            best = findings[0]
            return (f"No findings reached the reporting threshold. Highest estimate: "
                    f"{best['label']} at {best['probability']:.0%}. Screening estimate, "
                    "not a diagnosis.")
        parts = ", ".join(f"{f['label']} ({f['probability']:.0%})" for f in top)
        return (f"Leading {kind} finding(s): {parts}. Model screening estimate, "
                "not a diagnosis.")

    @staticmethod
    def _block(top: List[Dict], findings: List[Dict], kind: str) -> str:
        ranked = top if top else findings[:3]
        if not ranked:
            return f"Automated {kind} analysis produced no usable findings."
        lines = "; ".join(f"{f['label']} {f['probability']:.2f}" for f in ranked)
        return (f"Automated {kind} analysis (model-estimated probabilities, for "
                f"reasoning only, not a diagnosis): {lines}. Values near or below "
                "0.5 are uncertain.")

    @staticmethod
    def _unavailable(modality: str, error: Optional[str] = None) -> Dict[str, Any]:
        hint = ("pip install torchxrayvision scikit-image"
                if modality == "chest_xray" else
                "ensure transformers can download the skin model")
        return {
            "available": False, "modality": modality, "modality_label": modality,
            "findings": [], "top_findings": [], "prompt_block": "",
            "summary": f"Automated analysis unavailable for this image ({hint}).",
            "error": error or _BACKEND_ERROR,
        }


_ANALYZER: Optional[MedicalImageAnalyzer] = None


def get_analyzer() -> MedicalImageAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = MedicalImageAnalyzer()
    return _ANALYZER


def analyze_image(image, modality: str = "auto",
                  threshold: float = DEFAULT_FINDING_THRESHOLD) -> Dict[str, Any]:
    return get_analyzer().analyze(image, modality=modality, threshold=threshold)
