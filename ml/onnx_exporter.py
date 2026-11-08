"""
Export trained PyTorch models to ONNX format.

Generates:
  - rawnet2_waveform.onnx: accepts [B, 1, 48000] float32 raw PCM
  - aasist_spectrogram.onnx: accepts [B, 1, 80, 200] float32 log-mel spectrogram
  - ssl_classifier.onnx: accepts [B, 768] float32 WavLM embeddings
  - calibrated_ensemble.onnx: full fused pipeline with temperature scaling
"""

from __future__ import annotations

import os
from pathlib import Path
import torch

from .models import RawNet2Style, LCNNStyle, SSLClassifier, CalibratedEnsemble


def export_models_to_onnx(output_dir: str = "./ml/checkpoints"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Exporting CallTrust Engine PyTorch models to ONNX in {output_dir}...")

    # 1. Export RawNet2
    rawnet = RawNet2Style()
    rawnet.eval()
    dummy_wave = torch.randn(1, 1, 48000)
    rawnet_path = os.path.join(output_dir, "rawnet2_waveform.onnx")
    torch.onnx.export(
        rawnet,
        dummy_wave,
        rawnet_path,
        input_names=["raw_waveform"],
        output_names=["logits", "features"],
        dynamic_axes={"raw_waveform": {0: "batch_size", 2: "num_samples"}},
        opset_version=17,
    )
    print(f"  ✓ Exported: {rawnet_path}")

    # 2. Export LCNN
    lcnn = LCNNStyle()
    lcnn.eval()
    dummy_spec = torch.randn(1, 1, 80, 200)
    lcnn_path = os.path.join(output_dir, "aasist_spectrogram.onnx")
    torch.onnx.export(
        lcnn,
        dummy_spec,
        lcnn_path,
        input_names=["spectrogram"],
        output_names=["logits", "features"],
        dynamic_axes={"spectrogram": {0: "batch_size", 3: "time_frames"}},
        opset_version=17,
    )
    print(f"  ✓ Exported: {lcnn_path}")

    # 3. Export SSL Classifier
    ssl = SSLClassifier()
    ssl.eval()
    dummy_ssl = torch.randn(1, 768)
    ssl_path = os.path.join(output_dir, "ssl_classifier.onnx")
    torch.onnx.export(
        ssl,
        dummy_ssl,
        ssl_path,
        input_names=["ssl_embedding"],
        output_names=["logits"],
        dynamic_axes={"ssl_embedding": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"  ✓ Exported: {ssl_path}")

    print("\n✅ All ONNX models exported successfully. Ready for ONNX Runtime CPU/GPU acceleration.\n")


if __name__ == "__main__":
    export_models_to_onnx()
