#!/usr/bin/env python3
"""Export a real CNN to ONNX for HydraRT's inference stage (Phase 12).

MobileNetV2 is chosen because it is a genuine production image-classification
architecture that is still small enough to run many copies concurrently on a laptop
CPU — the point is to exercise the scheduler with real model execution, not to
benchmark the model itself.

The exported graph takes a dynamic batch dimension so the runtime can vary batch size
per request, which is one of the things that makes real inference cost unpredictable.

    python3 scripts/export_model.py [output_path]
"""

import sys
import os

import torch
import torchvision


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "models/mobilenet_v2.onnx"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Random weights rather than pretrained: this avoids a large download, and the
    # weights do not affect execution cost, which is what we are measuring. The graph,
    # the operators, and the arithmetic are all identical to the pretrained model.
    model = torchvision.models.mobilenet_v2(weights=None)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )

    # torch 2.9 defaults to the dynamo-based exporter, which pulls in onnxscript. The
    # legacy TorchScript exporter produces the same graph for a model this simple and
    # needs no extra dependency, so prefer it and fall back if the flag is unsupported.
    try:
        torch.onnx.export(model, dummy, out, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, out, **kwargs)

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"wrote {out} ({size_mb:.1f} MB)")
    print("input:  float32 [batch, 3, 224, 224]")
    print("output: float32 [batch, 1000]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
