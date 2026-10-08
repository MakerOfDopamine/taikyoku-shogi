"""Fold the attention pattern into com.microsoft.Attention, with whatever onnxruntime is here.

    python3 fuse.py net_fp16_raw.onnx net_fp16.onnx

The bundled graphs are already fused, and the fused graph has been run unchanged by
onnxruntime 1.20.0, 1.22.0, 1.24.4 and 1.29.0. This is the escape hatch for a version that
refuses one day: re-fusing the raw graph locally produced bit-identical fp32 values on all
four, so it is a genuine substitute rather than a fallback of last resort.

Without the fusion the graph still runs, but a 1296-token layer keeps a literal
8 x 1296 x 1296 score matrix -- 27 MB per board per layer, and about half the speed.
"""
import sys

import onnxruntime as ort
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

if len(sys.argv) != 3:
    sys.exit(__doc__)

m = optimizer.optimize_model(sys.argv[1], model_type="bert", num_heads=4, hidden_size=128,
                             opt_level=0, optimization_options=FusionOptions("bert"))
stats = m.get_fused_operator_statistics()
m.save_model_to_file(sys.argv[2])
print(f"fused with onnxruntime {ort.__version__}: "
      f"Attention={stats.get('Attention', 0)} (3 of 4 layers is expected -- layer 0 sits "
      f"behind the positional embedding's broadcast Add), "
      f"SkipLayerNormalization={stats.get('SkipLayerNormalization', 0)}")
if stats.get("Attention", 0) == 0:
    sys.exit("no attention fused -- the graph is not the BERT-shaped export")
