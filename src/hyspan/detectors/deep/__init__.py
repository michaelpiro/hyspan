# Deep / ML-based detectors.
#
# To add a new deep detector:
#   1. Create a file here (e.g. tsttd.py).
#   2. Subclass BaseDetector from hyspan.detectors.base.
#   3. Implement detect(image, target) -> (H, W) tensor.
#   4. Override fit(image, gt) if training or weight-loading is needed.
#   5. Export it from this __init__.py.
#
# Example stub:
#
#   from .tsttd import TSTTDDetector
#
