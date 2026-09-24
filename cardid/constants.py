"""Geometry shared by training, the torch runtime, the ONNX export and the browser.

The browser receives these through `manifest.json` (`constants`, see `export.py`), so each
has exactly one definition here; the detector, the reference bundle runtime and the export
all import them instead of repeating the numbers.
"""

SCENE = 640  # native px around the click (capture.html CROP; the first detector window)
DET_INPUT = 256  # detector input; every window is resampled to this square
CARD_ASPECT = 88 / 63  # long / short side of a Magic card (63 x 88 mm)
ROTATIONS = 4  # exact 90-degree turns of the refined window whose up vectors are summed
REFINE_FILL = 0.6  # the card's long side as a fraction of the refine (second-pass) window
REFINE_MIN_SIDE = 64.0  # smallest refine window side, in image px
