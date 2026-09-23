"""Export the recogniser, the detector and the gallery as a versioned runtime bundle.

    uv run python -m cardid.export --checkpoint data/runs/full-3/best.pt --detector data/runs/det4/last.pt

writes data/bundles/<version>/ (version defaults to <today>-<checkpoint run name>):

    manifest.json   version, source checkpoints, gallery size, constants, per-file sha256
    detector.onnx   window (256, 256, 4) uint8 RGBA -> quad, up, centre, short  (see graphs.DetectorGraph)
    embed.onnx      scene (H, W, 4) uint8 RGBA + quad (4, 2) -> embeddings (F, 128)
    search.onnx     embeddings (F, 128) -> indices (k,), scores (k,); gallery baked in
    arts.json       gallery entries in index order: id, name, set, collector_number, layout, frame
    SHA256SUMS      the same sums for `sha256sum -c` (cardid.publish checks them on the host)

The graphs are loadable by onnxruntime-web (wasm and webgpu; embed.onnx needs GridSample,
which the wasm backend always has) and by onnxruntime on a server. `cardid.bundle` runs a
bundle from Python and is what `--verify` compares against the torch pipeline: same clicks on
freshly rendered scenes through `Detector.locate_up` + `warp_card`/`art_crops` + `ArtIndex`
and through the bundle, reporting corner agreement and top-1 agreement.

New cards do not need retraining: `cardid.scryfall --update` pulls them into the gallery and
a re-export embeds them. `cardid.publish` ships a bundle to the server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from . import DATA_DIR
from .bundle import REFINE_FILL, REFINE_MIN_SIDE
from .degrade import INPUT_SIZE
from .detect import CARD_H, CARD_W, FRAME_NAMES, FRAME_PENALTY, art_crops, frame_penalties, warp_card
from .detector import CARD_ASPECT, CornerNet, Detector, load_checkpoint
from .graphs import ROTATIONS, DetectorGraph, EmbedGraph, SearchGraph
from .index import ArtIndex
from .synth import DET_INPUT, SCENE

BUNDLE_DIR = DATA_DIR / "bundles"
SUMS = "SHA256SUMS"
OPSET = 17
ART_FIELDS = ("id", "name", "set", "collector_number", "layout", "face", "lang")
CLEAR_MARGIN = 0.02  # torch top-1 lead over the runner-up above which the bundle must agree


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_sums(out: Path, manifest: dict) -> None:
    """`sha256sum -c`-style checksum file over the manifest's files plus the manifest itself."""
    lines = [f"{f['sha256']}  {name}" for name, f in manifest["files"].items()]
    lines.append(f"{sha256(out / 'manifest.json')}  manifest.json")
    (out / SUMS).write_text("\n".join(lines) + "\n")


def export_graph(
    module: torch.nn.Module, example: tuple, path: Path, inputs: list[str], outputs: list[str], dynamic: dict | None = None, fold: bool = True
) -> None:
    # the exporter puts the module back into whatever mode it found it in, so a wrapper left in
    # its default train mode would drag the shared network (dropout, batch-norm) along with it
    module.eval()
    torch.onnx.export(
        module,
        example,
        str(path),
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=dynamic,
        opset_version=OPSET,
        dynamo=False,
        do_constant_folding=fold,
    )
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def export_bundle(checkpoint: Path, detector: Path, out: Path, frame_penalty: float, topk: int, gallery_dtype: str) -> tuple[ArtIndex, Detector]:
    out.mkdir(parents=True, exist_ok=True)
    index = ArtIndex(checkpoint, frame_penalty)
    net = CornerNet(pretrained=False)
    load_checkpoint(net, detector, torch.device("cpu"))
    det = Detector(model=net)

    with torch.no_grad():
        rng = np.random.default_rng(0)
        window = torch.from_numpy(rng.integers(0, 256, (DET_INPUT, DET_INPUT, 4), dtype=np.uint8))
        export_graph(DetectorGraph(net), (window,), out / "detector.onnx", ["window"], ["quad", "up", "centre", "short"])

        scene = torch.from_numpy(rng.integers(0, 256, (SCENE, SCENE, 4), dtype=np.uint8))
        quad = torch.tensor([[200.0, 180.0], [330.0, 190.0], [320.0, 370.0], [190.0, 360.0]])
        export_graph(
            EmbedGraph(index.model),
            (scene, quad),
            out / "embed.onnx",
            ["scene", "quad"],
            ["embeddings"],
            dynamic={"scene": {0: "height", 1: "width"}},
        )

        dtype = torch.float16 if gallery_dtype == "f16" else torch.float32
        search = SearchGraph(index.embeddings, index.frames, frame_penalties(index.frames, frame_penalty), topk, dtype)
        # no constant folding here: it would materialise the gallery's float32 cast and undo the
        # half-precision storage; onnxruntime folds it once at session load instead
        example = (torch.zeros(len(FRAME_NAMES), index.embeddings.shape[1]),)
        export_graph(search, example, out / "search.onnx", ["embeddings"], ["indices", "scores"], fold=False)

    arts = [{**{k: a[k] for k in ART_FIELDS if k in a}, "frame": FRAME_NAMES[f]} for a, f in zip(index.arts, index.frames, strict=True)]
    (out / "arts.json").write_text(json.dumps(arts, separators=(",", ":")))
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out.iterdir()) if p.name not in ("manifest.json", SUMS)}
    manifest = {
        "version": out.name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "recogniser": {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint)},
        "detector": {"checkpoint": str(detector), "sha256": sha256(detector)},
        "gallery": {"arts": len(arts), "dtype": gallery_dtype, "embed_dim": int(index.embeddings.shape[1]), "frame_penalty": frame_penalty, "topk": topk},
        "constants": {
            "scene": SCENE,
            "det_input": DET_INPUT,
            "rotations": ROTATIONS,
            "refine_fill": REFINE_FILL,
            "refine_min_side": REFINE_MIN_SIDE,
            "card_aspect": CARD_ASPECT,  # the refine window is short * card_aspect / refine_fill
            "card_size": [CARD_W, CARD_H],
            "input_size": INPUT_SIZE,
            "frame_names": FRAME_NAMES,
        },
        "opset": OPSET,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_sums(out, manifest)
    total = sum(f["bytes"] for f in files.values())
    print(f"bundle {out} ({total / 1e6:.1f} MB, {len(arts)} arts)")
    return index, det


def torch_identify(index: ArtIndex, det: Detector, img: np.ndarray, click: tuple[float, float], k: int) -> tuple[np.ndarray, list[dict]]:
    """The Python pipeline the bundle must reproduce (`capture.Session.identify`, detector path)."""
    quad, _ = det.locate_up(img, click)
    vecs = index.embed(art_crops(warp_card(img, quad)))
    return quad, index.search(vecs, k)


def verify(bundle_path: Path, index: ArtIndex, det: Detector, n: int, seed: int, k: int) -> bool:
    """Compare the bundle with the torch pipeline on `n` rendered scenes, clicking a random
    point on the card. The two differ only in sub-pixel resampling (OpenCV's fixed-point
    bilinear against GridSample), which can flip a near-tie between two gallery arts or, rarely,
    snap a corner to a different heatmap peak (a one-level pixel difference in the refine
    window is enough). Passes when the median corner error is under a pixel and the top-1
    agrees on at least 97% of the scenes where the torch pipeline's top-1 leads its runner-up
    by more than `CLEAR_MARGIN`; every disagreement is listed."""
    from .bundle import Bundle
    from .synth import ArtBank, CardBank, render_scene

    bundle = Bundle(bundle_path)
    cards, arts = CardBank(), ArtBank()
    rng = np.random.default_rng(seed)
    corner_err, score_diff, disagreements = [], [], []
    agree_all, clear, agree_clear = 0, 0, 0
    t0 = time.time()
    for i in range(n):
        scene, quad = render_scene(rng, cards, arts, out=SCENE)
        centre = quad.mean(axis=0) + rng.uniform(-0.2, 0.2, size=2) * np.linalg.norm(quad[1] - quad[0])
        click = (float(centre[0]), float(centre[1]))
        ref_quad, ref_top = torch_identify(index, det, scene, click, k)
        got = bundle.identify(scene, click)
        corner_err.append(float(np.abs(got["quad"] - ref_quad).max()))
        agree = got["results"][0]["id"] == ref_top[0]["id"]
        is_clear = ref_top[0]["similarity"] - ref_top[1]["similarity"] > CLEAR_MARGIN
        agree_all += agree
        clear += is_clear
        agree_clear += agree and is_clear
        score_diff.append(abs(got["results"][0]["score"] - ref_top[0]["similarity"]))
        if not agree:
            kind = "clear" if is_clear else "near-tie"
            disagreements.append(
                f"  scene {i} ({kind}): torch {ref_top[0]['name']} {ref_top[0]['similarity']:.3f} vs bundle {got['results'][0]['name']} {got['results'][0]['score']:.3f}, corners off by {corner_err[-1]:.2f} px"
            )
    corner_err = np.array(corner_err)
    per = {name: ms / n for name, ms in bundle.timings.items()}
    print(
        f"verify on {n} rendered scenes ({time.time() - t0:.0f}s): top-1 agrees {agree_all}/{n} overall, {agree_clear}/{clear} where torch leads by > {CLEAR_MARGIN}; "
        f"corners max {corner_err.max():.2f} px (median {np.median(corner_err):.2f}), |score diff| mean {np.mean(score_diff):.4f} max {np.max(score_diff):.4f}"
    )
    print(f"onnxruntime CPU ms per click: detector {per.get('detector', 0):.0f} (2 passes), embed {per.get('embed', 0):.0f}, search {per.get('search', 0):.0f}")
    for line in disagreements:
        print(line)
    ok = agree_clear >= 0.97 * clear and float(np.median(corner_err)) < 1.0
    print("verify: OK" if ok else "verify: FAILED (bundle diverges from the torch pipeline)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="recogniser checkpoint (Embedder state dict)")
    parser.add_argument("--detector", required=True, help="detector checkpoint (CornerNet state dict)")
    parser.add_argument("--version", help="bundle name under data/bundles (default: <today>-<checkpoint run name>)")
    parser.add_argument("--out", help="bundle directory (overrides --version)")
    parser.add_argument("--frame-penalty", type=float, default=FRAME_PENALTY)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gallery-dtype", choices=["f16", "f32"], default="f16", help="storage of the gallery embeddings inside search.onnx")
    parser.add_argument("--verify", type=int, default=64, help="rendered scenes to compare against the torch pipeline (0 to skip)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    checkpoint, detector = Path(args.checkpoint), Path(args.detector)
    version = args.version or f"{datetime.now(UTC):%Y-%m-%d}-{checkpoint.parent.name}"
    out = Path(args.out) if args.out else BUNDLE_DIR / version
    index, det = export_bundle(checkpoint, detector, out, args.frame_penalty, args.topk, args.gallery_dtype)
    if args.verify and not verify(out, index, det, args.verify, args.seed, args.topk):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
