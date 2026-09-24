"""Render one synthetic table scene: background, cards, sleeves, occluders, webcam look.

What the renderer varies, because each is a failure mode of the classical edge finder:
busy playmats (random art crops as background), sleeves (a ring outside the card plus a
glossy tint and glare over it), rigid top-loaders (a larger clear rectangle with a specular
edge), borderless cards (the card image cropped inside its border),
neighbouring and overlapping cards, dice and fingers, any rotation, mild perspective, and
webcam photometrics.
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import DET_INPUT, SCENE
from .image_bank import ArtBank, CardBank
from .scene_geometry import PRINTED, expand, quad_from_pose, quad_roi, quad_short

CORNER_RADIUS = 0.045  # card corner radius as a fraction of the short side (3 mm on 63 mm)


def rounded_mask(w: int, h: int, radius: float) -> np.ndarray:
    """float32 HxW alpha of a rounded rectangle filling the image."""
    m = np.zeros((h, w), np.float32)
    r = round(radius)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 1.0, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 1.0, -1)
    for cx, cy in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
        cv2.circle(m, (cx, cy), r, 1.0, -1)
    return m


def warp_alpha(alpha: np.ndarray, quad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Canvas-sized float alpha of `alpha` (an image-shaped mask) warped onto `quad`."""
    out = np.zeros(shape[:2], np.float32)
    roi = quad_roi(quad, shape)
    if roi is None:
        return out
    x0, y0, x1, y1 = roi
    h, w = alpha.shape[:2]
    H = cv2.getPerspectiveTransform(PRINTED * np.float32([w, h]), (quad - np.float32([x0, y0])).astype(np.float32))
    out[y0:y1, x0:x1] = cv2.warpPerspective(alpha, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    return out


def paste(canvas: np.ndarray, img: np.ndarray, alpha: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp `img` (with float alpha) so its corners land on `quad` and blend onto the canvas.
    Returns the warped alpha (canvas-sized) so callers can draw shadows and sleeve effects.
    The blend is `dst += (src - dst) * a` in place over the quad's bounding box: three passes
    over the box instead of the four temporaries of `src * a + dst * (1 - a)`."""
    full_alpha = warp_alpha(alpha, quad, canvas.shape)
    roi = quad_roi(quad, canvas.shape)
    if roi is None:
        return full_alpha
    x0, y0, x1, y1 = roi
    h, w = img.shape[:2]
    H = cv2.getPerspectiveTransform(PRINTED * np.float32([w, h]), (quad - np.float32([x0, y0])).astype(np.float32))
    # converting the (small) source to float before the warp is cheaper than converting the warp
    src = cv2.warpPerspective(np.asarray(img, dtype=np.float32), H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    a = full_alpha[y0:y1, x0:x1, None]
    dst = canvas[y0:y1, x0:x1]
    src -= dst
    src *= a
    dst += src
    return full_alpha


def background(rng: np.random.Generator, arts: ArtBank, size: int) -> np.ndarray:
    """Float32 `size` x `size` x 3 table surface. Full-canvas passes go through cv2 (SIMD, no
    broadcasting temporaries); numpy's broadcast fills and `[..., None]` multiplies cost more
    here than the warp itself."""
    kind = rng.choice(["art", "flat", "gradient", "tiled"], p=[0.45, 0.35, 0.1, 0.1])
    if kind == "art" and arts:
        img = np.asarray(arts[int(rng.integers(len(arts)))], dtype=np.float32)  # half res; blown up anyway
        # cover the canvas at 1-2.5x so the mat's artwork is at playmat scale, any rotation
        scale = rng.uniform(1.0, 2.5) * size / min(img.shape[:2])
        # a slightly out-of-focus mat: blur the small source by sigma/scale, which is what a
        # blur of sigma on the upscaled canvas looks like, at a fraction of the pixels
        sigma = rng.uniform(0, 2.0) / scale
        if sigma > 0.12:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rng.uniform(0, 360), scale)
        M[:, 2] += np.float32([size / 2, size / 2]) - np.float32([img.shape[1] / 2, img.shape[0] / 2])
        M[:, 2] += rng.uniform(-0.3, 0.3, size=2) * size
        bg = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif kind == "tiled" and arts:
        img = arts[int(rng.integers(len(arts)))]
        tile = int(rng.integers(40, 160))
        t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        reps = size // tile + 2
        bg = np.tile(t, (reps, reps, 1))[:size, :size].astype(np.float32)
    elif kind == "gradient":
        c0, c1 = rng.uniform(0, 255, size=3).astype(np.float32), rng.uniform(0, 255, size=3).astype(np.float32)
        t = np.linspace(0, 1, size, dtype=np.float32)[:, None]
        strip = c0 * (1 - t) + c1 * t  # (size, 3): one colour per row
        horizontal = rng.random() < 0.5
        bg = cv2.resize(strip[None] if horizontal else strip[:, None], (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        # desks and plain mats: white, grey, black, wood, felt green/blue/red
        palette = np.float32([[240, 240, 238], [200, 200, 195], [30, 30, 32], [120, 80, 45], [160, 120, 80], [40, 90, 50], [40, 50, 100], [110, 30, 30]])
        col = palette[int(rng.integers(len(palette)))] * rng.uniform(0.8, 1.1)
        # monochrome grain (paper, felt, wood texture) at half resolution: the scene is
        # downscaled 2.5x for the detector anyway, and colour noise is the sensor's job.
        # Uniform noise with the same std as the old Gaussian; drawing it is 6x cheaper.
        grain = (rng.random(size=(size // 2, size // 2), dtype=np.float32) - 0.5) * np.float32(rng.uniform(1, 8) * np.sqrt(12))
        grain = cv2.resize(grain, (size, size), interpolation=cv2.INTER_LINEAR)
        bg = cv2.merge([grain, grain, grain])
        cv2.add(bg, (*(float(c) for c in col), 0.0), dst=bg)
    # uneven lighting across the table: a linear ramp, so it separates into a row and a column
    # vector (no full-size meshgrid)
    ramp = (np.arange(size, dtype=np.float32) / size - 0.5) * 2
    gx, gy = rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
    light = 1 + gx * ramp[None, :] + gy * ramp[:, None]
    cv2.multiply(bg, cv2.merge([light, light, light]), dst=bg)
    return np.clip(bg, 0, 255, out=bg)


def card_face(rng: np.random.Generator, cards: CardBank, index: int, short: float, detail: float = DET_INPUT / SCENE) -> tuple[np.ndarray, np.ndarray]:
    """A card image and its alpha. Sometimes cut inside the black border so the scene has
    borderless/extended-art cards even when the sample does not. `short` is the drawn size in
    canvas pixels and `detail` the canvas-to-output scale."""
    img = cards.load(index, short * detail)
    h, w = img.shape[:2]
    if rng.random() < 0.2:
        ix, iy = int(w * rng.uniform(0.035, 0.07)), int(h * rng.uniform(0.035, 0.07))
        img = img[iy : h - iy, ix : w - ix]
        h, w = img.shape[:2]
    return img, rounded_mask(w, h, CORNER_RADIUS * w)


def draw_card(
    canvas: np.ndarray,
    rng: np.random.Generator,
    cards: CardBank,
    quad: np.ndarray,
    shadow: bool = True,
    detail: float = DET_INPUT / SCENE,
    index: int | None = None,
) -> np.ndarray:
    """Draw a random card from the bank on `quad`; returns its canvas-sized alpha."""
    if quad_roi(quad, canvas.shape) is None:  # entirely outside the window: nothing to decode
        return np.zeros(canvas.shape[:2], np.float32)
    img, alpha = card_face(rng, cards, cards.sample_index(rng) if index is None else index, quad_short(quad), detail)
    if shadow and rng.random() < 0.7:
        # soft drop shadow: darken under a shifted, blurred copy of the card's alpha
        sh_quad = quad + rng.uniform(-6, 6, size=2).astype(np.float32)
        sigma = rng.uniform(2, 6)
        roi = quad_roi(expand(sh_quad, 1 + 4 * sigma / quad_short(sh_quad)), canvas.shape)  # room for the blur tail
        if roi is not None:
            x0, y0, x1, y1 = roi
            sh = warp_alpha(alpha, sh_quad, canvas.shape)[y0:y1, x0:x1]
            sh = cv2.GaussianBlur(sh, (0, 0), sigma)
            sh *= -rng.uniform(0.15, 0.45)
            sh += 1
            canvas[y0:y1, x0:x1] *= sh[..., None]
    return paste(canvas, img, alpha, quad)


def draw_sleeve_ring(canvas: np.ndarray, rng: np.random.Generator, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The part of a sleeve that shows outside the card: a slightly larger rounded card shape.
    Returns (ring quad, ring alpha) so the gloss can be applied over card and ring together."""
    ring = expand(quad, rng.uniform(1.025, 1.07))
    # sleeves are open at the top, so the card often sits low: shift the ring up a little
    ring[[0, 1]] += (ring[0] - ring[3]) / np.linalg.norm(ring[0] - ring[3]) * rng.uniform(0, 0.02) * quad_short(quad)
    dark = rng.random() < 0.6
    color = rng.uniform(5, 40, size=3) if dark else rng.uniform(30, 255, size=3)
    if rng.random() < 0.25:  # clear/frosted sleeve: ring is a translucent haze
        color = np.float32([230, 230, 235])
        opacity = rng.uniform(0.25, 0.6)
    else:
        opacity = 1.0
    unit = 200
    solid = np.empty((int(unit * 88 / 63), unit, 3), np.float32)
    solid[:] = color
    alpha = rounded_mask(solid.shape[1], solid.shape[0], CORNER_RADIUS * unit * 1.3) * opacity
    ring_alpha = paste(canvas, solid, alpha, ring)
    return ring, ring_alpha


def draw_toploader(canvas: np.ndarray, rng: np.random.Generator, quad: np.ndarray) -> np.ndarray:
    """A rigid top-loader (or semi-rigid card saver) around the card: a clear rectangle about
    a fifth larger than the card with square corners, a specular edge line where the plastic
    catches the light, a slight haze, and the card sitting low inside because the loader is
    open at the top. Its corners are a card-shaped rectangle a few percent of the short side
    outside the real ones, exactly where the corner heatmap would otherwise fire.
    Returns the loader alpha (canvas-sized) for the gloss pass."""
    short = quad_short(quad)
    up = (quad[0] - quad[3]) / np.linalg.norm(quad[0] - quad[3])
    sx, sy = rng.uniform(1.12, 1.26), rng.uniform(1.08, 1.2)  # 76x102 mm around 63x88, with some slop
    c = quad.mean(axis=0)
    right = (quad[1] - quad[0]) / np.linalg.norm(quad[1] - quad[0])
    hw, hh = short * sx / 2, short * 88 / 63 * sy / 2
    # the card rests on the loader's bottom edge: shift the loader up by most of the spare height
    c = c + up * rng.uniform(0.3, 1.0) * (hh - short * 88 / 63 / 2)
    loader = np.float32([c - right * hw + up * hh, c + right * hw + up * hh, c + right * hw - up * hh, c - right * hw - up * hh])
    unit = 200
    haze = np.empty((int(unit * hh / hw), unit, 3), np.float32)
    haze[:] = rng.uniform(200, 245, size=3)
    alpha = np.full(haze.shape[:2], rng.uniform(0.04, 0.22), np.float32)
    loader_alpha = paste(canvas, haze, alpha, loader)
    roi = quad_roi(expand(loader, 1.05), canvas.shape)
    if roi is not None:
        # specular edge: a thin bright (or dark, when it shadows the mat) line along the plastic's edge
        bright = rng.random() < 0.7
        color = tuple(float(v) for v in (rng.uniform(190, 255, size=3) if bright else rng.uniform(10, 70, size=3)))
        thickness = max(1, round(rng.uniform(0.008, 0.02) * short))
        x0, y0, x1, y1 = roi
        line = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.polylines(line, [np.round(loader - np.float32([x0, y0])).astype(np.int32)], True, 1.0, thickness, cv2.LINE_AA)
        a = (line * rng.uniform(0.5, 1.0))[..., None]
        dst = canvas[y0:y1, x0:x1]
        dst += (np.float32(color) - dst) * a
        np.maximum(loader_alpha[y0:y1, x0:x1], line, out=loader_alpha[y0:y1, x0:x1])
    return loader_alpha


def gloss(canvas: np.ndarray, rng: np.random.Generator, alpha: np.ndarray, quad: np.ndarray) -> None:
    """Sleeve/foil sheen over the card: a milky tint plus a highlight band or blob."""
    # `alpha` is zero outside the sleeve ring or top-loader, which sit within ~1.3x the card; work in that box
    roi = quad_roi(expand(quad, 1.4), canvas.shape)
    if roi is None:
        return
    x0, y0, x1, y1 = roi
    a = alpha[y0:y1, x0:x1]
    dst = canvas[y0:y1, x0:x1]
    tint = rng.uniform(0.03, 0.15)
    dst += (235 - dst) * (a * tint)[..., None]
    if rng.random() < 0.7:
        # highlights are smooth, so compute them on a coarse grid over the box and upsample
        step = 4
        xs = np.arange(x0, x1, step, dtype=np.float32) + step / 2
        ys = np.arange(y0, y1, step, dtype=np.float32) + step / 2
        c = quad.mean(axis=0)
        s = quad_short(quad)
        if rng.random() < 0.5:  # band across the card, any direction
            ang = rng.uniform(0, np.pi)
            d = (xs - c[0])[None, :] * np.cos(ang) + (ys - c[1])[:, None] * np.sin(ang) + rng.uniform(-0.5, 0.5) * s
            band = np.exp(-((d / (rng.uniform(0.06, 0.25) * s)) ** 2))
        else:
            gx, gy = c + rng.uniform(-0.5, 0.5, size=2) * s
            band = np.exp(-(((xs - gx)[None, :] / (rng.uniform(0.2, 0.6) * s)) ** 2 + ((ys - gy)[:, None] / (rng.uniform(0.2, 0.6) * s)) ** 2))
        band = cv2.resize(band.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        dst += (band * a * rng.uniform(40, 160))[..., None]


def occluders(canvas: np.ndarray, rng: np.random.Generator, quad: np.ndarray, cards: CardBank, detail: float = DET_INPUT / SCENE) -> None:
    c = quad.mean(axis=0)
    s = quad_short(quad)
    if rng.random() < 0.15:  # another card lying partly over this one (aura, equipment, a sloppy stack)
        off = rng.uniform(0.6, 1.1) * s
        a = rng.uniform(0, 2 * np.pi)
        q = quad_from_pose(c[0] + off * np.cos(a), c[1] + off * np.sin(a), s * rng.uniform(0.9, 1.1), rng.uniform(0, 360), rng)
        draw_card(canvas, rng, cards, q, detail=detail)
    if rng.random() < 0.3:  # dice and counters
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.uniform(0.05, 0.13) * s)
            p = c + rng.uniform(-0.9, 0.9, size=2) * s
            color = tuple(float(v) for v in rng.uniform(0, 255, size=3))
            cv2.circle(canvas, (int(p[0]), int(p[1])), r, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(p[0]), int(p[1])), r, tuple(v * 0.6 for v in color), 2, lineType=cv2.LINE_AA)
    if rng.random() < 0.15:  # a finger reaching in from an edge
        size = canvas.shape[0]
        edge = rng.integers(0, 4)
        length = rng.uniform(0.3, 0.7) * size
        width = rng.uniform(0.06, 0.12) * size
        start = {0: (rng.uniform(0, size), 0), 1: (size, rng.uniform(0, size)), 2: (rng.uniform(0, size), size), 3: (0, rng.uniform(0, size))}[int(edge)]
        toward = c - np.float32(start)
        toward = toward / (np.linalg.norm(toward) + 1e-6)
        end = np.float32(start) + toward * length
        skin = np.float32([rng.uniform(150, 240), rng.uniform(100, 180), rng.uniform(80, 150)])
        cv2.line(canvas, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), tuple(float(v) for v in skin), int(width), lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(end[0]), int(end[1])), int(width / 2), tuple(float(v) for v in skin * 0.95), -1, lineType=cv2.LINE_AA)


def photometrics(img: np.ndarray, rng: np.random.Generator, scale: float = DET_INPUT / SCENE) -> np.ndarray:
    """Webcam look: exposure, white balance, gamma, saturation, defocus, sensor noise, and
    the stream's compression. Applied at detector-input resolution, so blur and noise are
    scaled by `scale` (native px -> input px) from what a 1080p sensor produces. Returns uint8."""
    # exposure/white balance/contrast/brightness/gamma are per-value maps, so apply them to the
    # 256-entry channel LUTs rather than to every pixel
    levels = np.arange(256, dtype=np.float32)[:, None] * rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    levels = (levels - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-30, 30)
    levels = (255 * (np.clip(levels, 0, 255) / 255) ** rng.uniform(0.8, 1.25)).astype(np.float32)
    x = np.stack([levels[:, ch][img[..., ch]] for ch in range(3)], axis=2)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.7, 1.2)
    sigma = rng.uniform(0, 1.6) * scale
    if sigma > 0.15:
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    if rng.random() < 0.1:
        k = int(rng.integers(3, 6))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0), (k, k))
        x = cv2.filter2D(x, -1, kernel / max(kernel.sum(), 1e-6))
    # averaging 1/scale^2 sensor pixels per input pixel shrinks the noise by `scale`
    x += rng.standard_normal(size=x.shape, dtype=np.float32) * np.float32(rng.uniform(1, 8) * max(scale, 0.4))
    x = np.clip(x, 0, 255, out=x).astype(np.uint8)
    quality = int(rng.integers(50, 95))
    _ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def render_scene(
    rng: np.random.Generator,
    cards: CardBank,
    arts: ArtBank,
    size: int = SCENE,
    out: int = DET_INPUT,
    target_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a `size` x `size` native-pixel window and return it downscaled to `out` x `out`
    RGB uint8 with the 4x2 float32 printed-order corners of the clicked card in `out` pixels.
    `target_index` fixes the clicked scan for layout-stratified evaluation. Full scans retain
    both halves, their text and rotations; never paint a small half into a modern art box."""
    canvas = background(rng, arts, size)
    # short side of the clicked card: ~70 px (1080p over a 4-player table) to ~380 (4K, close)
    short = float(np.exp(rng.uniform(np.log(70), np.log(380))))
    angle = rng.uniform(0, 360)
    # neighbours underneath: other cards at a similar scale anywhere in the window
    for _ in range(int(rng.integers(0, 4))):
        q = quad_from_pose(rng.uniform(0, size), rng.uniform(0, size), short * rng.uniform(0.7, 1.3), rng.uniform(0, 360), rng)
        draw_card(canvas, rng, cards, q, detail=out / size)
    # pose the clicked card so a random point on its face sits at the window centre
    quad = quad_from_pose(0, 0, short, angle, rng)
    u, v = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
    click = (1 - v) * ((1 - u) * quad[0] + u * quad[1]) + v * ((1 - u) * quad[3] + u * quad[2])
    quad = quad - click + np.float32([size / 2, size / 2]) + rng.uniform(-12, 12, size=2).astype(np.float32)
    sleeved = rng.random() < 0.55
    loader_alpha = draw_toploader(canvas, rng, quad) if rng.random() < 0.15 else None
    ring_alpha = None
    if sleeved:
        _, ring_alpha = draw_sleeve_ring(canvas, rng, quad)
    card_alpha = draw_card(canvas, rng, cards, quad, shadow=not sleeved, detail=out / size, index=target_index)
    if loader_alpha is not None:
        # the loader's plastic catches the light over card, sleeve and its own margin alike
        alpha = np.maximum(card_alpha, loader_alpha) if ring_alpha is None else np.maximum(np.maximum(card_alpha, ring_alpha), loader_alpha)
        gloss(canvas, rng, alpha, quad)
    elif sleeved:
        gloss(canvas, rng, np.maximum(card_alpha, ring_alpha), quad)
    elif rng.random() < 0.25:  # foil or a glossy unsleeved card
        gloss(canvas, rng, card_alpha, quad)
    occluders(canvas, rng, quad, cards, detail=out / size)
    small = cv2.resize(np.clip(canvas, 0, 255, out=canvas).astype(np.uint8), (out, out), interpolation=cv2.INTER_AREA)
    return photometrics(small, rng, out / size), (quad * (out / size)).astype(np.float32)
