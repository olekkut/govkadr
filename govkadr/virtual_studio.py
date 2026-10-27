"""virtual-studio mode: physically based relighting with virtual softboxes (no generative model).

1. Geometry. Per-pixel surface normals: the MediaPipe mesh supplies real 3-D face geometry (its z is
   in the same units as x/y), the matte silhouette supplies a rounded "inflated" shape for hair,
   neck and shoulders, and the two are feathered together.
2. Existing light. The photo's illumination is modelled as 2nd-order spherical harmonics of the
   normal (9 coefficients, the standard low-frequency model of diffuse light; Ramamoorthi &
   Hanrahan 2001), fitted to skin luminance, where albedo is roughly constant.
3. New light. Irradiance from virtual area lights - each softbox is sampled as a disc of the given
   angular size - plus a white studio bounce and optional rim light.
4. The photo is multiplied by new / old irradiance. The field is smooth (it depends only on the
   normal), so every pore, mole and hair in the result is the camera's: no pixel is generated.

Self-shadowing is not modelled; with large soft sources it is faint, and shadows already baked
into the photo are softened only as far as the spherical-harmonics fit explains them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@dataclass(frozen=True)
class Light:
    azimuth: float  # degrees, + = from the image's right side
    elevation: float  # degrees, + = from above
    size: float  # angular radius of the source in degrees (softbox ~ 20-40)
    intensity: float

    def direction(self) -> np.ndarray:
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        # Light space: x right, y down, z towards the camera.
        return np.array([math.sin(az) * math.cos(el), -math.sin(el), math.cos(az) * math.cos(el)])


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    lights: tuple[Light, ...]
    ambient: float  # white studio bounce, weighted towards camera-facing surfaces
    id_suitable: bool


PRESETS = {
    "clamshell": Preset(
        "clamshell",
        "large key softbox above the lens + reflector below: even, symmetric, ID-friendly",
        (Light(0, 30, 28, 1.0), Light(0, -28, 32, 0.45)),
        0.45,
        True,
    ),
    "high-key": Preset(
        "high-key",
        "big frontal softbox + two side fills + strong white bounce: almost shadowless",
        (Light(0, 15, 40, 1.0), Light(60, 10, 30, 0.35), Light(-60, 10, 30, 0.35)),
        0.7,
        True,
    ),
    "loop": Preset(
        "loop",
        "classic portrait: key 35 deg to the side and above, soft fill, hair/rim light (not for ID)",
        (Light(-40, 35, 20, 1.0), Light(45, 0, 35, 0.2), Light(150, 40, 15, 0.5)),
        0.08,
        False,
    ),
}


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def mesh_normals(pts3: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel normals (H, W, 3) inside the face mesh, and its coverage mask."""
    v = pts3[:468].astype(np.float64)
    tri = Delaunay(v[:, :2])
    faces = tri.simplices
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    fn = np.cross(b - a, c - a)  # area-weighted face normals, (x right, y down, z away from camera)
    fn[fn[:, 2] > 0] *= -1  # orient towards the camera (negative MediaPipe z)
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn = _normalize(vn)
    vn[:, 2] *= -1  # light space: z towards the camera
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    interp = LinearNDInterpolator(v[:, :2], vn)
    n = interp(np.column_stack([xs.ravel(), ys.ravel()])).reshape(h, w, 3)
    inside = np.isfinite(n[..., 0])
    n[~inside] = 0
    return _normalize(n).astype(np.float32), inside


def silhouette_normals(alpha: np.ndarray, radius: float) -> np.ndarray:
    """Normals of the matte "inflated" into a rounded relief (circular profile of ``radius`` px)."""
    mask = (alpha > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    d = np.minimum(d, radius)
    hgt = np.sqrt(np.maximum(2 * radius * d - d * d, 0))
    hgt = cv2.GaussianBlur(hgt, (0, 0), radius * 0.08 + 1)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    # Height rises towards the camera, so the normal is (-dh/dx, -dh/dy, 1).
    return _normalize(np.dstack([-gx, -gy, np.ones_like(gx)]))


def sh_basis(n: np.ndarray) -> np.ndarray:
    """Real spherical harmonics up to order 2 (9 terms) of unit normals (..., 3)."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack(
        [
            np.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3 * z * z - 1),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ],
        axis=-1,
    )


def irradiance(n: np.ndarray, preset: Preset, samples: int = 24) -> np.ndarray:
    """Diffuse irradiance from the preset's area lights and studio bounce."""
    e = preset.ambient * (0.5 + 0.5 * np.clip(n[..., 2], -1, 1))
    golden = math.pi * (3 - math.sqrt(5))
    for light in preset.lights:
        d = light.direction()
        # Orthonormal frame around the light direction; sample the disc with a Fibonacci spiral.
        t = np.cross(d, [0.0, 1.0, 0.0] if abs(d[1]) < 0.9 else [1.0, 0.0, 0.0])
        t /= np.linalg.norm(t)
        bt = np.cross(d, t)
        acc = np.zeros(n.shape[:2], np.float32)
        tan_r = math.tan(math.radians(light.size))
        for i in range(samples):
            r = tan_r * math.sqrt((i + 0.5) / samples)
            phi = i * golden
            ld = d + r * (math.cos(phi) * t + math.sin(phi) * bt)
            ld /= np.linalg.norm(ld)
            acc += np.clip(n @ ld.astype(np.float32), 0, None)
        e = e + light.intensity * acc / samples
    return e.astype(np.float32)


@dataclass
class Geometry:
    normals: np.ndarray  # quarter resolution
    skin: np.ndarray
    face_weight: np.ndarray
    sh_old: np.ndarray  # fitted 9-term spherical harmonics of the existing light
    fit_r2: float


def build_geometry(lin: np.ndarray, alpha: np.ndarray, pts3_work: np.ndarray, skin_work: np.ndarray, face_w: float, q: int = 4) -> Geometry:
    h, w = alpha.shape
    small = (w // q, h // q)
    a = cv2.resize(alpha, small, interpolation=cv2.INTER_AREA)
    p = pts3_work.copy()
    p[:, :2] = (p[:, :2] + 0.5) / q - 0.5
    p[:, 2] /= q
    n_face, inside = mesh_normals(p, a.shape)
    n_sil = silhouette_normals(a, 0.42 * face_w / q)
    wf = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    wf = cv2.GaussianBlur(wf, (0, 0), 0.03 * face_w / q)
    n = _normalize(wf[..., None] * n_face + (1 - wf[..., None]) * n_sil)
    n = _normalize(cv2.GaussianBlur(n, (0, 0), 2.0))  # soften fine mesh detail (eye sockets)
    skin = cv2.resize(skin_work.astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0.99
    y = cv2.resize(lin @ _LUMA, small, interpolation=cv2.INTER_AREA)
    # Shading is low-frequency; pores, stubble and freckles are not. Fit on skin-normalised blur.
    sig = 0.03 * face_w / q
    sk = skin.astype(np.float32)
    y = cv2.GaussianBlur(y * sk, (0, 0), sig) / np.maximum(cv2.GaussianBlur(sk, (0, 0), sig), 1e-4)
    basis = sh_basis(n[skin])
    target = y[skin] / np.median(y[skin])
    lam = 0.005 * len(target)
    coef = np.linalg.solve(basis.T @ basis + lam * np.diag([0, 1, 1, 1, 4, 4, 4, 4, 4]), basis.T @ target)
    pred = basis @ coef
    r2 = 1 - float(np.var(target - pred) / max(np.var(target), 1e-9))
    return Geometry(n, skin, wf, coef.astype(np.float32), r2)


def relight(lin: np.ndarray, alpha: np.ndarray, geo: Geometry, preset: Preset, removal: float = 0.8, max_ev: float = 1.2) -> tuple[np.ndarray, np.ndarray, dict]:
    # ID presets may brighten freely but darken at most ~0.6 EV (no contouring on an ID photo).
    min_ev = 0.6 if preset.id_suitable else max_ev
    """Multiply the photo by new / old irradiance (exposure-normalised on skin)."""
    h, w = alpha.shape
    e_old = np.clip(sh_basis(geo.normals) @ geo.sh_old, 0.25, 4.0)
    e_new = irradiance(geo.normals, preset)
    # Spherical harmonics fitted on the face extrapolate poorly to hair/shoulder normals: remove
    # the old light fully on the face, partially elsewhere.
    beta = removal * (0.5 + 0.5 * geo.face_weight)
    ratio = e_new / np.power(e_old, beta)
    ratio /= np.median(ratio[geo.skin])
    log_g = np.clip(np.log(ratio), -min_ev * math.log(2), max_ev * math.log(2))
    log_g = cv2.GaussianBlur(log_g.astype(np.float32), (0, 0), 1.5)
    gain = np.exp(cv2.resize(log_g, (w, h), interpolation=cv2.INTER_CUBIC))
    ev = log_g[geo.skin] / math.log(2)
    info = {
        "preset": preset.name,
        "description": preset.description,
        "id_suitable": preset.id_suitable,
        "skin_gain_ev_p5_p50_p95": [round(float(v), 3) for v in np.percentile(ev, [5, 50, 95])],
        "old_light_fit_r2": round(geo.fit_r2, 3),
    }
    return lin * gain[..., None], gain, info


def normals_visual(n: np.ndarray, alpha_small: np.ndarray) -> np.ndarray:
    vis = 0.5 + 0.5 * np.dstack([n[..., 0], -n[..., 1], n[..., 2]])
    a = alpha_small[..., None]
    return (vis * a + 0.92 * (1 - a)).astype(np.float32)
