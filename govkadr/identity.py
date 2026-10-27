"""Face-identity embeddings for the biometric drift audit.

InsightFace buffalo_l: SCRFD-10GF detector (5 keypoints) + ArcFace ResNet-50 trained on
WebFace600K, run through ONNX Runtime on the CPU (no VRAM contention with the generative stage).
Faces are aligned to the canonical 112x112 ArcFace template with a least-squares similarity
transform (Umeyama), exactly as the reference implementation does.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from . import geometry as g
from . import weights

ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (3x3) mapping src points onto dst points."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1
    r = u @ s @ vt
    scale = np.trace(np.diag(d) @ s) / sc.var(0).sum()
    m = np.eye(3)
    m[:2, :2] = scale * r
    m[:2, 2] = mu_d - scale * r @ mu_s
    return m


class FaceIdentity:
    def __init__(self, threads: int = 8):
        d = weights.fetch_url(weights.INSIGHTFACE_BUFFALO_L)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.det = ort.InferenceSession(str(d / "det_10g.onnx"), so, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(d / "w600k_r50.onnx"), so, providers=["CPUExecutionProvider"])
        self.det_in = self.det.get_inputs()[0].name
        self.rec_in = self.rec.get_inputs()[0].name

    def detect(self, rgb: np.ndarray, size: int = 640, thresh: float = 0.5) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """SCRFD: returns [(bbox xyxy, score, 5x2 keypoints)] in ``rgb`` pixel coordinates."""
        h, w = rgb.shape[:2]
        s = size / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        m = g.resample_scale(nw / w, nh / h)
        small = g.warp(rgb, m, (nw, nh), interp=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), np.float32)
        canvas[:nh, :nw] = small
        blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.det.run(None, {self.det_in: blob})
        outs = [o[0] if o.ndim == 3 else o for o in outs]
        boxes, scores, kpss = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            sc, bb, kp = outs[i][:, 0], outs[i + 3] * stride, outs[i + 6] * stride
            fh, fw = size // stride, size // stride
            ctr = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            ctr = np.repeat(ctr, 2, axis=0)  # two anchors per location
            keep = np.where(sc >= thresh)[0]
            for k in keep:
                c = ctr[k]
                boxes.append([c[0] - bb[k, 0], c[1] - bb[k, 1], c[0] + bb[k, 2], c[1] + bb[k, 3]])
                kpss.append(c + kp[k].reshape(5, 2))
                scores.append(float(sc[k]))
        if not boxes:
            return []
        boxes, scores, kpss = np.array(boxes), np.array(scores), np.array(kpss)
        order = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes.tolist()], scores.tolist(), thresh, 0.4)
        inv = np.linalg.inv(m)
        res = []
        for i in np.array(order).reshape(-1):
            bb = g.apply(inv, boxes[i].reshape(2, 2)).reshape(-1)
            res.append((bb, scores[i], g.apply(inv, kpss[i])))
        return sorted(res, key=lambda r: -(r[0][2] - r[0][0]) * (r[0][3] - r[0][1]))

    def embed(self, rgb: np.ndarray) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """L2-normalised 512-d ArcFace embedding of the largest face."""
        faces = self.detect(rgb)
        if not faces:
            return None, {"faces": 0}
        bbox, score, kps = faces[0]
        m = umeyama(kps, ARCFACE_TEMPLATE)
        crop = g.warp(rgb, m, (112, 112), interp=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT)
        blob = ((crop.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = self.rec.run(None, {self.rec_in: blob})[0][0]
        return e / np.linalg.norm(e), {"faces": len(faces), "det_score": round(score, 4), "kps": np.round(kps, 1).tolist()}


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b))
