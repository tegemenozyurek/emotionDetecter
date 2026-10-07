"""Face alignment shared by training and the live web app.

MediaPipe's face detector returns 6 keypoints per face. We use four of them
(right eye, left eye, nose tip, mouth center) and fit a similarity transform
(scale + rotation + translation) that moves them onto a fixed template. Every
face, in training images and in webcam frames, is cut out this same way, so
the model always sees faces framed identically.

web/app.js implements exactly the same math (fitSimilarity) in JavaScript.
"""
import numpy as np

KEYPOINTS = [0, 1, 2, 3]  # right eye, left eye, nose tip, mouth center


def fit_similarity(src, dst):
    """Least-squares similarity transform (Umeyama, no reflection) mapping src -> dst.
    src, dst: (N, 2). Returns a 2x3 matrix M with dst ≈ M @ [x, y, 1]."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    var_s = (s ** 2).sum() / len(src)
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    sign = np.diag([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rot = u @ sign @ vt
    scale = (sig * np.diag(sign)).sum() / var_s
    t = mu_d - scale * rot @ mu_s
    return np.hstack([scale * rot, t[:, None]])


def plausible(kp, width, height):
    """Reject keypoint sets that cannot be a frontal-ish face."""
    kp = np.asarray(kp)
    right_eye, left_eye, nose, mouth = kp[KEYPOINTS]
    eye_dist = np.linalg.norm(left_eye - right_eye)
    eye_mid = (left_eye + right_eye) / 2
    size = max(width, height)
    return (0.12 * size < eye_dist < 0.9 * size            # sensible face size
            and np.linalg.norm(mouth - eye_mid) > 0.4 * eye_dist  # mouth below the eyes
            and np.linalg.norm(nose - eye_mid) < 1.5 * eye_dist)
