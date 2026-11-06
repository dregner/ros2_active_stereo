"""Projector light-sheet consistency for fringe-projection clouds.

The fringe z-scan accepts a lattice node when the absolute phase seen by the
left and right cameras agrees (|phi_L - phi_R| below a threshold). The phase
only encodes the projector *column*, so any pair of camera pixels that observe
the same column agrees, wherever the two pixels actually look. Over a long
depth sweep every background node therefore finds a chance agreement, and
because the phase maps are smooth those chance matches form smooth phantom
surfaces that no purely geometric outlier filter can tell from real ones.

Physics gives the missing constraint: every surface point lit by projector
column ``c`` lies on that column's light sheet, a plane through the projector
centre. The sheets form a pencil, i.e. the projector is a pinhole whose column
coordinate is observed through the phase::

    phi(X) = a * (r1 . (X - C)) / (r3 . (X - C)) + b

with ``R = [r1; r2; r3]`` the projector orientation and ``C`` its centre,
which is a ratio of two affine functions of ``X``. No projector calibration is
needed: the eight coefficients are fitted robustly (IRLS-DLT) to the
reconstruction itself, where the real surface dominates, and points whose phase
disagrees with the model by more than a few times the residual scale are
phantoms. Unlike fitting one plane per phase bin, this is well-posed even when
a column's trace on the object is a straight line (e.g. a cylinder whose axis
is parallel to the sheets).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class LightSheetReport:
    input_points: int
    kept_points: int
    fitted_points: int
    residual_sigma_rad: float
    threshold_rad: float
    projector_center: list

    def __str__(self) -> str:
        return (f'light-sheet consistency: {self.input_points} -> {self.kept_points} points '
                f'(projector column model fitted on {self.fitted_points} points, residual sigma '
                f'{self.residual_sigma_rad:.3f} rad, rejected beyond {self.threshold_rad:.2f} rad; '
                f'projector centre ~ {np.round(self.projector_center, 0).tolist()} mm)')


def _model(params: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Projective column model ``phi = (P.X + p0) / (Q.X + q0)`` (8 parameters, defined up to scale)."""
    num = xyz @ params[:3] + params[3]
    den = xyz @ params[4:7] + params[7]
    return num / den


def fit_projector_column_model(xyz: np.ndarray, phase: np.ndarray, max_points: int = 20000, seed: int = 0,
                               iterations: int = 12, ransac_samples: int = 300, inlier_rad: float = 0.3,
                               center_prior: Optional[np.ndarray] = None,
                               prior_weight: float = 0.05) -> Tuple[np.ndarray, float]:
    """Robust fit of the projective column model. Returns (params, robust residual sigma in rad).

    The model is linear in its parameters once written as
    ``phi * (Q.X + q0) - (P.X + p0) = 0``, so each IRLS round is a DLT (SVD of
    the weighted design matrix); Tukey weights computed from the *geometric*
    residual remove phantom points, whose residuals reach tens of radians.
    No initial projector pose is required. Coordinates are normalised for
    conditioning and the parameters mapped back afterwards.

    A planar scene cannot determine the projector centre (a homography already
    explains the phase on the plane), which leaves the prediction *off* the
    plane - where the phantoms are - undetermined. ``center_prior`` adds two
    soft linear rows ``P.C + p0 = 0`` and ``Q.C + q0 = 0`` that place the
    centre near a plausible position (e.g. between the cameras); with a
    curved scene the data dominate and the prior is irrelevant.
    """
    rng = np.random.default_rng(seed)
    if len(xyz) > max_points:
        pick = rng.choice(len(xyz), max_points, replace=False)
        xyz, phase = xyz[pick], phase[pick]
    c = xyz.mean(0)
    s = xyz.std(0).max() + 1e-9
    x = (xyz - c) / s
    pc = phase.mean()
    ps = phase.std() + 1e-9
    f = (phase - pc) / ps
    ones = np.ones(len(x))
    design = np.c_[x, ones, -f[:, None] * x, -f]
    prior_rows = None
    if center_prior is not None:
        cn = (np.asarray(center_prior, float) - c) / s
        prior_rows = np.array([[*cn, 1., 0., 0., 0., 0.], [0., 0., 0., 0., *cn, 1.]])

    def solve(rows: np.ndarray, row_w: np.ndarray) -> np.ndarray:
        rows = rows * row_w[:, None]
        if prior_rows is not None:
            rows = np.r_[rows, prior_rows * prior_weight * np.sqrt(max(float((row_w > 0).sum()), 1.))]
        _, _, vt = np.linalg.svd(rows, full_matrices=False)
        return vt[-1]

    def geometric(p: np.ndarray):
        den = x @ p[4:7] + p[7]
        den = np.where(np.abs(den) < 1e-6, 1e-6, den)
        return (x @ p[:3] + p[3]) / den - f, den

    # RANSAC initialisation: a plain least-squares start is dominated by the
    # phantoms (their residuals are enormous), so sample minimal 8-point sets
    # and keep the model explaining most points within ``inlier_rad``.
    inlier = inlier_rad / ps
    best, best_count = None, -1
    for _ in range(ransac_samples):
        idx = rng.choice(len(x), 8, replace=False)
        p = solve(design[idx], np.ones(8))
        count = int((np.abs(geometric(p)[0]) <= inlier).sum())
        if count > best_count:
            best, best_count = p, count
    resid, den = geometric(best)
    w = (np.abs(resid) <= inlier).astype(float)
    params = best
    for _ in range(iterations):
        # Dividing each row by the current denominator turns the algebraic error
        # into the geometric one at the current estimate (iteratively re-weighted
        # DLT), so the IRLS converges to the robust geometric solution.
        params = solve(design, w / np.abs(den))
        resid, den = geometric(params)
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.02 / ps)
        t = np.abs(resid) / (4.685 * scale)
        w = np.where(t < 1, (1 - t * t) ** 2, 0.)
    # Map back to raw coordinates: phase = ps * model_norm + pc
    P, p0, Q, q0 = params[:3], params[3], params[4:7], params[7]
    P_raw, Q_raw = P / s, Q / s
    p0_raw, q0_raw = p0 - P_raw @ c, q0 - Q_raw @ c
    raw = np.r_[ps * P_raw + pc * Q_raw, ps * p0_raw + pc * q0_raw, Q_raw, q0_raw]
    resid = _model(raw, xyz) - phase
    good = w > 0
    sigma = 1.4826 * np.median(np.abs(resid[good] - np.median(resid[good]))) if good.any() else float('inf')
    return raw, float(sigma)


def light_sheet_mask(zscan, xyz: torch.Tensor, sigma_factor: float = 6., floor_rad: float = 0.3,
                     max_points: int = 20000):
    """Keep points whose absolute phase agrees with the fitted projector column model.

    Parameters
    ----------
    zscan : PyTorchStereoCorrel with fringe images loaded (channel 0 = absolute phase).
    sigma_factor, floor_rad : rejection threshold ``max(sigma_factor * sigma, floor_rad)``
        where ``sigma`` is the robust residual scale of the fit. The floor covers
        projector lens distortion, which the pinhole model ignores; phantom
        matches are off by several radians (often by whole fringe periods).
    """
    n = len(xyz)
    keep = torch.ones(n, dtype=torch.bool, device=xyz.device)
    if n < 500:
        return keep, LightSheetReport(n, n, 0, 0., 0., [0., 0., 0.])
    uv_l, in_l = zscan.transform_gcs2ccs(xyz, 'left', image_shape=zscan.left_images.shape[1:])
    uv_r, in_r = zscan.transform_gcs2ccs(xyz, 'right', image_shape=zscan.right_images.shape[1:])
    ph_l = zscan.interpolate_images(zscan.left_images, uv_l, uv_mask=in_l, method='fringe')[:, 0]
    ph_r = zscan.interpolate_images(zscan.right_images, uv_r, uv_mask=in_r, method='fringe')[:, 0]
    phase = torch.where(torch.isfinite(ph_l), ph_l, ph_r)
    phase = torch.where(torch.isfinite(phase) & torch.isfinite(ph_r), 0.5 * (phase + ph_r), phase)
    valid = torch.isfinite(phase)
    pts = xyz[valid].double().cpu().numpy()
    ph = phase[valid].double().cpu().numpy()
    if len(pts) < 500:
        return keep, LightSheetReport(n, n, 0, 0., 0., [0., 0., 0.])

    cams = [zscan.camera_params[c] for c in ('left', 'right')]
    centers = [(-cam['r'].T @ cam['t']).squeeze(1).double().cpu().numpy() for cam in cams]
    params, sigma = fit_projector_column_model(pts, ph, max_points=max_points,
                                               center_prior=0.5 * (centers[0] + centers[1]))
    thr = max(sigma_factor * sigma, floor_rad)
    resid = np.abs(_model(params, pts) - ph)
    ok = torch.as_tensor(np.isfinite(resid) & (resid <= thr), device=xyz.device)
    keep[torch.nonzero(valid).squeeze(1)] = ok
    # The projector centre is the intersection of the planes P.X + p0 = 0 and
    # Q.X + q0 = 0 with the (unobservable) row direction: report the point of
    # that line closest to the cloud centroid for diagnostics only.
    A = np.stack([params[:3], params[4:7]])
    b = -np.array([params[3], params[7]])
    centre = np.linalg.lstsq(A, b - A @ pts.mean(0), rcond=None)[0] + pts.mean(0)
    report = LightSheetReport(n, int(keep.sum()), min(len(pts), max_points), sigma, thr, centre.tolist())
    return keep, report
