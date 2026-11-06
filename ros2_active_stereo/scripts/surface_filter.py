"""Surface-aware outlier rejection for inverse-triangulation (z-scan) clouds.

The z-scan produces *one depth per XY lattice node*, so a reconstruction is a
2.5-D depth map ``z(x, y)`` sampled on a regular grid, not an unstructured
cloud. The original Euclidean filter ignored this: it counted neighbours in a
3-D ball, which

* is O(N^2) (all-pairs ``cdist``),
* erodes borders and thins steep or sparsely-populated surface regions
  (neighbour spacing grows with slope, so fewer points fall in the ball), and
* cannot tell a spike of a few mm from the surface (everything within the
  ball counts, whatever its depth).

``LatticeSurfaceFilter`` replaces it with a local-surface consistency test
that exploits the lattice:

1. Rasterise the candidates into a depth image ``Z[ix, iy]``.
2. For every node, gather the lattice neighbours inside an XY disk of
   ``radius`` mm (a fixed stencil, so this is a gather instead of a search).
3. Fit a robust local surface (plane or quadratic patch, IRLS / Tukey)
   through those neighbours *only*, starting from the neighbours whose depth
   is close to the node's own depth (a depth gate) so that the fit describes
   the surface the node belongs to even next to a depth discontinuity.
4. Keep the node when it lies within ``tolerance`` of that surface, at least
   ``min_neighbors`` neighbours are themselves within ``tolerance`` of it,
   those consistent neighbours are the majority of the gated neighbourhood and
   they are spread in 2-D (a one-cell-wide string of wrong depths along an
   edge is self-consistent but is not a surface).

A tilted or curved surface has many consistent neighbours, a border node still
has a half-disk of them and both sides of a step keep their own surface, while
a spike, an isolated point, or a node inside a random-depth region (even one
confined to a narrow depth band) does not. Complexity is O(N * K) with K the
stencil size, fully vectorised on the GPU.

The same lattice structure is used to turn the filtered coarse scan into a
dense, smooth *seed depth map* that guides the fine pass
(:func:`build_seed_map`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class FilterReport:
    """Counts produced by one filter application (useful for diagnostics)."""
    input_points: int
    kept_points: int
    tolerance: float
    radius: float
    min_neighbors: int
    rejected_off_plane: int
    rejected_few_neighbors: int

    def __str__(self) -> str:
        return (f'lattice surface filter: {self.input_points} -> {self.kept_points} points '
                f'(radius {self.radius:g} mm, tol {self.tolerance:g} mm, >= {self.min_neighbors} consistent neighbours; '
                f'off-plane {self.rejected_off_plane}, sparse {self.rejected_few_neighbors})')


def lattice_indices(xyz: torch.Tensor, x_vals: torch.Tensor, y_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Map XY coordinates back to their lattice node indices."""
    step = float(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 1.0
    ix = torch.round((xyz[:, 0].double() - float(x_vals[0])) / step).long()
    iy = torch.round((xyz[:, 1].double() - float(y_vals[0])) / step).long()
    return ix, iy, step


def disk_stencil(radius: float, step: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lattice offsets (di, dj) with 0 < ||(di, dj)|| * step <= radius."""
    r_cells = int(radius // step)
    if r_cells < 1:
        raise ValueError(f'radius ({radius} mm) must cover at least one lattice step ({step} mm)')
    rng = torch.arange(-r_cells, r_cells + 1, device=device)
    di, dj = torch.meshgrid(rng, rng, indexing='ij')
    di, dj = di.ravel(), dj.ravel()
    keep = ((di * di + dj * dj).double().sqrt() * step <= radius + 1e-9) & ((di != 0) | (dj != 0))
    return di[keep], dj[keep]


class LatticeSurfaceFilter:
    """Local-plane consistency filter on the z-scan lattice.

    Parameters
    ----------
    radius : float
        XY neighbourhood radius in mm (same meaning as the old Euclidean radius).
    min_neighbors : int
        Minimum number of neighbours that must agree with the local surface
        (raised internally to twice the model's parameter count: 6 for a
        plane, 12 for a quadratic patch).
    tolerance : float, optional
        Maximum distance (mm) from the local surface for a node or neighbour to
        be considered consistent. Defaults to ``radius / 10`` with a floor of
        two depth steps when ``z_step`` is given, which ties the slab thickness
        to the depth quantisation of the pass being filtered.
    z_step : float, optional
        Depth sampling of the scan that produced the points (mm).
    adaptive : bool
        Estimate the residual noise of the cloud (robust sigma of the node-to-
        local-surface residuals) and tighten the tolerance to ``3 * sigma`` when
        that is smaller than ``tolerance``; never below two depth steps. The
        nominal tolerance thus acts as an upper bound and the effective value
        follows the actual measurement noise (``report.tolerance``).
    order : int
        Local surface model: 1 = plane, 2 = quadratic patch. The quadratic
        model tolerates curvature inside the neighbourhood (cylinders, fillets,
        smooth steps) at the price of needing a few more consistent neighbours.
    gate : float, optional
        Depth gate (mm): only neighbours within ``gate`` of the node's depth
        seed the fit. Defaults to ``max(3 * tolerance, radius / 2)``.
    min_ratio : float
        Minimum fraction of the gated neighbours that must be consistent.
    max_slope : float
        Maximum |dz/dxy| of the local surface at the node (mm/mm; 0 = no
        limit). A depth map cannot represent surfaces nearly parallel to the
        viewing rays; chance matches at silhouettes produce wild local slopes.
    iterations : int
        IRLS iterations of the robust fit.
    chunk : int
        Nodes processed per batch (bounds memory to ``chunk * K`` floats).
    """

    def __init__(self, radius: float = 10., min_neighbors: int = 5, tolerance: Optional[float] = None,
                 z_step: Optional[float] = None, order: int = 2, gate: Optional[float] = None,
                 min_ratio: float = 0.5, adaptive: bool = True, max_slope: float = 0.,
                 iterations: int = 3, chunk: int = 16384):
        if radius <= 0 or min_neighbors < 1:
            raise ValueError('radius must be positive and min_neighbors >= 1')
        if order not in (1, 2):
            raise ValueError('order must be 1 (plane) or 2 (quadratic)')
        self.order = int(order)
        self.min_ratio = float(min_ratio)
        self.max_slope = float(max_slope)
        self.radius = float(radius)
        self.min_neighbors = int(min_neighbors)
        if tolerance is None:
            tolerance = self.radius / 10.
            if z_step is not None:
                tolerance = max(tolerance, 2. * float(z_step))
        if tolerance <= 0:
            raise ValueError('tolerance must be positive')
        self.tolerance = float(tolerance)
        self.z_step = None if z_step is None else float(z_step)
        self.adaptive = bool(adaptive)
        self.gate = None if gate is None else float(gate)
        self.iterations = int(iterations)
        self.chunk = int(chunk)
        self.report: Optional[FilterReport] = None

    # ── core ─────────────────────────────────────────────────────────────────

    def mask(self, xyz: torch.Tensor, x_vals: torch.Tensor, y_vals: torch.Tensor,
             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Boolean keep-mask for points that lie on the XY lattice ``x_vals`` x ``y_vals``."""
        n = xyz.shape[0]
        device = xyz.device
        if n == 0:
            self.report = FilterReport(0, 0, self.tolerance, self.radius, self.min_neighbors, 0, 0)
            return torch.zeros(0, dtype=torch.bool, device=device)

        ix, iy, step = lattice_indices(xyz, x_vals, y_vals)
        nx, ny = len(x_vals), len(y_vals)
        on_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        di, dj = disk_stencil(self.radius, step, device)
        pad = int(max(di.abs().max(), dj.abs().max()))

        # Depth image (NaN = no candidate). If several points share a node (e.g.
        # merged regions) the last write wins; callers normally pass one per node.
        z_img = torch.full((nx + 2 * pad, ny + 2 * pad), float('nan'), dtype=torch.float32, device=device)
        z_img[ix[on_grid] + pad, iy[on_grid] + pad] = xyz[on_grid, 2].float()
        w_img = None
        if weights is not None:
            w_img = torch.zeros_like(z_img)
            w_img[ix[on_grid] + pad, iy[on_grid] + pad] = weights[on_grid].float().clamp_min(0.)

        # Shared design matrix for the stencil; the constant term is last so that
        # the fitted value at the centre is simply the last coefficient.
        A = design_matrix(di, dj, step, self.order)  # (1, K, P)
        # A fit is only meaningful when clearly over-determined: demand at least
        # twice the parameter count of consistent neighbours.
        min_needed = max(self.min_neighbors, 2 * A.shape[-1])
        ctx = dict(xyz=xyz, ix=ix + pad, iy=iy + pad, idx=torch.nonzero(on_grid).squeeze(1), z_img=z_img,
                   w_img=w_img, di=di, dj=dj, A=A, step=step, min_needed=min_needed)

        tol = self.tolerance
        if self.adaptive:
            resid, n_cons, _, _, _ = self._evaluate(ctx, tol)
            reliable = torch.isfinite(resid) & (n_cons >= min_needed)
            if int(reliable.sum()) >= 50:
                sigma = 1.4826 * torch.median(resid[reliable].abs()).item()
                floor = 2. * self.z_step if self.z_step is not None else tol / 4.
                tol = float(min(tol, max(3. * sigma, floor)))

        resid, n_cons, n_gated, spread_ok, slope = self._evaluate(ctx, tol)
        ok_plane = torch.isfinite(resid) & (resid.abs() <= tol)
        if self.max_slope > 0:
            ok_plane &= slope <= self.max_slope
        ok_count = (n_cons >= min_needed) & (n_cons >= self.min_ratio * n_gated) & spread_ok
        keep = ok_plane & ok_count
        self.report = FilterReport(n, int(keep.sum()), tol, self.radius, self.min_neighbors,
                                   int((~ok_plane & ok_count).sum()), int((~ok_count).sum()))
        return keep

    def _evaluate(self, ctx: dict, tol: float):
        """Per-node centre residual, consistent/gated neighbour counts, spread flag and local slope."""
        xyz, idx_all, step = ctx['xyz'], ctx['idx'], ctx['step']
        n = xyz.shape[0]
        device = xyz.device
        resid_all = torch.full((n,), float('nan'), device=device)
        slope_all = torch.full((n,), float('inf'), device=device)
        n_cons_all = torch.zeros(n, dtype=torch.long, device=device)
        n_gated_all = torch.zeros(n, dtype=torch.long, device=device)
        spread_all = torch.zeros(n, dtype=torch.bool, device=device)
        di, dj, A = ctx['di'], ctx['dj'], ctx['A']
        gate = max(3. * tol, self.radius / 2.) if self.gate is None else self.gate
        for start in range(0, len(idx_all), self.chunk):
            sel = idx_all[start:start + self.chunk]
            cx, cy = ctx['ix'][sel], ctx['iy'][sel]
            zn = ctx['z_img'][cx[:, None] + di[None, :], cy[:, None] + dj[None, :]]  # (m, K)
            zc = xyz[sel, 2].float()
            gated = torch.isfinite(zn) & ((zn - zc[:, None]).abs() <= gate)
            w0 = gated.float()
            if ctx['w_img'] is not None:
                w0 = w0 * ctx['w_img'][cx[:, None] + di[None, :], cy[:, None] + dj[None, :]]
            params, resid = self._robust_fit(A, zn, w0, tol)
            consistent = gated & (resid.abs() <= tol)
            resid_all[sel] = zc - params[:, -1]
            slope_all[sel] = torch.sqrt(params[:, 0] ** 2 + params[:, 1] ** 2)  # gradient at the centre
            n_cons_all[sel] = consistent.sum(1)
            n_gated_all[sel] = gated.sum(1)
            # The consistent neighbours must form a surface, not a line: a thin
            # string of wrong-depth nodes along an edge is self-consistent but
            # has (almost) no spread across its minor axis.
            spread_all[sel] = minor_axis_variance(A[:, :, 0], A[:, :, 1], consistent.float()) >= (0.6 * step) ** 2
        return resid_all, n_cons_all, n_gated_all, spread_all, slope_all

    def _robust_fit(self, A: torch.Tensor, zn: torch.Tensor, w: torch.Tensor, tol: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched IRLS (Tukey biweight) fit of the local model.

        Returns the parameters (m, P) and the residuals (m, K); residuals of
        invalid neighbours are NaN. The residual scale is estimated per node from
        the MAD and floored at a fraction of the tolerance so quantised depths do
        not collapse it.
        """
        z0 = torch.nan_to_num(zn, nan=0.)
        weights = w
        c = 4.685
        scale_floor = tol / 4.
        params = resid = None
        for it in range(self.iterations + 1):
            params = weighted_fit(A, zn, weights)
            resid = z0 - (A @ params[:, :, None])[:, :, 0]
            resid = torch.where(w > 0, resid, torch.full_like(resid, float('nan')))
            if it == self.iterations:
                break
            abs_r = resid.abs()
            mad = torch.nanmedian(abs_r, dim=1).values * 1.4826
            scale = torch.clamp(mad, min=scale_floor)
            u = abs_r / (c * scale[:, None])
            tukey = torch.where(u < 1, (1 - u * u) ** 2, torch.zeros_like(u))
            weights = torch.nan_to_num(tukey, nan=0.) * w
        return params, resid

    # ── convenience ───────────────────────────────────────────────────────────

    def __call__(self, xyz: torch.Tensor, corr: Optional[torch.Tensor], x_vals: torch.Tensor, y_vals: torch.Tensor,
                 *extra: Optional[torch.Tensor]):
        """Filter ``xyz`` (and any aligned tensors); returns the filtered tensors."""
        keep = self.mask(xyz, x_vals, y_vals)
        out = [xyz[keep], None if corr is None else corr[keep]]
        out += [None if e is None else e[keep] for e in extra]
        return tuple(out)


# ── Seed depth map for the guided fine pass ──────────────────────────────────

def design_matrix(di: torch.Tensor, dj: torch.Tensor, step: float, order: int) -> torch.Tensor:
    """Stencil design matrix (1, K, P) of a local polynomial surface.

    Columns are ``[dx, dy, (dx^2, dx*dy, dy^2,) 1]``: the constant term is last so
    the model value at the centre is the last coefficient.
    """
    dx = (di.float() * step)[None, :]
    dy = (dj.float() * step)[None, :]
    cols = [dx, dy]
    if order >= 2:
        cols += [dx * dx, dx * dy, dy * dy]
    cols.append(torch.ones_like(dx))
    return torch.stack(cols, dim=-1)


def minor_axis_variance(dx: torch.Tensor, dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Smallest eigenvalue of the weighted XY covariance of the stencil offsets (m,)."""
    wsum = w.sum(1).clamp_min(1e-12)
    mx = (w * dx).sum(1) / wsum
    my = (w * dy).sum(1) / wsum
    ex, ey = dx - mx[:, None], dy - my[:, None]
    cxx = (w * ex * ex).sum(1) / wsum
    cyy = (w * ey * ey).sum(1) / wsum
    cxy = (w * ex * ey).sum(1) / wsum
    return 0.5 * (cxx + cyy) - torch.sqrt((0.5 * (cxx - cyy)) ** 2 + cxy ** 2)


def weighted_fit(A: torch.Tensor, zn: torch.Tensor, w: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Batched weighted least squares of the local model.

    ``A`` is the shared stencil design matrix (1, K, P), ``zn`` the neighbour
    depths (m, K) (NaN allowed where ``w`` is zero) and ``w`` the weights (m, K).
    Returns (m, P) parameters; rows with fewer than P weighted samples, or whose
    normal equations are singular, are NaN. ``ridge`` is a Tikhonov term
    relative to each diagonal entry of the normal matrix.
    """
    p = A.shape[-1]
    z0 = torch.nan_to_num(zn, nan=0., posinf=0., neginf=0.)
    w = torch.nan_to_num(w, nan=0., posinf=0., neginf=0.)
    enough = (w > 0).sum(1) >= p
    params = torch.full((zn.shape[0], p), float('nan'), dtype=torch.float32, device=zn.device)
    if not enough.any():
        return params
    # Normal equations in double precision: accumulating in float32 and adding
    # a tiny ridge before the cast loses the ridge entirely for entries of
    # magnitude 1e3-1e4, which made degenerate (collinear) neighbourhoods
    # reach the solver exactly singular. The ridge is relative to the matrix
    # scale, and solve_ex never raises: failed rows are reported as NaN.
    Ad = A.double()
    AtW = Ad * w[enough].double()[:, :, None]
    AtA = AtW.transpose(1, 2) @ Ad
    AtZ = (AtW * z0[enough].double()[:, :, None]).sum(1)
    # Per-column (Jacobi-scaled) ridge: every coefficient is shrunk by the same
    # relative amount, so the constant term is not biased by the large dx^4 entries.
    diag = AtA.diagonal(dim1=1, dim2=2)
    AtA = AtA + torch.diag_embed(ridge * diag + 1e-9)
    sol, info = torch.linalg.solve_ex(AtA, AtZ[:, :, None])
    sol = sol[:, :, 0]
    sol[(info != 0) | ~torch.isfinite(sol).all(1)] = float('nan')
    params[enough] = sol.float()
    return params


def build_seed_map(xyz: torch.Tensor, x_vals: torch.Tensor, y_vals: torch.Tensor,
                   fill_radius: float = 12., smooth_radius: Optional[float] = None, min_support: int = 6,
                   chunk: int = 32768) -> torch.Tensor:
    """Dense, smooth depth map ``z_seed[ix, iy]`` from sparse accepted lattice points.

    Every lattice node (occupied or not) receives the depth of a Gaussian-weighted
    local plane fitted through the accepted points within ``fill_radius`` mm.
    A first-order fit, unlike a plain weighted mean, interpolates holes and
    extrapolates past borders *along the local slope*, which keeps the seed close
    to tilted or curved surfaces. Nodes with fewer than ``min_support`` points in
    reach stay NaN, so callers know where no seed exists.

    ``smooth_radius`` is the Gaussian sigma of the weights (mm, default
    ``fill_radius / 2``); the effective smoothing is mild because only the
    plane's constant term is used.
    """
    if smooth_radius is None:
        smooth_radius = fill_radius / 2.
    device = xyz.device
    ix, iy, step = lattice_indices(xyz, x_vals, y_vals)
    nx, ny = len(x_vals), len(y_vals)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[ok], iy[ok]
    r_cells = max(1, int(fill_radius // step))
    pad = r_cells
    z_img = torch.full((nx + 2 * pad, ny + 2 * pad), float('nan'), dtype=torch.float32, device=device)
    z_img[ix + pad, iy + pad] = xyz[ok, 2].float()

    rng = torch.arange(-r_cells, r_cells + 1, device=device)
    di, dj = torch.meshgrid(rng, rng, indexing='ij')
    di, dj = di.ravel(), dj.ravel()
    dist = (di * di + dj * dj).double().sqrt() * step
    inside = dist <= fill_radius + 1e-9
    di, dj, dist = di[inside], dj[inside], dist[inside]
    sigma = max(float(smooth_radius), step)
    gauss = torch.exp(-0.5 * (dist / sigma) ** 2).float()[None, :]
    A = design_matrix(di, dj, step, order=1)

    seed = torch.full((nx * ny,), float('nan'), dtype=torch.float32, device=device)
    gx, gy = torch.meshgrid(torch.arange(nx, device=device), torch.arange(ny, device=device), indexing='ij')
    gx, gy = gx.ravel() + pad, gy.ravel() + pad
    for start in range(0, nx * ny, chunk):
        cx, cy = gx[start:start + chunk], gy[start:start + chunk]
        zn = z_img[cx[:, None] + di[None, :], cy[:, None] + dj[None, :]]
        valid = torch.isfinite(zn)
        w = gauss * valid.float()
        plane = weighted_fit(A, zn, w)
        # A plane cannot be extrapolated from (nearly) collinear support, e.g. a
        # single column of points at the reach limit: fall back to the weighted
        # mean when the support's minor-axis spread is below half a cell.
        minor = minor_axis_variance(A[:, :, 0], A[:, :, 1], w)
        mean_z = (w * torch.nan_to_num(zn, nan=0.)).sum(1) / w.sum(1).clamp_min(1e-12)
        value = torch.where(minor >= (0.5 * step) ** 2, plane[:, -1], mean_z)
        support = valid.sum(1) >= min_support
        seed[start:start + chunk] = torch.where(support, value, torch.full_like(value, float('nan')))
    return seed.view(nx, ny)


# ── Connected components on the lattice ──────────────────────────────────────

def lattice_components(xyz: torch.Tensor, x_vals: torch.Tensor, y_vals: torch.Tensor,
                       jump: float, reach: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Label depth-continuous connected components of lattice points.

    Two lattice points within ``reach`` mm of each other in XY belong to the
    same component when their depths differ by at most ``jump`` mm per mm of
    XY distance (a slope bound), so a cluster that merely touches the surface
    in XY at a different depth stays a separate component, while a sparsely
    populated surface (small gaps) stays connected.
    Returns ``(labels, sizes)`` with ``labels`` per input point (-1 for points
    off the lattice) and ``sizes[label]`` the component point count.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(xyz)
    labels = torch.full((n,), -1, dtype=torch.long, device=xyz.device)
    if n == 0:
        return labels, torch.zeros(0, dtype=torch.long, device=xyz.device)
    ix, iy, _ = lattice_indices(xyz, x_vals, y_vals)
    nx, ny = len(x_vals), len(y_vals)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    idx_img = torch.full((nx, ny), -1, dtype=torch.long, device=xyz.device)
    idx_img[ix[ok], iy[ok]] = torch.nonzero(ok).squeeze(1)
    z_img = torch.full((nx, ny), float('nan'), dtype=torch.float32, device=xyz.device)
    z_img[ix[ok], iy[ok]] = xyz[ok, 2].float()
    step = float(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 1.0
    r_cells = max(1, int(reach // step))
    offsets = [(di, dj) for di in range(0, r_cells + 1) for dj in range(-r_cells, r_cells + 1)
               if (di > 0 or dj > 0) and (di * di + dj * dj) * step * step <= reach * reach + 1e-9]
    rows, cols = [], []
    for di, dj in offsets:
        dist = float(np.hypot(di, dj)) * step
        a = idx_img[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)]
        b = idx_img[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj)]
        za = z_img[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)]
        zb = z_img[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj)]
        edge = (a >= 0) & (b >= 0) & ((za - zb).abs() <= jump * dist)
        rows.append(a[edge])
        cols.append(b[edge])
    rows = torch.cat(rows).cpu().numpy()
    cols = torch.cat(cols).cpu().numpy()
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n))
    _, lab = connected_components(graph, directed=False)
    lab = torch.as_tensor(lab, device=xyz.device, dtype=torch.long)
    lab[~ok] = -1
    sizes = torch.bincount(lab[ok], minlength=int(lab.max()) + 1) if ok.any() else torch.zeros(0, dtype=torch.long)
    return lab, sizes


def object_groups(xyz: torch.Tensor, distance: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group points into objects by 3-D proximity (voxel graph).

    Points are binned into voxels of ``distance / 3`` mm; occupied voxels
    whose centres are within ``distance`` mm are connected and the connected
    components of that graph are the object groups. A part separated from its
    object by a step or a small gap stays in the group, while other surfaces
    lit by the projector (background panels, fixtures) form separate groups.
    Returns ``(labels, sizes)`` per point.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = len(xyz)
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=xyz.device), torch.zeros(0, dtype=torch.long, device=xyz.device)
    pts = xyz.detach().double().cpu().numpy()
    voxel = max(distance / 3., 1e-3)
    keys = np.floor(pts / voxel).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    centres = (uniq + 0.5) * voxel
    pairs = cKDTree(centres).query_pairs(distance + voxel, output_type='ndarray')
    m = len(uniq)
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    _, vox_lab = connected_components(graph, directed=False)
    labels = vox_lab[inverse]
    sizes = np.bincount(labels)
    return (torch.as_tensor(labels, device=xyz.device, dtype=torch.long),
            torch.as_tensor(sizes, device=xyz.device, dtype=torch.long))