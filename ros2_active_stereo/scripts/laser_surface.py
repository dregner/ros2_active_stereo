"""Spatial support and disjoint depth bands for laser inverse triangulation."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def supported_components(xyz, radius=5., min_points=32, neighbors=16):
    """Keep all substantial connected components in a bounded 3D k-NN graph.

    Radius is in calibration units (mm). This rejects isolated depth trails,
    not every possible smooth false surface. Small or sparse real surfaces can
    be rejected too. Bounded neighbors keep graph storage linear in point count.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must have shape (N, 3)')
    if radius <= 0 or min_points < 1 or neighbors < 2:
        raise ValueError('radius, min_points and neighbors must be positive (neighbors >= 2)')
    mask = np.zeros(len(xyz), dtype=bool)
    finite = np.isfinite(xyz).all(1)
    points = xyz[finite].astype(np.float64, copy=False)
    if not len(points):
        return mask, []
    distance, index = cKDTree(points).query(points, k=neighbors, distance_upper_bound=radius)
    row = np.broadcast_to(np.arange(len(points))[:, None], index.shape)
    valid = (index < len(points)) & (index != row)
    graph = csr_matrix((np.ones(valid.sum(), dtype=np.uint8), (row[valid], index[valid])),
                       shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    kept = counts[labels] >= min_points
    mask[finite] = kept
    # Group once, rather than rescanning all points for every small component.
    order = np.argsort(labels)
    offsets = np.r_[0, np.cumsum(counts)]
    components = []
    for label in np.flatnonzero(counts >= min_points):
        p = points[order[offsets[label]:offsets[label+1]]]
        components.append(dict(points=int(len(p)), xyz_min=p.min(0).tolist(), xyz_max=p.max(0).tolist()))
    return mask, sorted(components, key=lambda c: c['points'], reverse=True)


def depth_bands(components, margin=6., limits=(-800., 800.)):
    """Merge padded component intervals, preserving multiple separated surfaces."""
    if margin < 0 or limits[0] >= limits[1]:
        raise ValueError('invalid depth margin or limits')
    intervals = sorted((max(limits[0], c['xyz_min'][2]-margin),
                        min(limits[1], c['xyz_max'][2]+margin)) for c in components)
    merged = []
    for lo, hi in intervals:
        if lo > hi:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([float(lo), float(hi)])
    return merged


def select_depths(z_values, bands):
    """Mask a regular fine lattice without changing its depth sampling phase."""
    keep = np.zeros(len(z_values), dtype=bool)
    for lo, hi in bands:
        keep |= (z_values >= lo) & (z_values <= hi)
    return keep


def refinement_regions(components, xy_margin=6., z_margin=6.):
    """Merge overlapping padded 3D boxes; preserve independent object regions."""
    pending = [(np.asarray(c['xyz_min'])-np.array([xy_margin, xy_margin, z_margin]),
                np.asarray(c['xyz_max'])+np.array([xy_margin, xy_margin, z_margin])) for c in components]
    merged = []
    while pending:
        lo, hi = pending.pop()
        changed = True
        while changed:
            changed = False
            for i, (other_lo, other_hi) in enumerate(merged):
                if np.all(lo <= other_hi) and np.all(other_lo <= hi):
                    lo, hi = np.minimum(lo, other_lo), np.maximum(hi, other_hi)
                    merged.pop(i)
                    changed = True
                    break
        merged.append((lo, hi))
    return merged


def refine_regions(model, components, kernel, xy_margin=6., z_margin=6., validate_split=True,
                   seed_points=None, tile_size=0.):
    """Reuse the fine lattice, scan only component boxes, and merge XY winners.

    If several depth regions cover an XY site, both the full-frame winner and
    each split winner are selected across every region covering that site.
    The original global lattice is restored even when processing raises.
    """
    import torch
    original = model.x_vals, model.y_vals, model.z_vals
    xs, ys, zs = original
    pad = kernel // 2
    step = float(xs[1]-xs[0]) if len(xs)>1 else 1.
    records = []
    voxel_count = 0
    regions = refinement_regions(components, xy_margin, z_margin)
    if seed_points is not None and tile_size > 0:
        # A regional range uses every supported seed in a spatial neighborhood,
        # rather than the single coarse winner that can alias a narrow peak.
        # Keep separated depth populations in each tile as separate hypotheses.
        p = np.asarray(seed_points)
        cells = np.floor(p[:,:2]/tile_size).astype(np.int64)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        order = np.argsort(inverse)
        counts = np.bincount(inverse)
        offsets = np.r_[0,np.cumsum(counts)]
        regions = []
        margin = np.array([xy_margin,xy_margin,z_margin])
        for i in range(len(counts)):
            tile = p[order[offsets[i]:offsets[i+1]]]
            tile = tile[np.argsort(tile[:,2])]
            cuts = np.r_[0, np.flatnonzero(np.diff(tile[:,2])>2*z_margin)+1, len(tile)]
            for start, end in zip(cuts[:-1],cuts[1:]):
                group = tile[start:end]
                regions.append((group.min(0)-margin,group.max(0)+margin))
    try:
        for lo, hi in regions:
            x_mask = (xs >= lo[0]-pad*step) & (xs <= hi[0]+pad*step)
            y_mask = (ys >= lo[1]-pad*step) & (ys <= hi[1]+pad*step)
            z_mask = (zs >= lo[2]) & (zs <= hi[2])
            model.x_vals, model.y_vals, model.z_vals = xs[x_mask], ys[y_mask], zs[z_mask]
            if min(len(model.x_vals),len(model.y_vals)) < kernel or not len(model.z_vals):
                continue
            voxel_count += len(model.x_vals)*len(model.y_vals)*len(model.z_vals)
            xyz, corr, depth_index = model.process_segmented_z(kernel, kernel, Nz_block_voxels=10,
                                                      method='laser', validate_split=validate_split)
            diag = model.laser_diagnostics
            # A region-edge winner is not an interior depth peak.
            interior = (depth_index > 0) & (depth_index < len(model.z_vals)-1)
            records.append((xyz, torch.where(interior, corr, float('-inf')),
                            diag['split_z'] if validate_split else None,
                            diag['split_score'] if validate_split else None))
    finally:
        model.x_vals, model.y_vals, model.z_vals = original
    stats = dict(refinement_regions=len(regions), fine_projected_voxels=voxel_count,
                 rectangular_band_voxels=len(xs)*len(ys)*len(zs))
    if not records:
        model.laser_diagnostics = None
        return torch.empty((0,3),device=model.device), torch.empty(0,device=model.device), stats
    xyz = torch.cat([r[0] for r in records])
    corr = torch.cat([r[1] for r in records])
    # CPU grouping is small relative to projecting the 3D volume and avoids
    # depending on a particular CUDA scatter-reduce implementation/version.
    xy_index = torch.round((xyz[:,:2]-torch.stack([xs[0],ys[0]]))/step).long().cpu().numpy()
    keys = xy_index[:,0]*len(ys)+xy_index[:,1]
    scores = corr.cpu().numpy()
    order = np.lexsort((xyz[:,2].cpu().numpy(), -scores, keys))
    _, starts = np.unique(keys[order], return_index=True)
    chosen = order[starts]
    if validate_split:
        split_z = torch.cat([r[2] for r in records]).cpu().numpy()
        split_score = torch.cat([r[3] for r in records]).cpu().numpy()
        best_z, best_score = [], []
        for split in range(2):
            split_order = np.lexsort((split_z[:,split], -split_score[:,split], keys))
            _, split_starts = np.unique(keys[split_order], return_index=True)
            best_indices = split_order[split_starts]
            best_z.append(split_z[best_indices,split])
            best_score.append(split_score[best_indices,split])
        valid = np.isfinite(best_score[0]) & np.isfinite(best_score[1]) & (best_score[0]>model.epsilon) & (best_score[1]>model.epsilon)
        error = np.where(valid, abs(best_z[0]-best_z[1]), np.inf)
        model.laser_diagnostics = {'split_depth_error': torch.as_tensor(error,device=model.device)}
    else:
        model.laser_diagnostics = None
    chosen = torch.as_tensor(chosen,device=model.device)
    return xyz[chosen], corr[chosen], stats
