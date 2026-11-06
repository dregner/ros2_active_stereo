"""Reconstruction stages shared by the ROS scan workers."""
import torch

from fringe_consistency import light_sheet_mask
from laser_surface import depth_bands, refine_regions, select_depths, supported_components
from surface_filter import LatticeSurfaceFilter, build_seed_map


def _lattice(model, limits, xy_step, z_step):
    model.points3d(*limits, xy_step=xy_step, z_step=z_step)


def _fine_limits(points, limits, padding, depth_margin):
    lo, hi = points.amin(0).tolist(), points.amax(0).tolist()
    return tuple((max(a, low - pad), min(b, high + pad))
                 for (a, b), low, high, pad in zip(limits, lo, hi,
                                                   (padding, padding, depth_margin)))


def _phase_filter(model, xyz, score, threshold, modulation, radius, neighbors,
                  step, floor, log):
    keep = torch.isfinite(score) & (score < threshold)
    xyz, score = xyz[keep], score[keep]
    if not len(xyz):
        return xyz, score
    xyz, score, _ = model.std_mask_points(xyz, score, modulation, method='fringe')
    if not len(xyz):
        return xyz, score
    surface = LatticeSurfaceFilter(radius=radius, min_neighbors=neighbors, z_step=step)
    keep = surface.mask(xyz, model.x_vals, model.y_vals)
    log(str(surface.report))
    xyz, score = xyz[keep], score[keep]
    if len(xyz):
        keep, report = light_sheet_mask(model, xyz, floor_rad=floor)
        log(str(report))
        xyz, score = xyz[keep], score[keep]
    return xyz, score


@torch.inference_mode()
def reconstruct_fringe(model, limits, threshold, modulation, radius, neighbors,
                       fine_window=4., seed_radius=12., log=print):
    model.fringe_min_modulation = float(modulation)
    model.image_margin = 1
    _lattice(model, limits, xy_step=2., z_step=1.)
    xyz, score, index = model.process_segmented_z(1, 1, Nz_block_voxels=10, method='fringe')
    interior = (index > 0) & (index < len(model.z_vals) - 1)
    coarse, _ = _phase_filter(model, xyz[interior], score[interior], 2 * threshold,
                              modulation, radius, neighbors, 1., 2., log)
    if not len(coarse):
        return coarse, coarse
    fine_step = 0.25
    fine_limits = _fine_limits(coarse, limits, seed_radius, fine_window)
    _lattice(model, fine_limits, xy_step=1., z_step=fine_step)
    seed = build_seed_map(coarse, model.x_vals, model.y_vals, fill_radius=seed_radius,
                          smooth_radius=1., chunk=4096)
    xyz, score, offset, _ = model.process_guided_z(
        1, 1, seed, half_window=fine_window, z_step=fine_step, method='fringe',
        z_limits=limits[2])
    interior = offset.abs() < max(1, round(fine_window / fine_step))
    fine, _ = _phase_filter(model, xyz[interior], score[interior], threshold,
                            modulation, radius, neighbors, fine_step, 0.3, log)
    return coarse, fine


def _laser_filter(model, xyz, score, threshold, texture, crop):
    keep = torch.isfinite(score) & (score > threshold)
    xyz, score = xyz[keep], score[keep]
    if len(xyz):
        xyz, score, _ = model.mask_uv_points(xyz, score, crop_factor=crop)
    if len(xyz):
        xyz, score, _ = model.std_mask_points(xyz, score, texture, method='laser')
    return xyz, score


def _laser_support(xyz, radius, min_points):
    keep, components = supported_components(xyz.cpu().numpy(), radius=radius,
                                             min_points=min_points)
    return xyz[torch.as_tensor(keep, device=xyz.device)], components


@torch.inference_mode()
def reconstruct_laser(model, limits, kernel, stride, threshold, texture, crop,
                      support_radius=5., min_points=32, depth_margin=6.,
                      split_tolerance=1., tile_size=32., log=print):
    validate_split = split_tolerance >= 0
    if validate_split and model.left_images.shape[0] < 4:
        raise ValueError('Laser frame-split validation needs at least four stereo pairs')
    model.image_margin = 2  # bicubic interpolation support
    _lattice(model, limits, xy_step=2., z_step=2.)
    xyz, score, index = model.process_segmented_z(kernel, kernel, stride=stride,
                                                  Nz_block_voxels=10, method='laser')
    interior = (index > 0) & (index < len(model.z_vals) - 1)
    xyz, _ = _laser_filter(model, xyz[interior], score[interior], threshold, texture, crop)
    coarse, components = _laser_support(xyz, support_radius, min_points)
    bands = depth_bands(components, margin=depth_margin, limits=limits[2])
    log(f'Laser coarse support: {len(coarse)} points, {len(components)} components, {len(bands)} depth bands')
    if not bands:
        return coarse, coarse[:0]
    fine_limits = _fine_limits(coarse, limits, kernel // 2 + 6., depth_margin)
    _lattice(model, fine_limits, xy_step=1., z_step=0.1)
    keep_z = select_depths(model.z_vals.cpu().numpy(), bands)
    model.z_vals = model.z_vals[torch.as_tensor(keep_z, device=model.device)]
    xyz, score, stats = refine_regions(
        model, components, kernel, xy_margin=6., z_margin=depth_margin,
        validate_split=validate_split, seed_points=coarse.cpu().numpy(), tile_size=tile_size)
    log(f'Laser refinement: {stats}')
    if validate_split and len(xyz):
        keep = model.laser_diagnostics['split_depth_error'] <= split_tolerance
        xyz, score = xyz[keep], score[keep]
    xyz, _ = _laser_filter(model, xyz, score, threshold, texture, crop)
    fine, _ = _laser_support(xyz, support_radius, min_points)
    return coarse, fine
