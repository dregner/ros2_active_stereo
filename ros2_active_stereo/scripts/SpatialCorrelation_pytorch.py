import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import cv2
from typing import Tuple

class PyTorchStereoCorrel(nn.Module):
    def __init__(self, yaml_file):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch a ser executado no dispositivo: {self.device}")

        self.left_images: torch.Tensor | None = None
        self.right_images: torch.Tensor | None = None
        self.grid: torch.Tensor | None = None
        self.x_vals: torch.Tensor | None = None
        self.y_vals: torch.Tensor | None = None
        self.z_vals: torch.Tensor | None = None

        # Single source of truth for near-zero guard — used by ZNCC and depth check
        self.epsilon = 1e-8
        # Pixels a projected centre must keep from the image border so that its
        # whole correlation patch (plus interpolation support) is inside the
        # image. grid_sample pads with replicated edge pixels, which correlate
        # spuriously between the two views when both patches hang over the edge.
        self.image_margin = 0
        self.max_block_voxels = 500_000
        self.laser_diagnostics = None
        self._images_undistorted = False
        self.fringe_min_modulation = 10.0
        self.fringe_max_phase_jump = np.pi
        self.camera_params = self.read_yaml_file(yaml_file)


    def run_grid_diagnostics(self, limits: dict, steps: dict):
        """
        Executa um diagnóstico de sensibilidade do grid, verificando como os passos
        no espaço 3D se traduzem em movimento de pixels na imagem.
        """
        print("\n[Diagnóstico] Calculando a sensibilidade do grid para os parâmetros atuais...")

        x_mid = limits['x'][0] + (limits['x'][1] - limits['x'][0]) / 2
        y_mid = limits['y'][0] + (limits['y'][1] - limits['y'][0]) / 2
        z_mid = limits['z'][0] + (limits['z'][1] - limits['z'][0]) / 2

        # #1: float32 — sufficient precision for pixel-level projection
        p_center = torch.tensor([[x_mid, y_mid, z_mid]], dtype=torch.float32, device=self.device)
        p_step_x = torch.tensor([[x_mid + steps['xy'], y_mid, z_mid]], dtype=torch.float32, device=self.device)
        p_step_z = torch.tensor([[x_mid, y_mid, z_mid + steps['z']]], dtype=torch.float32, device=self.device)

        uv_center = self.transform_gcs2ccs(p_center, 'left')
        uv_step_x = self.transform_gcs2ccs(p_step_x, 'left')
        uv_step_z = self.transform_gcs2ccs(p_step_z, 'left')

        if uv_center.min() > 0 and uv_step_x.min() > 0 and uv_step_z.min() > 0:
            dist_pix_x = torch.linalg.norm(uv_step_x - uv_center).item()
            dist_pix_z = torch.linalg.norm(uv_step_z - uv_center).item()

            print(f"  > Passo XY de {steps['xy']:.1f} mm equivale a um deslocamento de {dist_pix_x:.3f} pixels na imagem.")
            print(f"  > Passo Z de {steps['z']:.1f} mm equivale a um deslocamento de {dist_pix_z:.3f} pixels na imagem.")
        print("-" * 20)

    def read_yaml_file(self, yaml_file: str) -> dict:
        """Lê os parâmetros de calibração de um arquivo YAML e os retorna."""
        with open(yaml_file) as file:
            params = yaml.safe_load(file)

        camera_params = {
            'left': {},
            'right': {},
            'stereo': {}
        }

        # #1: float32 — GPU tensor cores operate at fp32; no accuracy loss for calibration maths
        for cam in ['left', 'right']:
            camera_params[cam]['kk'] = torch.tensor(params[f'camera_matrix_{cam}'], dtype=torch.float32, device=self.device)
            camera_params[cam]['kc'] = torch.tensor(params[f'dist_coeffs_{cam}'], dtype=torch.float32, device=self.device)
            camera_params[cam]['r'] = torch.tensor(params[f'rot_matrix_{cam}'], dtype=torch.float32, device=self.device)
            camera_params[cam]['t'] = torch.tensor(params[f't_{cam}'], dtype=torch.float32, device=self.device).view(3, 1)

        camera_params['stereo']['R'] = torch.tensor(params['R'], dtype=torch.float32, device=self.device)
        camera_params['stereo']['T'] = torch.tensor(params['T'], dtype=torch.float32, device=self.device).view(3, 1)

        return camera_params

    def convert_images(self, left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True, apply_blur=False, blur=1, tile=8, climp=2.0):
        if len(left_imgs_cpu) != len(right_imgs_cpu) or not len(left_imgs_cpu):
            raise ValueError('Expected nonempty stereo image stacks with matching channel counts')
        shapes = {np.shape(img) for stack in (left_imgs_cpu, right_imgs_cpu) for img in stack}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2 or min(next(iter(shapes))) < 2:
            raise ValueError('Expected matching 2D stereo image dimensions of at least 2x2')
        self._images_undistorted = undist
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=climp, tileGridSize=(tile, tile))

        def process_image(img, cam_params):
            if apply_clahe:
                img = clahe.apply(img)
            if undist:
                kk = cam_params['kk'].cpu().numpy()
                kc = cam_params['kc'].cpu().numpy()
                img = cv2.undistort(img, kk, kc)
            if apply_blur:
                img = cv2.blur(img, (blur, blur))
            return img

        processed_left_imgs = [process_image(img, self.camera_params['left']) for img in left_imgs_cpu]
        processed_right_imgs = [process_image(img, self.camera_params['right']) for img in right_imgs_cpu]

        # #1: float32, #9: pin_memory() enables async DMA (host→device overlap via non_blocking=True)
        def to_device(images):
            tensor = torch.from_numpy(np.stack(images, axis=0))
            if self.device.type == 'cuda':
                tensor = tensor.pin_memory()
            return tensor.to(self.device, dtype=torch.float32,
                             non_blocking=self.device.type == 'cuda')

        self.left_images = to_device(processed_left_imgs)
        self.right_images = to_device(processed_right_imgs)
        # Background temporal-noise level of the (equalised) frames: the 20th
        # percentile of the per-pixel std over time is dominated by unlit
        # pixels. Used to make the texture mask exposure-independent.
        if self.left_images.shape[0] > 1:
            sub = self.left_images[:, ::4, ::4].std(0).flatten()
            self.noise_level = float(torch.quantile(sub, 0.2))
        else:
            self.noise_level = 0.0

    def remove_img_distortion(self, img, cam_name):
        """Remove a distorção de uma imagem usando os parâmetros da câmera."""
        k = self.camera_params[cam_name]['kk'].cpu().numpy()
        kc = self.camera_params[cam_name]['kc'].cpu().numpy()
        return cv2.undistort(img, k, kc)

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        def axis(bounds, step):
            lo, hi = map(float, bounds)
            if not np.isfinite((lo, hi, step)).all() or step <= 0 or hi < lo:
                raise ValueError(f'Invalid lattice bounds {bounds} or step {step}')
            count = int(np.floor((hi - lo) / step + 1e-6)) + 1
            return lo + torch.arange(count, dtype=torch.float32, device=self.device) * step

        self.x_vals = axis(x_lim, xy_step)
        self.y_vals = axis(y_lim, xy_step)
        self.z_vals = axis(z_lim, z_step)
        self.grid = None

    def transform_gcs2ccs(self, points_3d, cam_name, image_shape=None, *, distort=None):
        if distort is None:
            distort = not self._images_undistorted
        k, r, t = self.camera_params[cam_name]['kk'], self.camera_params[cam_name]['r'], self.camera_params[cam_name]['t']

        num_points = points_3d.shape[0]
        if num_points == 0:
            if image_shape is not None:
                return torch.empty((0, 2), device=self.device), torch.empty((0,), dtype=torch.bool, device=self.device)
            else:
                return torch.empty((0, 2), device=self.device)

        ones = torch.ones((num_points, 1), device=self.device, dtype=points_3d.dtype)
        xyz_gcs_1 = torch.cat([points_3d, ones], dim=1)
        rt_matrix = torch.cat([r, t], dim=1)
        # #3: removed dead code — the 4×4 homogeneous matrix was computed but never used
        # #10: cast to r.dtype (float32) instead of hardcoded float64
        xyz_ccs = torch.matmul(rt_matrix, xyz_gcs_1.T.to(r.dtype)).T

        zc = xyz_ccs[:, 2]
        valid_mask = zc > self.epsilon  # #13: use self.epsilon consistently
        uv_points = torch.full((num_points, 2), -1.0, device=self.device, dtype=r.dtype)

        if torch.any(valid_mask):
            xn = xyz_ccs[valid_mask, 0] / zc[valid_mask]
            yn = xyz_ccs[valid_mask, 1] / zc[valid_mask]
            if distort == True:
                r2 = xn**2 + yn**2
                coeffs = self.camera_params[cam_name]['kc'].flatten()
                k1, k2, p1, p2, k3 = F.pad(coeffs, (0, max(0, 5 - coeffs.numel())))[:5]
                radial_distortion = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
                x_distorted = xn * radial_distortion + 2 * p1 * xn * yn + p2 *(r2 + 2 * xn**2)
                y_distorted = yn * radial_distortion + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn
                xn, yn = x_distorted, y_distorted
            xyz_ccs = torch.matmul(k, torch.stack([xn, yn, torch.ones_like(xn)], dim=0))

            uv_points[valid_mask] = xyz_ccs[:2, :].T

        if image_shape is not None:
            H, W = image_shape
            m = float(getattr(self, 'image_margin', 0))
            # Mask for points inside image boundaries (minus the patch margin)
            inside_mask = (
                (uv_points[:, 0] >= m) & (uv_points[:, 0] < W - m) &
                (uv_points[:, 1] >= m) & (uv_points[:, 1] < H - m)
            )
            return uv_points, inside_mask
        else:
            return uv_points

    def interpolate_images(self, images, uv_points, uv_mask=None, method="correl"):
        if method == "fringe":
            return self.interpolate_phase(images, uv_points, uv_mask)
        if uv_points.numel() == 0:
            return torch.empty((0, images.shape[0]), device=self.device)
        T, H, W = images.shape
        N = uv_points.shape[0]

        u_norm = (uv_points[:, 0] / (W - 1)) * 2 - 1
        v_norm = (uv_points[:, 1] / (H - 1)) * 2 - 1

        grid = torch.stack([u_norm, v_norm], dim=1).view(1, N, 1, 2)
        images_batch = images.unsqueeze(0)

        # #2: padding_mode='border' extrapolates edge pixels instead of zero-padding,
        # preventing zeros from contaminating ZNCC patch windows near image boundaries.
        # .clamp(0, 255) guards against bicubic overshoot (Runge's phenomenon).
        interpolated = F.grid_sample(images_batch.float(), grid.float(),
                                     mode='bicubic', padding_mode='border',
                                     align_corners=True)
        interpolated = interpolated.clamp(0, 255).view(T, N).T

        if uv_mask is not None:
            interpolated[~uv_mask] = torch.nan

        return interpolated

    def interpolate_phase(self, images, uv_points, uv_mask=None):
        """Bilinear absolute phase sampling, without intensity clipping.

        Never interpolate across an unwrap jump larger than pi. The optional
        modulation floor applies to every contributing pixel before depth
        selection, so a low-quality minimum cannot hide a valid alternative.
        Channel 0 is absolute phase in radians; channel 1 is modulation.
        """
        T, H, W = images.shape
        if uv_points.numel() == 0:
            return images.new_empty((0, T))
        u, v = uv_points.unbind(1)
        valid = torch.isfinite(uv_points).all(1) & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1)
        if uv_mask is not None:
            valid &= uv_mask
        safe = torch.nan_to_num(uv_points, nan=0., posinf=0., neginf=0.)
        u = safe[:, 0].clamp(0, W-1)
        v = safe[:, 1].clamp(0, H-1)
        x0, y0 = u.floor().long(), v.floor().long()
        x1, y1 = (x0+1).clamp_max(W-1), (y0+1).clamp_max(H-1)
        du, dv = u-x0, v-y0
        weights = torch.stack([(1-du)*(1-dv), du*(1-dv), (1-du)*dv, du*dv], 1)
        samples = torch.stack([images[:, y0, x0].T, images[:, y0, x1].T,
                               images[:, y1, x0].T, images[:, y1, x1].T], 1)
        contributing = weights > 0
        phase = samples[:, :, 0]
        lo = phase.masked_fill(~contributing, torch.inf).amin(1)
        hi = phase.masked_fill(~contributing, -torch.inf).amax(1)
        valid &= (hi-lo <= getattr(self, 'fringe_max_phase_jump', np.pi))
        valid &= (torch.isfinite(samples).all(2) | ~contributing).all(1)
        floor = getattr(self, 'fringe_min_modulation', 0.)
        if T > 1 and floor > 0:
            valid &= ((samples[:, :, 1] > floor) | ~contributing).all(1)
        result = (samples.masked_fill(~contributing[:, :, None], 0.) * weights[:, :, None]).sum(1)
        return result.masked_fill(~valid[:, None], torch.nan)

    def phase_map_difference(self, L_patches, R_patches):
        #Compute absolute difference between left and right patches
        return torch.abs(L_patches - R_patches)[:,0]

    def _unfold_and_zncc(self, interp_L: torch.Tensor, interp_R: torch.Tensor,
                         cur_Nz: int, T: int, Nx: int, Ny: int, Kx: int, Ky: int,
                         stride: int, method: str) -> torch.Tensor:
        """
        Ultra-fast batched Box-Sum equivalent of F.unfold + ZNCC.
        Maintains 100% mathematical parity with F.unfold, including NaN boundary poisoning.
        Eliminates the massive VRAM spikes and Python loops.
        """
        L_4d = interp_L.reshape(cur_Nz, Nx, Ny, T).permute(0, 3, 1, 2) # (cur_Nz, T, Nx, Ny)
        R_4d = interp_R.reshape(cur_Nz, Nx, Ny, T).permute(0, 3, 1, 2)

        pad_x, pad_y = Kx // 2, Ky // 2

        if method == 'fringe':
            # Fringe samples the reported kernel center at phase channel 0.
            L_center = L_4d[:, 0, pad_x:Nx-pad_x:stride, pad_y:Ny-pad_y:stride] # (cur_Nz, Nc_x, Nc_y)
            R_center = R_4d[:, 0, pad_x:Nx-pad_x:stride, pad_y:Ny-pad_y:stride]

            diff = torch.abs(L_center - R_center)
            out = diff.reshape(cur_Nz, -1).T # (Nc, cur_Nz)
            return out.masked_fill(~torch.isfinite(out), float('inf'))

        elif method == 'census':
            left = F.unfold(L_4d, (Kx, Ky), stride=stride).transpose(1, 2)
            right = F.unfold(R_4d, (Kx, Ky), stride=stride).transpose(1, 2)
            scores = self.census_similarity(left.flatten(0, 1), right.flatten(0, 1), Kx, Ky)
            return scores.reshape(cur_Nz, -1).T
        else:
            # Track NaNs to ensure patches with ANY out-of-bounds pixels are poisoned, matching F.unfold exactly
            is_nan = (torch.isnan(L_4d).any(dim=1, keepdim=True) | torch.isnan(R_4d).any(dim=1, keepdim=True)).float()
            has_nan = F.max_pool2d(is_nan, kernel_size=(Kx, Ky), stride=(stride, stride)) == 1.0

            L = torch.nan_to_num(L_4d, nan=0.0)
            R = torch.nan_to_num(R_4d, nan=0.0)

            L_sum = L.sum(dim=1, keepdim=True)
            R_sum = R.sum(dim=1, keepdim=True)
            LL_sum = (L * L).sum(dim=1, keepdim=True)
            RR_sum = (R * R).sum(dim=1, keepdim=True)
            LR_sum = (L * R).sum(dim=1, keepdim=True)

            K_area = Kx * Ky
            sL = F.avg_pool2d(L_sum, kernel_size=(Kx, Ky), stride=(stride, stride), padding=0) * K_area
            sR = F.avg_pool2d(R_sum, kernel_size=(Kx, Ky), stride=(stride, stride), padding=0) * K_area
            sLL = F.avg_pool2d(LL_sum, kernel_size=(Kx, Ky), stride=(stride, stride), padding=0) * K_area
            sRR = F.avg_pool2d(RR_sum, kernel_size=(Kx, Ky), stride=(stride, stride), padding=0) * K_area
            sLR = F.avg_pool2d(LR_sum, kernel_size=(Kx, Ky), stride=(stride, stride), padding=0) * K_area

            N_pixels = T * K_area
            num = sLR - (sL * sR) / N_pixels
            den_L = sLL - (sL * sL) / N_pixels
            den_R = sRR - (sR * sR) / N_pixels

            corr = (num / (den_L.clamp_min(0) * den_R.clamp_min(0)).sqrt().clamp(min=self.epsilon)).clamp(-1, 1)
            corr[(den_L <= self.epsilon) | (den_R <= self.epsilon)] = float('-inf')

            corr[has_nan] = float('-inf')

            out = corr.squeeze(1).reshape(cur_Nz, -1).T # (Nc, cur_Nz)
            return out.masked_fill(~torch.isfinite(out), float('-inf'))

    def zncc_correlation(self, L_patches, R_patches):
        """Per-point ZNCC for a single Z slice (used only by fine pass surface-following). L/R_patches: (Nc, patch_dim)."""
        N = L_patches.shape[1]
        sum_L = L_patches.sum(dim=1)
        sum_R = R_patches.sum(dim=1)

        num = (L_patches * R_patches).sum(dim=1) - (sum_L * sum_R) / N
        den_L = (L_patches * L_patches).sum(dim=1) - (sum_L * sum_L) / N
        den_R = (R_patches * R_patches).sum(dim=1) - (sum_R * sum_R) / N

        denominator = (den_L * den_R).sqrt().clamp(min=self.epsilon)
        return num / denominator

    def census_similarity(self, L_patches, R_patches, Kx, Ky):
        """Census on reprojected spatial patches, independently for each frame.

        Return 1 - normalized Hamming distance (higher is better). All
        samples must be finite; flat patches have no matching evidence.
        Bilinear sampling happens before comparison, never on binary codes.
        """
        if Kx < 1 or Ky < 1 or Kx % 2 == 0 or Ky % 2 == 0 or Kx * Ky == 1:
            raise ValueError("Census requires odd spatial windows with at least two samples")
        area = Kx * Ky
        left = L_patches.reshape(L_patches.shape[0], -1, area)
        right = R_patches.reshape(R_patches.shape[0], -1, area)
        center = (Kx // 2) * Ky + Ky // 2
        neighbors = torch.arange(area, device=left.device) != center
        left_bits = left[:, :, neighbors] < left[:, :, center:center + 1]
        right_bits = right[:, :, neighbors] < right[:, :, center:center + 1]
        score = 1.0 - (left_bits != right_bits).to(left.dtype).mean(dim=(1, 2))
        valid = torch.isfinite(left).all(dim=(1, 2)) & torch.isfinite(right).all(dim=(1, 2))
        textured = ((left.amax(dim=2) > left.amin(dim=2)) &
                    (right.amax(dim=2) > right.amin(dim=2))).any(dim=1)
        return score.masked_fill(~(valid & textured), float('-inf'))

    def _scan_block_size(self, nx, ny, requested):
        if requested <= 0 or self.max_block_voxels <= 0:
            raise ValueError('Scan block sizes must be positive')
        return min(requested, max(1, self.max_block_voxels // max(1, nx * ny)))

    @staticmethod
    def _validate_scan(Kx, Ky, stride, method):
        if method not in ('correl', 'fringe', 'laser', 'census'):
            raise ValueError(f'Unknown matching method: {method}')
        if min(Kx, Ky, stride) < 1 or Kx % 2 == 0 or Ky % 2 == 0:
            raise ValueError('Scan requires odd positive patch dimensions and a positive stride')
        if method == 'census' and Kx * Ky == 1:
            raise ValueError('Census requires more than one spatial sample')

    def process_segmented_z(self, Kx, Ky, stride=1, Nz_block_voxels=40, method='correl', validate_split=False):
        self._validate_scan(Kx, Ky, stride, method)
        Nx = len(self.x_vals)
        Ny = len(self.y_vals)
        Nz_block_voxels = self._scan_block_size(Nx, Ny, Nz_block_voxels)
        Nz_total = len(self.z_vals)
        T = self.left_images.shape[0]
        if validate_split and (method != 'laser' or T < 4):
            raise ValueError('Frame-split validation requires laser mode and at least four frames')
        self.laser_diagnostics = None

        if Nx < Kx or Ny < Ky or not Nz_total:
            empty = torch.empty(0, device=self.device)
            return empty.new_empty((0, 3)), empty, empty.long()

        pad_x, pad_y = Kx // 2, Ky // 2
        ix_centers = torch.arange(pad_x, Nx - pad_x, stride, device=self.device)
        iy_centers = torch.arange(pad_y, Ny - pad_y, stride, device=self.device)

        if len(ix_centers) == 0 or len(iy_centers) == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0)

        IX_centers, IY_centers = torch.meshgrid(ix_centers, iy_centers, indexing='ij')
        IX_centers, IY_centers = IX_centers.ravel(), IY_centers.ravel()
        Nc_for_xy_plane = IX_centers.shape[0]

        # Pre-compute the flat XY grid — constant across all Z-blocks.
        # Shape: (Nx*Ny, 2). For 800×800 this is ~5 MB, always fits in memory.
        # indexing='ij' matches the original meshgrid ordering so downstream
        # reshapes (.view(Nz, Nx, Ny, T)) remain correct.
        X_2d, Y_2d = torch.meshgrid(self.x_vals, self.y_vals, indexing='ij')
        XY_flat = torch.stack([X_2d.ravel(), Y_2d.ravel()], dim=1)  # (Nx*Ny, 2)
        del X_2d, Y_2d

        # Running best-score / best-index: O(Nc) — replaces the old
        # corr_map_overall_z of shape (Nc, Nz_total) which was ~20 GB.
        if method == 'fringe':
            best_score = torch.full((Nc_for_xy_plane,), float('inf'),
                                    device=self.device, dtype=torch.float32)
        else:
            best_score = torch.full((Nc_for_xy_plane,), float('-inf'),
                                    device=self.device, dtype=torch.float32)
        best_z_idx = torch.zeros(Nc_for_xy_plane, dtype=torch.long, device=self.device)
        if validate_split:
            split_scores = [torch.full_like(best_score, float('-inf')) for _ in range(2)]
            split_indices = [torch.zeros_like(best_z_idx) for _ in range(2)]

        for z0_idx in range(0, Nz_total, Nz_block_voxels):
            z1_idx = min(z0_idx + Nz_block_voxels, Nz_total)
            z_block = self.z_vals[z0_idx:z1_idx]          # (current_Nz,)
            current_Nz_in_slice = z1_idx - z0_idx

            # Build 3-D points for this Z-block on-the-fly.
            # Ordering: for each z in z_block, iterate over all (x, y) in XY_flat.
            # Result shape: (current_Nz * Nx*Ny, 3) — identical ordering to the
            # original grid_slice.permute(2,0,1,3).reshape(current_Nz, Nx*Ny, 3).reshape(-1, 3).
            Z_col = z_block.repeat_interleave(Nx * Ny).unsqueeze(1)              # (current_Nz*Nx*Ny, 1)
            # #8: expand() is a zero-copy broadcast; avoids a large allocation each block
            XY_expanded = XY_flat.unsqueeze(0).expand(current_Nz_in_slice, -1, -1).reshape(-1, 2)
            grid_flat = torch.cat([XY_expanded, Z_col], dim=1)                   # (current_Nz*Nx*Ny, 3)
            del XY_expanded, Z_col

            uv_left,  uv_left_mask  = self.transform_gcs2ccs(grid_flat, 'left',  image_shape=self.left_images.shape[1:])
            uv_right, uv_right_mask = self.transform_gcs2ccs(grid_flat, 'right', image_shape=self.right_images.shape[1:])
            del grid_flat

            interp_L = self.interpolate_images(self.left_images,  uv_left,  uv_mask=uv_left_mask, method=method)
            interp_R = self.interpolate_images(self.right_images, uv_right, uv_mask=uv_right_mask, method=method)
            del uv_left, uv_right, uv_left_mask, uv_right_mask
            # #7: empty_cache() removed from hot path — PyTorch allocator reuses memory automatically

            # #5: Vectorized correlation across ALL Z slices in this block using ultra-fast Box-Sums
            # Bypasses F.unfold completely, keeping VRAM minimal and performance maxed
            corr_all = self._unfold_and_zncc(interp_L, interp_R, current_Nz_in_slice, T, Nx, Ny, Kx, Ky, stride, method)
            if validate_split:
                # Reuse projections/interpolation; only reductions are repeated.
                for split in range(2):
                    l_split, r_split = interp_L[:, split::2], interp_R[:, split::2]
                    score_split = self._unfold_and_zncc(l_split, r_split, current_Nz_in_slice,
                                                      l_split.shape[1], Nx, Ny, Kx, Ky, stride, method)
                    value, index = score_split.max(1)
                    update = value > split_scores[split]
                    split_scores[split][update] = value[update]
                    split_indices[split][update] = z0_idx + index[update]
            del interp_L, interp_R

            if method == 'fringe':
                block_best_score, block_best_local_idx = corr_all.min(dim=1)     # (Nc,)
                update_mask = block_best_score < best_score
            else:
                block_best_score, block_best_local_idx = corr_all.max(dim=1)     # (Nc,)
                update_mask = block_best_score > best_score

            best_score[update_mask] = block_best_score[update_mask]
            best_z_idx[update_mask] = z0_idx + block_best_local_idx[update_mask].long()

            del corr_all

        # #7: single cache clear after the full loop — not inside each block
        torch.cuda.empty_cache()

        z_best_values_overall = self.z_vals[best_z_idx]
        x_coords_final = self.x_vals[IX_centers]
        y_coords_final = self.y_vals[IY_centers]

        xyz_final = torch.stack([x_coords_final, y_coords_final, z_best_values_overall],
                                dim=1).to(torch.float32)

        if validate_split:
            error = (self.z_vals[split_indices[0]] - self.z_vals[split_indices[1]]).abs()
            # nan_to_num may turn -inf into a finite negative sentinel. Also
            # reject flat, zero-energy subsets: they have no meaningful peak.
            split_valid = [(torch.isfinite(s) & (s > self.epsilon) & (i > 0) & (i < Nz_total-1))
                           for s, i in zip(split_scores, split_indices)]
            finite = split_valid[0] & split_valid[1]
            self.laser_diagnostics = {'split_depth_error': torch.where(finite, error, float('inf'))}
            self.laser_diagnostics['split_z'] = torch.stack([self.z_vals[i] for i in split_indices], 1)
            self.laser_diagnostics['split_score'] = torch.stack(
                [torch.where(v, s, float('-inf')) for s, v in zip(split_scores, split_valid)], 1)

        return xyz_final, best_score, best_z_idx

    def process_guided_z(self, Kx, Ky, seed_map, half_window, z_step, stride=1, Nz_block_voxels=10,
                         method='correl', node_mask=None, exclusion=2.0, z_limits=None):
        """Depth scan restricted to a per-node window around a seed surface.

        Instead of sweeping every lattice node through the same global list of
        depths, node ``(ix, iy)`` is evaluated at ``seed_map[ix, iy] + k*z_step``
        for ``k`` in ``[-n, n]`` with ``n = round(half_window / z_step)``. The
        correlation patch of a centre therefore uses its lattice neighbours at
        *their own* seed depth, i.e. the patch follows the seed surface instead
        of a fronto-parallel plane, which improves the correlation on tilted or
        curved surfaces. Nodes whose seed is NaN (or excluded by ``node_mask``)
        are not evaluated.

        Parameters
        ----------
        seed_map : (Nx, Ny) tensor aligned with ``self.x_vals`` / ``self.y_vals``.
        half_window : half width of the depth window in mm.
        z_step : depth sampling in mm.
        node_mask : optional (Nx, Ny) bool tensor selecting which centres to report.

        Returns
        -------
        xyz (Nc, 3), score (Nc,), offset index (Nc,), runner-up score (Nc,) for
        the reported centres. The runner-up is the best score found at least
        ``exclusion`` mm away from the winner (from per-block maxima), so that
        ``score - runner_up`` measures how distinct the chosen peak is: a true
        correspondence stands clearly above the rest of the window, a chance
        match does not.
        """
        self._validate_scan(Kx, Ky, stride, method)
        Nx, Ny = len(self.x_vals), len(self.y_vals)
        Nz_block_voxels = self._scan_block_size(Nx, Ny, Nz_block_voxels)
        T = self.left_images.shape[0]
        if Nx < Kx or Ny < Ky:
            empty = torch.empty(0, device=self.device)
            return empty.new_empty((0, 3)), empty, empty.long(), empty
        seed = seed_map.to(self.device, torch.float32)
        if seed.shape != (Nx, Ny):
            raise ValueError(f'seed_map shape {tuple(seed.shape)} does not match lattice {(Nx, Ny)}')
        n_off = max(1, int(round(half_window / z_step)))
        offsets = torch.arange(-n_off, n_off + 1, device=self.device, dtype=torch.float32) * z_step

        pad_x, pad_y = Kx // 2, Ky // 2
        ix_centers = torch.arange(pad_x, Nx - pad_x, stride, device=self.device)
        iy_centers = torch.arange(pad_y, Ny - pad_y, stride, device=self.device)
        if len(ix_centers) == 0 or len(iy_centers) == 0:
            empty = torch.empty(0, device=self.device)
            return empty.new_empty((0, 3)), empty, empty.long(), empty
        IX, IY = torch.meshgrid(ix_centers, iy_centers, indexing='ij')
        IX, IY = IX.ravel(), IY.ravel()
        Nc = IX.shape[0]

        X_2d, Y_2d = torch.meshgrid(self.x_vals, self.y_vals, indexing='ij')
        XY_flat = torch.stack([X_2d.ravel(), Y_2d.ravel()], dim=1)
        seed_flat = seed.ravel()
        worst = float('inf') if method == 'fringe' else float('-inf')
        best_score = torch.full((Nc,), worst, device=self.device, dtype=torch.float32)
        best_k = torch.zeros(Nc, dtype=torch.long, device=self.device)
        n_blocks = (len(offsets) + Nz_block_voxels - 1) // Nz_block_voxels
        block_best = torch.full((Nc, n_blocks), worst, device=self.device, dtype=torch.float32)

        for bi, k0 in enumerate(range(0, len(offsets), Nz_block_voxels)):
            k1 = min(k0 + Nz_block_voxels, len(offsets))
            cur = k1 - k0
            z_col = (seed_flat[None, :] + offsets[k0:k1, None]).reshape(-1, 1)       # (cur*Nx*Ny, 1)
            if z_limits is not None:
                z_col = z_col.masked_fill((z_col < z_limits[0]) | (z_col > z_limits[1]), torch.nan)
            xy = XY_flat.unsqueeze(0).expand(cur, -1, -1).reshape(-1, 2)
            grid_flat = torch.cat([xy, z_col], dim=1)
            del xy, z_col
            uv_l, m_l = self.transform_gcs2ccs(grid_flat, 'left', image_shape=self.left_images.shape[1:])
            uv_r, m_r = self.transform_gcs2ccs(grid_flat, 'right', image_shape=self.right_images.shape[1:])
            del grid_flat
            interp_l = self.interpolate_images(self.left_images, uv_l, uv_mask=m_l, method=method)
            interp_r = self.interpolate_images(self.right_images, uv_r, uv_mask=m_r, method=method)
            del uv_l, uv_r, m_l, m_r
            corr_all = self._unfold_and_zncc(interp_l, interp_r, cur, T, Nx, Ny, Kx, Ky, stride, method)
            del interp_l, interp_r
            if method == 'fringe':
                blk_best, blk_idx = corr_all.min(dim=1)
                update = blk_best < best_score
            else:
                blk_best, blk_idx = corr_all.max(dim=1)
                update = blk_best > best_score
            block_best[:, bi] = blk_best
            best_score[update] = blk_best[update]
            best_k[update] = k0 + blk_idx[update].long()
            del corr_all

        # Runner-up: best block maximum at least ``exclusion`` mm from the winner.
        excl_blocks = max(1, int(round(exclusion / (z_step * Nz_block_voxels))))
        best_block = best_k // Nz_block_voxels
        far = (torch.arange(n_blocks, device=self.device)[None, :] - best_block[:, None]).abs() > excl_blocks
        masked = torch.where(far, block_best, torch.full_like(block_best, worst))
        runner_up = masked.min(dim=1).values if method == 'fringe' else masked.max(dim=1).values

        z_best = seed[IX, IY] + offsets[best_k]
        xyz = torch.stack([self.x_vals[IX], self.y_vals[IY], z_best], dim=1).to(torch.float32)
        # Poisoned patches carry +-inf (or the +-FLT_MAX that nan_to_num maps them to).
        valid = torch.isfinite(z_best) & (best_score.abs() < 1e30)
        if node_mask is not None:
            valid &= node_mask.to(self.device)[IX, IY]
        return xyz[valid], best_score[valid], (best_k - n_off)[valid], runner_up[valid]

    # ── Accuracy enhancements ─────────────────────────────────────────────────

    def _eval_surface_gradient(self, IX_centers: torch.Tensor, IY_centers: torch.Tensor,
                               z_coarse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate local surface slopes (dZ/dX, dZ/dY) from the coarse depth map
        using central finite differences on a reconstructed Nx×Ny depth grid.

        Returns
        -------
        sz_x, sz_y : (Nc,) dimensionless slopes [mm/mm]
        """
        Nx      = len(self.x_vals)
        Ny      = len(self.y_vals)
        xy_step = (self.x_vals[1] - self.x_vals[0]).item() if Nx > 1 else 1.0

        # Scatter coarse Z onto the full XY grid
        depth_flat = torch.full((Nx * Ny,), float('nan'), device=self.device, dtype=torch.float32)
        depth_flat[IX_centers * Ny + IY_centers] = z_coarse
        depth_map = depth_flat.view(Nx, Ny)

        # Replace NaN border pixels (kernel padding) with coarse mean to avoid gradient spikes
        depth_map = torch.where(depth_map.isnan(), z_coarse.mean(), depth_map)

        # Central differences (interior) + one-sided (border)
        gx = torch.zeros_like(depth_map)
        gy = torch.zeros_like(depth_map)
        if Nx > 2:
            gx[1:-1, :] = (depth_map[2:, :]  - depth_map[:-2, :]) / (2.0 * xy_step)
        if Nx > 1:
            gx[0,  :] = (depth_map[1,  :] - depth_map[0,  :]) / xy_step
            gx[-1, :] = (depth_map[-1, :] - depth_map[-2, :]) / xy_step
        if Ny > 2:
            gy[:, 1:-1] = (depth_map[:, 2:]  - depth_map[:, :-2]) / (2.0 * xy_step)
        if Ny > 1:
            gy[:, 0]  = (depth_map[:, 1]  - depth_map[:, 0])  / xy_step
            gy[:, -1] = (depth_map[:, -1] - depth_map[:, -2]) / xy_step

        return gx[IX_centers, IY_centers], gy[IX_centers, IY_centers]

    def _score_at_z_per_point(self, x_pts: torch.Tensor, y_pts: torch.Tensor,
                               z_per_point: torch.Tensor, Kx: int, Ky: int,
                               method: str,
                               sz_x: torch.Tensor | None = None,
                               sz_y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Evaluate ZNCC / fringe score for Nc points each at its own Z depth.

        Surface-following (when sz_x / sz_y provided)
            Kernel offset (di, dj) is projected at
            ``z + sz_x · di · xy_step + sz_y · dj · xy_step``
            so the correlation patch lies on the local tangent plane instead of
            the fronto-parallel XY plane.  Eliminates the ~0.10 mm angular bias
            on slopes ≳10° with no extra Z evaluations.

        Parameters
        ----------
        x_pts, y_pts   : (Nc,) world XY of each correlation centre [mm]
        z_per_point    : (Nc,) world Z per centre [mm]
        sz_x, sz_y     : (Nc,) surface slopes dZ/dX, dZ/dY  (None → fronto-parallel)

        Returns
        -------
        scores : (Nc,)
        """
        Nc      = x_pts.shape[0]
        pad_x, pad_y = Kx // 2, Ky // 2
        xy_step = (self.x_vals[1] - self.x_vals[0]).item() if len(self.x_vals) > 1 else 1.0
        di_vals = torch.arange(-pad_x, pad_x + 1, device=self.device)
        dj_vals = torch.arange(-pad_y, pad_y + 1, device=self.device)
        DI, DJ = torch.meshgrid(di_vals, dj_vals, indexing='ij')
        DI = DI.ravel() # (K_area,)
        DJ = DJ.ravel()
        K_area = Kx * Ky

        x_grid = x_pts.unsqueeze(1) + (DI * xy_step).unsqueeze(0) # (Nc, K_area)
        y_grid = y_pts.unsqueeze(1) + (DJ * xy_step).unsqueeze(0)

        if sz_x is not None and sz_y is not None:
            dz_corr = sz_x.unsqueeze(1) * (DI * xy_step).unsqueeze(0) + sz_y.unsqueeze(1) * (DJ * xy_step).unsqueeze(0)
        else:
            dz_corr = 0.0

        z_grid = (z_per_point.unsqueeze(1) + dz_corr).expand_as(x_grid)

        # Flatten to (Nc * K_area)
        pts = torch.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], dim=1) # (Nc * K_area, 3)

        uv_L, m_L = self.transform_gcs2ccs(pts, 'left', image_shape=self.left_images.shape[1:])
        uv_R, m_R = self.transform_gcs2ccs(pts, 'right', image_shape=self.right_images.shape[1:])

        T = self.left_images.shape[0]
        L_interp = self.interpolate_images(self.left_images, uv_L, uv_mask=m_L, method=method) # (Nc * K_area, T)
        R_interp = self.interpolate_images(self.right_images, uv_R, uv_mask=m_R, method=method)

        # Reshape to match the (Nc, K_area * T) layout exactly
        # Using .reshape() instead of .view() handles non-contiguous memory from interpolate_images
        L_all = L_interp.reshape(Nc, K_area * T)
        R_all = R_interp.reshape(Nc, K_area * T)

        if method == 'fringe':
            return torch.nan_to_num(torch.abs(L_interp.reshape(Nc, K_area, T)[:, K_area//2, 0] - R_interp.reshape(Nc, K_area, T)[:, K_area//2, 0]), nan=float('inf'))
        else:
            return torch.nan_to_num(self.zncc_correlation(L_all, R_all), nan=float('-inf'))

    def _parabolic_refine(self, best_z_idx: torch.Tensor, best_score: torch.Tensor,
                           score_m1: torch.Tensor, score_p1: torch.Tensor,
                           method: str = 'correl') -> torch.Tensor:
        """
        Sub-voxel Z refinement via quadratic (parabolic) peak interpolation.

        Fits a parabola through the three scores at (best_z−1, best_z, best_z+1):

            offset = 0.5 × (s₋₁ − s₊₁) / (s₋₁ − 2·s₀ + s₊₁)
            z_refined = z_vals[best_z] + offset × dz

        Refinement is only applied when the parabola is well-conditioned (genuine
        local optimum) and the index is not at the Z-range boundary.
        Reduces quantisation RMS from ``dz/√12`` to roughly ``0.05·dz``.

        Returns
        -------
        z_refined : (Nc,) sub-voxel Z [mm]
        """
        Nz = len(self.z_vals)
        dz = (self.z_vals[1] - self.z_vals[0]).item() if Nz > 1 else 1.0

        denom        = score_m1 - 2.0 * best_score + score_p1      # (Nc,)
        at_boundary  = (best_z_idx <= 0) | (best_z_idx >= Nz - 1)

        # Concave-down (denom < 0) for max (ZNCC); concave-up (denom > 0) for min (fringe)
        if method == 'fringe':
            valid = (denom >  self.epsilon) & ~at_boundary
        else:
            valid = (denom < -self.epsilon) & ~at_boundary

        offset = torch.zeros_like(best_score)
        offset[valid] = 0.5 * (score_m1[valid] - score_p1[valid]) / denom[valid]
        offset = offset.clamp(-0.5, 0.5)                            # sanity: ≤ ½ voxel

        z_base    = self.z_vals[best_z_idx.clamp(0, Nz - 1)]
        return z_base + offset * dz

    def process_coarse_to_fine(self, Kx: int, Ky: int, stride: int = 1,
                                dz_coarse: float = 1.0, fine_window_mm: float = 3.0,
                                Nz_block_coarse: int = 20, method: str = 'correl',
                                surface_following: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-pass coarse-to-fine Z-scan with surface-following and sub-voxel refinement.

        Pass 1 — Coarse  (fast, full Z range)
            Scans z_min → z_max at ``dz_coarse`` using the standard vectorised block
            scanner.  Locates the approximate surface Z (``z_seed``) per XY point.

        Pass 2 — Fine  (precise, windowed)
            Each point scans only ``[z_seed − fine_window_mm, z_seed + fine_window_mm]``
            at the original fine z_step.  All XY points share the same Z block loop; a
            per-point boolean mask gates which updates are applied, so the loop naturally
            skips entire Z blocks that fall outside every active window.
            Typical reduction: 2000-slice brute force → ≈60 fine slices per point.

        Surface-following  (optional, default ON)
            After the coarse pass, the local depth gradient (dZ/dX, dZ/dY) is estimated
            via central differences on the coarse depth map.  Each kernel offset (di, dj)
            in the fine pass is projected at
            ``z + sz_x·di·xy_step + sz_y·dj·xy_step``
            so the ZNCC patch lies on the local surface tangent plane, eliminating the
            ~0.10 mm angular bias present on slopes ≳10° with the fronto-parallel window.
            Computational overhead: Kx·Ky separate projection calls per fine Z slice
            (instead of one large call); the total voxel count remains far below brute force.

        Parabolic sub-voxel refinement  (always applied)
            After the fine argmax, scores at ``best_z ± 1`` are evaluated (2 extra calls to
            ``_score_at_z_per_point``) and a parabola is fit to locate the sub-voxel peak.
            Reduces Z quantisation RMS from ``dz/√12`` to ~``0.05·dz``
            (e.g. 0.14 mm → 0.025 mm at dz = 0.5 mm).

        Parameters
        ----------
        Kx, Ky           : correlation kernel size
        stride           : XY stride between correlation centres
        dz_coarse        : Z step for the coarse pass [mm]  (0.5–1 mm recommended)
        fine_window_mm   : half-width of the per-point fine Z window [mm]
        Nz_block_coarse  : block size for the coarse pass
        method           : 'correl' (ZNCC) | 'fringe' (phase diff)
        surface_following: enable slope-corrected patch sampling in the fine pass

        Returns
        -------
        xyz_final  : (Nc, 3) — sub-voxel 3-D point cloud [mm]
        corr_final : (Nc,)   — peak ZNCC / fringe score
        best_z_idx : (Nc,)   — integer fine Z index (before sub-voxel correction)
        """
        orig_z  = self.z_vals.clone()
        z_min   = orig_z[0].item()
        z_max   = orig_z[-1].item()
        dz_fine = (orig_z[1] - orig_z[0]).item() if len(orig_z) > 1 else dz_coarse

        # ── Pass 1: Coarse ────────────────────────────────────────────────────
        Nz_c = max(2, int(round((z_max - z_min) / dz_coarse)) + 1)
        print(f"[C2F] Coarse pass — dz={dz_coarse:.1f} mm  Nz={Nz_c}")
        self.z_vals = torch.linspace(z_min, z_max, Nz_c,
                                     dtype=torch.float32, device=self.device)
        xyz_c, _, _ = self.process_segmented_z(
            Kx, Ky, stride=stride, Nz_block_voxels=Nz_block_coarse, method=method
        )
        z_seed_coarse = xyz_c[:, 2].clone()      # (Nc_coarse,) coarse Z seed per XY point
        self.z_vals = orig_z              # restore fine grid

        # ── XY centres (Fine Pass always runs at full resolution: stride=1) ───────
        Nx, Ny   = len(self.x_vals), len(self.y_vals)
        Nz_total = len(self.z_vals)
        T        = self.left_images.shape[0]
        pad_x, pad_y = Kx // 2, Ky // 2
        xy_step  = (self.x_vals[1] - self.x_vals[0]).item() if Nx > 1 else 1.0

        ix_c = torch.arange(pad_x, Nx - pad_x, 1, device=self.device)
        iy_c = torch.arange(pad_y, Ny - pad_y, 1, device=self.device)
        IX_c, IY_c = torch.meshgrid(ix_c, iy_c, indexing='ij')
        IX_c, IY_c = IX_c.ravel(), IY_c.ravel()
        Nc    = IX_c.shape[0]
        x_pts = self.x_vals[IX_c]         # (Nc,) world X
        y_pts = self.y_vals[IY_c]         # (Nc,) world Y

        if stride > 1:
            # Upsample the coarse z_seed to match the fine grid
            Nc_x_c = len(torch.arange(pad_x, Nx - pad_x, stride))
            Nc_y_c = len(torch.arange(pad_y, Ny - pad_y, stride))
            z_seed_2d = z_seed_coarse.view(1, 1, Nc_x_c, Nc_y_c)
            z_seed_2d_fine = F.interpolate(z_seed_2d, size=(len(ix_c), len(iy_c)), mode='bilinear', align_corners=False)
            z_seed = z_seed_2d_fine.view(-1)
        else:
            z_seed = z_seed_coarse

        # ── Surface gradient (once, after coarse pass) ─────────────────────────
        sz_x = sz_y = None
        if surface_following:
            print("[C2F] Estimating surface gradient for slope-following patches...")
            sz_x, sz_y = self._eval_surface_gradient(IX_c, IY_c, z_seed)

        Nz_win = max(3, int(round(2 * fine_window_mm / dz_fine)) + 1)
        print(f"[C2F] Fine pass  — dz={dz_fine:.3f} mm  window=±{fine_window_mm:.1f} mm "
              f"({Nz_win} slices/point)  surface_following={surface_following}")

        # ── Pass 2: Fine (per-point windowed Z, surface-following patches) ─────
        if method == 'fringe':
            best_score = torch.full((Nc,), float('inf'),  device=self.device, dtype=torch.float32)
        else:
            best_score = torch.full((Nc,), float('-inf'), device=self.device, dtype=torch.float32)
        best_z_idx = torch.zeros(Nc, dtype=torch.long, device=self.device)

        # Fine blocks are small (window is narrow); ~Nz_win/3 slices each
        Nz_block_fine = max(1, Nz_win // 3)

        for z0 in range(0, Nz_total, Nz_block_fine):
            z1      = min(z0 + Nz_block_fine, Nz_total)
            z_block = self.z_vals[z0:z1]                             # (cur_Nz,)
            cur_Nz  = z_block.shape[0]

            # Per-point window gate: True where this Z slice is inside z_seed ± window
            in_win = ((z_block.unsqueeze(0) - z_seed.unsqueeze(1)).abs()
                      <= fine_window_mm)                             # (Nc, cur_Nz)
            if not in_win.any():
                continue                                             # skip irrelevant blocks

            if surface_following:
                # Per-Z-slice loop with slope-corrected patch sampling.
                # cur_Nz is tiny (≤ Nz_block_fine ≈ Nz_win/3), so the Python loop is cheap.
                for z_loc in range(cur_Nz):
                    active = in_win[:, z_loc]                        # (Nc,) bool
                    if not active.any():
                        continue
                    z_pt = torch.full((Nc,), z_block[z_loc].item(),
                                      device=self.device, dtype=torch.float32)
                    corr_s = self._score_at_z_per_point(
                        x_pts, y_pts, z_pt, Kx, Ky, method, sz_x, sz_y
                    )                                                # (Nc,)
                    fill = float('inf') if method == 'fringe' else float('-inf')
                    corr_s = torch.where(active, corr_s,
                                         torch.tensor(fill, device=self.device, dtype=torch.float32))
                    update = (corr_s < best_score) if method == 'fringe' else (corr_s > best_score)
                    best_score[update] = corr_s[update]
                    best_z_idx[update] = z0 + z_loc
            else:
                # Fronto-parallel vectorised path — identical to process_segmented_z inner loop
                X2, Y2  = torch.meshgrid(self.x_vals, self.y_vals, indexing='ij')
                XY_flat = torch.stack([X2.ravel(), Y2.ravel()], dim=1)
                del X2, Y2
                Z_col  = z_block.repeat_interleave(Nx * Ny).unsqueeze(1)
                XY_exp = XY_flat.unsqueeze(0).expand(cur_Nz, -1, -1).reshape(-1, 2)
                grid_f = torch.cat([XY_exp, Z_col], dim=1)
                del XY_exp, Z_col, XY_flat

                uv_L, m_L = self.transform_gcs2ccs(grid_f, 'left',  image_shape=self.left_images.shape[1:])
                uv_R, m_R = self.transform_gcs2ccs(grid_f, 'right', image_shape=self.right_images.shape[1:])
                del grid_f

                iL = self.interpolate_images(self.left_images,  uv_L, uv_mask=m_L, method=method)
                iR = self.interpolate_images(self.right_images, uv_R, uv_mask=m_R, method=method)
                del uv_L, uv_R, m_L, m_R

                corr_all = self._unfold_and_zncc(iL, iR, cur_Nz, T, Nx, Ny, Kx, Ky, stride, method)
                fill_val = float('inf') if method == 'fringe' else float('-inf')
                del iL, iR

                # Apply per-point window mask before argmax/argmin
                corr_all = torch.where(in_win, corr_all,
                                       torch.tensor(fill_val, device=self.device, dtype=torch.float32))

                if method == 'fringe':
                    blk_best, blk_idx = corr_all.min(dim=1)
                    update = blk_best < best_score
                else:
                    blk_best, blk_idx = corr_all.max(dim=1)
                    update = blk_best > best_score

                best_score[update] = blk_best[update]
                best_z_idx[update] = z0 + blk_idx[update].long()
                del corr_all

        torch.cuda.empty_cache()

        # ── Parabolic sub-voxel refinement (2 targeted Z evaluations per point) ──
        print("[C2F] Parabolic sub-voxel refinement...")
        zm1 = self.z_vals[(best_z_idx - 1).clamp(0, Nz_total - 1)]
        zp1 = self.z_vals[(best_z_idx + 1).clamp(0, Nz_total - 1)]
        s_m1 = self._score_at_z_per_point(x_pts, y_pts, zm1, Kx, Ky, method, sz_x, sz_y)
        s_p1 = self._score_at_z_per_point(x_pts, y_pts, zp1, Kx, Ky, method, sz_x, sz_y)
        z_refined = self._parabolic_refine(best_z_idx, best_score, s_m1, s_p1, method)

        xyz_final = torch.stack([x_pts, y_pts, z_refined], dim=1)
        return xyz_final, best_score, best_z_idx

    def get_cropped_image_bounds(self, image_shape, crop_factor: float):
        """
        Returns the bounds (ymin, ymax, xmin, xmax) for a crop centered in the image.
        crop_factor: float in (0, 1], e.g. 0.5 means crop to 50% of original size.
        """
        H, W = image_shape
        crop_H = int(H * crop_factor)
        crop_W = int(W * crop_factor)
        center_y = H // 2
        center_x = W // 2
        ymin = max(center_y - crop_H // 2, 0)
        ymax = min(center_y + crop_H // 2, H)
        xmin = max(center_x - crop_W // 2, 0)
        xmax = min(center_x + crop_W // 2, W)
        return ymin, ymax, xmin, xmax

    def std_mask_points(self, xyz_gpu: torch.Tensor, corr_gpu: torch.Tensor, bounds, method='correl') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        uv_left_final, in_left = self.transform_gcs2ccs(xyz_gpu, 'left', image_shape=self.left_images.shape[1:])
        uv_right_final, in_right = self.transform_gcs2ccs(xyz_gpu, 'right', image_shape=self.right_images.shape[1:])
        L_interp, R_interp = self.interpolate_images(self.left_images, uv_left_final, uv_mask=in_left, method=method), self.interpolate_images(self.right_images, uv_right_final, uv_mask=in_right, method=method)

        if method == 'fringe':
            # Fringe maps carry absolute phase in channel 0 and modulation in channel 1.
            if L_interp.shape[1] < 2 or R_interp.shape[1] < 2:
                raise ValueError("Fringe modulation masking requires [phase, modulation] images")
            std_mask = (L_interp[:, 1] > bounds) & (R_interp[:, 1] > bounds)
            L_masked = L_interp[:, 1]
        else:
            L_std, R_std = L_interp.std(dim=1), R_interp.std(dim=1)
            # ``bounds`` is an absolute std on equalised intensities, which
            # depends on exposure: a dark capture fails it everywhere. Use it as
            # a cap and otherwise require 3x the background noise level.
            noise = getattr(self, 'noise_level', 0.0)
            eff = min(float(bounds), max(2.0, 3.0 * noise)) if noise > 0 else float(bounds)
            std_mask = (eff < L_std) & (eff < R_std)
            L_masked = L_std

        combined_mask = std_mask & torch.isfinite(L_interp).all(1) & torch.isfinite(R_interp).all(1)
        xyz_masked = xyz_gpu[combined_mask]
        corr_masked = corr_gpu[combined_mask]
        L_masked = L_masked[combined_mask]

        return xyz_masked, corr_masked, L_masked

    def mask_uv_points(self, xyz_gpu: torch.Tensor, corr_gpu: torch.Tensor, crop_factor: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uv_left_final, uv_left_final_mask = self.transform_gcs2ccs(xyz_gpu, 'left', image_shape=self.left_images.shape[1:])
        _, uv_right_final_mask = self.transform_gcs2ccs(xyz_gpu, 'right', image_shape=self.right_images.shape[1:])
        L_masked = self.interpolate_images(self.left_images, uv_left_final)[:,0]

        # Crop mask based on center
        if crop_factor < 1.0:
            H, W = self.left_images.shape[1:]
            ymin, ymax, xmin, xmax = self.get_cropped_image_bounds((H, W), crop_factor)
            crop_mask = (
                (uv_left_final[:, 0] >= xmin) & (uv_left_final[:, 0] < xmax) &
                (uv_left_final[:, 1] >= ymin) & (uv_left_final[:, 1] < ymax)
            )
        else:
            crop_mask = torch.ones_like(uv_left_final_mask, dtype=torch.bool)

        combined_mask = uv_left_final_mask & uv_right_final_mask & crop_mask
        xyz_masked = xyz_gpu[combined_mask]
        corr_masked = corr_gpu[combined_mask]
        L_masked = L_masked[combined_mask]

        return xyz_masked, corr_masked, L_masked

    def euclidean_filter(self, xyz_gpu: torch.Tensor, corr_gpu: torch.Tensor = None, interp: torch.Tensor = None, min_neighbors: int = 5, radius: float = 10.0, batch_size=1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = xyz_gpu.shape[0]
        final_mask = torch.zeros(n, dtype=torch.bool, device=xyz_gpu.device)

        # Itera sobre os pontos em lotes
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)

            # Pega o lote de pontos atual
            xyz_batch = xyz_gpu[i:i_end]

            # Calcula a matriz de distância APENAS entre o lote atual e TODOS os outros pontos.
            # Isso ainda pode ser grande, mas é uma melhora significativa.
            dist_batch = torch.cdist(xyz_batch, xyz_gpu)

            # 1. Aplica o filtro de raio (mask_dist)
            # As distâncias entre um ponto e ele mesmo são 0, então `dist_batch > 0` já exclui a diagonal do lote
            # (se o lote for a matriz completa) e garante que o ponto não seja seu próprio vizinho.
            mask_dist = (dist_batch > 0) & (dist_batch < radius)

            # 2. Conta os vizinhos para cada ponto do lote
            neighbors_count = mask_dist.sum(dim=1)

            # 3. Cria a máscara para os pontos que têm vizinhos suficientes
            mask_neighbors = neighbors_count > min_neighbors

            # 4. Atualiza a máscara final
            final_mask[i:i_end] = mask_neighbors

            del dist_batch, mask_dist, neighbors_count, mask_neighbors

        if corr_gpu is None and interp is None:
            return xyz_gpu[final_mask], None, None

        elif corr_gpu is None and interp is not None:
            return xyz_gpu[final_mask], None, interp[final_mask]

        elif corr_gpu is not None and interp is None:
            return xyz_gpu[final_mask], corr_gpu[final_mask], None
        else:
            # Aplica a máscara final aos tensores originais
            return xyz_gpu[final_mask], corr_gpu[final_mask], interp[final_mask]

    def euclidean_filter_kdtree(self, xyz_gpu: torch.Tensor, corr_gpu: torch.Tensor = None,
                               interp: torch.Tensor = None, min_neighbors: int = 5,
                               radius: float = 10.0):
        """CPU spatial-index alternative to the quadratic all-pairs filter.

        Preserves strict 0 < distance < radius and count > min_neighbors.
        Exact duplicate coordinates, including the query point, do not count.
        Returned tensors stay on the input device. GPU timings are workload-dependent.
        """
        from scipy.spatial import cKDTree
        if radius <= 0:
            raise ValueError('radius must be positive')
        xyz_cpu = xyz_gpu.detach().cpu().numpy().astype(np.float64, copy=False)
        if len(xyz_cpu):
            tree = cKDTree(xyz_cpu)
            counts = tree.query_ball_point(xyz_cpu, np.nextafter(float(radius), -np.inf),
                                           return_length=True)
            coincident = tree.query_ball_point(xyz_cpu, 0., return_length=True)
            mask = torch.as_tensor(counts - coincident > min_neighbors, device=xyz_gpu.device)
        else:
            mask = torch.zeros(0, dtype=torch.bool, device=xyz_gpu.device)
        return (xyz_gpu[mask], None if corr_gpu is None else corr_gpu[mask],
                None if interp is None else interp[mask])

    def set_rectification(self, alpha=-1):
        """
        Compute rectification maps using cv2.stereoRectify and the current camera params,
        and rectify the loaded left and right images.
        """
        cl = self.camera_params['left']
        cr = self.camera_params['right']

        K1 = cl['kk'].cpu().numpy()
        D1 = cl['kc'].cpu().numpy()
        R1_calib = cl['r'].cpu().numpy()
        T1_calib = cl['t'].cpu().numpy()

        K2 = cr['kk'].cpu().numpy()
        D2 = cr['kc'].cpu().numpy()
        if self._images_undistorted:
            D1, D2 = np.zeros_like(D1), np.zeros_like(D2)
        R2_calib = cr['r'].cpu().numpy()
        T2_calib = cr['t'].cpu().numpy()

        # Relative rotation and translation between cameras
        R = R2_calib @ R1_calib.T
        T = T2_calib - R @ T1_calib

        T_frames, H, W = self.left_images.shape
        size = (W, H)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha, newImageSize=size
        )

        self.rect = {
            'R1': torch.from_numpy(R1).to(self.device).to(torch.float32),
            'R2': torch.from_numpy(R2).to(self.device).to(torch.float32),
            'P1': torch.from_numpy(P1).to(self.device).to(torch.float32),
            'P2': torch.from_numpy(P2).to(self.device).to(torch.float32),
            'Q': torch.from_numpy(Q).to(self.device).to(torch.float32)
        }
        self.rect_f = float(P1[0, 0])
        self.rect_cx = float(P1[0, 2])
        self.rect_cy = float(P1[1, 2])
        self.rect_B = float(-P2[0, 3] / P2[0, 0])

        m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
        m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)

        # Use GPU grid_sample for instantaneous rectification instead of slow CPU cv2.remap loop
        map1_x = torch.from_numpy(m1[0]).to(self.device)
        map1_y = torch.from_numpy(m1[1]).to(self.device)
        map2_x = torch.from_numpy(m2[0]).to(self.device)
        map2_y = torch.from_numpy(m2[1]).to(self.device)

        # Normalize maps to [-1, 1] for grid_sample
        map1_x = (map1_x / (W - 1)) * 2 - 1
        map1_y = (map1_y / (H - 1)) * 2 - 1
        map2_x = (map2_x / (W - 1)) * 2 - 1
        map2_y = (map2_y / (H - 1)) * 2 - 1

        grid1 = torch.stack([map1_x, map1_y], dim=-1).unsqueeze(0).expand(T_frames, -1, -1, -1)
        grid2 = torch.stack([map2_x, map2_y], dim=-1).unsqueeze(0).expand(T_frames, -1, -1, -1)

        # unsqueeze(1) makes shape (T, 1, H, W) for grid_sample, which returns (T, 1, H, W). squeeze(1) brings it back to (T, H, W).
        self.left_rect = F.grid_sample(self.left_images.unsqueeze(1), grid1, mode='bilinear', align_corners=True, padding_mode='border').squeeze(1)
        self.right_rect = F.grid_sample(self.right_images.unsqueeze(1), grid2, mode='bilinear', align_corners=True, padding_mode='border').squeeze(1)

        self.left_rect_c = self.left_rect - self.left_rect.mean(dim=0, keepdim=True)
        self.right_rect_c = self.right_rect - self.right_rect.mean(dim=0, keepdim=True)
        print(f"[Rectify] f={self.rect_f:.1f} px, B={self.rect_B:.2f} mm, size {W}x{H}, T={T_frames}")

    def world_to_rect(self, Xw: torch.Tensor):
        """World points (N, 3) -> (u, v, d, Z_rect) in rectified left camera."""
        cl = self.camera_params['left']
        Xl = torch.matmul(cl['r'], Xw.T) + cl['t']
        Xr = torch.matmul(self.rect['R1'], Xl).T

        u = self.rect_f * Xr[:, 0] / Xr[:, 2] + self.rect_cx
        v = self.rect_f * Xr[:, 1] / Xr[:, 2] + self.rect_cy
        d = self.rect_f * self.rect_B / Xr[:, 2]
        return u, v, d, Xr[:, 2]

    def rect_to_world(self, u: torch.Tensor, v: torch.Tensor, d: torch.Tensor):
        """Rectified left pixel (u, v) + disparity d -> world (N, 3)."""
        Z = self.rect_f * self.rect_B / d
        X = (u - self.rect_cx) * Z / self.rect_f
        Y = (v - self.rect_cy) * Z / self.rect_f
        Xr = torch.stack([X, Y, Z], dim=1)

        cl = self.camera_params['left']
        Xl = torch.matmul(self.rect['R1'].T, Xr.T)
        Xw = torch.matmul(cl['r'].T, Xl - cl['t']).T
        return Xw

    def box_sum_2d(self, tensor: torch.Tensor, kh: int, kw: int):
        t = tensor.unsqueeze(0).unsqueeze(0)
        s = F.avg_pool2d(t, (kh, kw), stride=1, divisor_override=1)
        return s.squeeze(0).squeeze(0)

    def process_epipolar_coarse(self, roi, d_range, k: int = 9, corr_threshold: float = 0.3):
        import time
        t0 = time.time()

        u0, u1, v0, v1 = roi
        d_min, d_max = d_range

        L = self.left_rect_c[:, v0:v1, u0:u1]

        T_frames, H, W = L.shape
        ru0 = max(0, u0 - d_max)
        ru1 = min(self.right_rect_c.shape[2], u1 - d_min)
        Rfull = self.right_rect_c[:, v0:v1, ru0:ru1]

        Hc, Wc = H - k + 1, W - k + 1
        n = float(k * k * T_frames)

        # Pre-sum over T axis to avoid shifting 3D tensors in the loop
        L_sum = L.sum(dim=0)
        LL_sum = (L * L).sum(dim=0)
        Rfull_sum = Rfull.sum(dim=0)
        RRfull_sum = (Rfull * Rfull).sum(dim=0)

        sL = self.box_sum_2d(L_sum, k, k)
        sLL = self.box_sum_2d(LL_sum, k, k)
        varL = sLL - (sL * sL) / n

        neg = float('-inf')
        best = torch.full((Hc, Wc), neg, device=self.device, dtype=torch.float32)
        second = torch.full((Hc, Wc), neg, device=self.device, dtype=torch.float32)
        best_d = torch.zeros((Hc, Wc), device=self.device, dtype=torch.int32)

        cm1 = torch.full((Hc, Wc), neg, device=self.device, dtype=torch.float32)
        cp1 = torch.full((Hc, Wc), neg, device=self.device, dtype=torch.float32)
        prev2 = torch.full((Hc, Wc), neg, device=self.device, dtype=torch.float32)
        prev1 = torch.full((Hc, Wc), neg, device=self.device, dtype=torch.float32)
        pend = torch.zeros((Hc, Wc), device=self.device, dtype=torch.bool)

        ds = list(range(d_min, d_max + 1))
        eps = 1e-6

        for d in ds:
            off = u0 - d - ru0

            # Instead of shifting 3D Rs, shift 2D sums!
            Rs_sum = torch.zeros((H, W), device=self.device, dtype=torch.float32)
            RRs_sum = torch.zeros((H, W), device=self.device, dtype=torch.float32)
            valid = torch.zeros((H, W), dtype=torch.bool, device=self.device)

            a = max(0, -off)
            b = min(W, Rfull.shape[2] - off)

            if b > a:
                Rs_sum[:, a:b] = Rfull_sum[:, a+off:b+off]
                RRs_sum[:, a:b] = RRfull_sum[:, a+off:b+off]
                valid[:, a:b] = True

                # Cross term L * Rs still needs a shift of Rfull and multiply by L
                Rs = Rfull[:, :, a+off:b+off]
                LRs_sum = torch.zeros((H, W), device=self.device, dtype=torch.float32)
                LRs_sum[:, a:b] = (L[:, :, a:b] * Rs).sum(dim=0)
            else:
                LRs_sum = torch.zeros((H, W), device=self.device, dtype=torch.float32)

            sR = self.box_sum_2d(Rs_sum, k, k)
            sRR = self.box_sum_2d(RRs_sum, k, k)
            sLR = self.box_sum_2d(LRs_sum, k, k)
            cnt = self.box_sum_2d(valid.to(torch.float32), k, k)

            varR = sRR - (sR * sR) / n
            cov = sLR - (sL * sR) / n

            c = torch.where(
                (cnt >= k * k - 0.5) & (varL > eps) & (varR > eps),
                cov / torch.sqrt(torch.clamp(varL, min=eps) * torch.clamp(varR, min=eps)),
                torch.tensor(neg, device=self.device)
            )

            cp1 = torch.where(pend, c, cp1)
            pend.zero_()

            is_peak = (prev1 > prev2) & (prev1 >= c) & torch.isfinite(prev1)
            upd = is_peak & (prev1 > best)

            second = torch.where(upd, torch.maximum(second, best),
                                 torch.where(is_peak, torch.maximum(second, prev1), second))

            best = torch.where(upd, prev1, best)
            best_d = torch.where(upd, d - 1, best_d)
            cm1 = torch.where(upd, prev2, cm1)
            cp1 = torch.where(upd, c, cp1)

            prev2 = prev1
            prev1 = c

        is_peak = (prev1 > prev2) & torch.isfinite(prev1)
        upd = is_peak & (prev1 > best)
        best = torch.where(upd, prev1, best)
        best_d = torch.where(upd, ds[-1], best_d)
        cm1 = torch.where(upd, prev2, cm1)
        cp1 = torch.where(upd, torch.tensor(neg, device=self.device), cp1)

        found = torch.isfinite(best)
        boundary = ~(torch.isfinite(cm1) & torch.isfinite(cp1)) | ~found

        zero = torch.tensor(0.0, device=self.device)
        a_ = torch.where(boundary, zero, cm1)
        b_ = torch.where(boundary, zero, cp1)
        m_ = torch.where(found, best, zero)

        denom = a_ - 2 * m_ + b_
        good = found & ~boundary & (denom < -1e-9)

        offset = torch.where(good, 0.5 * (a_ - b_) / torch.where(good, denom, torch.tensor(-1.0, device=self.device)), zero)
        offset = torch.clamp(offset, -1.0, 1.0)

        d_sub = best_d.to(torch.float32) + offset

        vv, uu = torch.meshgrid(
            torch.arange(v0 + k // 2, v0 + k // 2 + Hc, device=self.device),
            torch.arange(u0 + k // 2, u0 + k // 2 + Wc, device=self.device),
            indexing="ij"
        )

        sel = found & ~boundary & (best >= corr_threshold)
        u_sel = uu[sel]
        v_sel = vv[sel]
        d_sel = d_sub[sel]
        corr_sel = best[sel]

        xyz = self.rect_to_world(u_sel.to(torch.float32), v_sel.to(torch.float32), d_sel)
        print(f"[Epipolar Coarse] {len(ds)} disparities, {xyz.shape[0]} valid points found in {time.time()-t0:.2f}s")
        return xyz, corr_sel

    def resample_to_lattice(self, xyz: torch.Tensor, corr: torch.Tensor, min_pts: int = 1):
        Nx, Ny = len(self.x_vals), len(self.y_vals)
        x0, y0 = self.x_vals[0].item(), self.y_vals[0].item()
        xy_step = (self.x_vals[1] - self.x_vals[0]).item() if Nx > 1 else 1.0

        ix = torch.round((xyz[:, 0] - x0) / xy_step).to(torch.long)
        iy = torch.round((xyz[:, 1] - y0) / xy_step).to(torch.long)

        ok = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)
        ix, iy, z = ix[ok], iy[ok], xyz[ok, 2]
        lin = ix * Ny + iy

        z_seed = torch.full((Nx * Ny,), float('nan'), device=self.device)
        if len(lin) > 0:
            # Vectorized mean instead of slow python median loop
            sums = torch.zeros(Nx * Ny, device=self.device, dtype=torch.float32)
            counts = torch.zeros(Nx * Ny, device=self.device, dtype=torch.float32)

            sums.scatter_add_(0, lin, z.to(torch.float32))
            counts.scatter_add_(0, lin, torch.ones_like(z, dtype=torch.float32))

            valid = counts >= min_pts
            z_seed[valid] = sums[valid] / counts[valid]

        return z_seed

    def _zncc_batch(self, left, right):
        left = left - left.mean(dim=1, keepdim=True)
        right = right - right.mean(dim=1, keepdim=True)
        numerator = (left * right).sum(dim=1)
        norm = (left.square().sum(dim=1) * right.square().sum(dim=1)).sqrt()
        return (numerator / norm.clamp_min(self.epsilon)).masked_fill(norm <= self.epsilon, float('-inf'))

    def process_epipolar_to_fine(self, Kx: int, Ky: int, stride: int = 1,
                                 fine_window_mm: float = 3.0, method: str = 'correl',
                                 surface_following: bool = True):
        import time
        t_start = time.time()

        if not hasattr(self, 'rect'):
            self.set_rectification()

        x_lim = [self.x_vals[0].item(), self.x_vals[-1].item()]
        y_lim = [self.y_vals[0].item(), self.y_vals[-1].item()]
        z_lim = [self.z_vals[0].item(), self.z_vals[-1].item()]

        c = torch.tensor([
            [x_lim[0], y_lim[0], z_lim[0]], [x_lim[1], y_lim[0], z_lim[0]],
            [x_lim[0], y_lim[1], z_lim[0]], [x_lim[1], y_lim[1], z_lim[0]],
            [x_lim[0], y_lim[0], z_lim[1]], [x_lim[1], y_lim[0], z_lim[1]],
            [x_lim[0], y_lim[1], z_lim[1]], [x_lim[1], y_lim[1], z_lim[1]]
        ], dtype=torch.float32, device=self.device)

        u, v, d, _ = self.world_to_rect(c)
        margin_px = 10
        u0 = max(0, int(torch.floor(u.min()).item()) - margin_px)
        u1 = min(self.left_rect.shape[2], int(torch.ceil(u.max()).item()) + margin_px)
        v0 = max(0, int(torch.floor(v.min()).item()) - margin_px)
        v1 = min(self.left_rect.shape[1], int(torch.ceil(v.max()).item()) + margin_px)

        d_min = int(torch.floor(d.min()).item()) - 2
        d_max = int(torch.ceil(d.max()).item()) + 2

        print(f"[Hybrid C2F] ROI: u({u0}-{u1}) v({v0}-{v1}), disparity: {d_min} to {d_max}")
        xyz_coarse, corr_coarse = self.process_epipolar_coarse((u0, u1, v0, v1), (d_min, d_max), k=Kx)
        z_seed_flat = self.resample_to_lattice(xyz_coarse, corr_coarse, min_pts=1)
        z_seed_flat = torch.nan_to_num(z_seed_flat, nan=z_lim[0] + (z_lim[1] - z_lim[0]) / 2)

        Nx, Ny = len(self.x_vals), len(self.y_vals)
        pad_x, pad_y = Kx // 2, Ky // 2
        ix_c = torch.arange(pad_x, Nx - pad_x, stride, device=self.device)
        iy_c = torch.arange(pad_y, Ny - pad_y, stride, device=self.device)
        IX_c, IY_c = torch.meshgrid(ix_c, iy_c, indexing='ij')
        IX_c, IY_c = IX_c.ravel(), IY_c.ravel()
        Nc = IX_c.shape[0]
        x_pts, y_pts = self.x_vals[IX_c], self.y_vals[IY_c]
        z_seed = z_seed_flat[IX_c * Ny + IY_c]

        sz_x = sz_y = None
        if surface_following:
            sz_x, sz_y = self._eval_surface_gradient(IX_c, IY_c, z_seed)

        dz_fine = (self.z_vals[1] - self.z_vals[0]).item() if len(self.z_vals) > 1 else 1.0
        Nz_win = max(3, int(round(2 * fine_window_mm / dz_fine)) + 1)
        Nz_total = len(self.z_vals)
        print(f"[Hybrid C2F] Fine pass — dz={dz_fine:.3f} mm, window=±{fine_window_mm:.1f} mm")

        best_score = torch.full((Nc,), float('inf') if method == 'fringe' else float('-inf'), device=self.device, dtype=torch.float32)
        best_z_idx = torch.zeros(Nc, dtype=torch.long, device=self.device)
        Nz_block_fine = max(1, Nz_win // 3)

        for z0 in range(0, Nz_total, Nz_block_fine):
            z1 = min(z0 + Nz_block_fine, Nz_total)
            z_block = self.z_vals[z0:z1]
            cur_Nz = z_block.shape[0]

            in_win = ((z_block.unsqueeze(0) - z_seed.unsqueeze(1)).abs() <= fine_window_mm)
            if not in_win.any():
                continue

            if surface_following:
                for z_loc in range(cur_Nz):
                    active = in_win[:, z_loc]
                    if not active.any():
                        continue
                    z_pt = torch.full((Nc,), z_block[z_loc].item(), device=self.device, dtype=torch.float32)
                    corr_s = self._score_at_z_per_point(x_pts, y_pts, z_pt, Kx, Ky, method, sz_x, sz_y)
                    fill = float('inf') if method == 'fringe' else float('-inf')
                    corr_s = torch.where(active, corr_s, torch.tensor(fill, device=self.device, dtype=torch.float32))
                    update = (corr_s < best_score) if method == 'fringe' else (corr_s > best_score)
                    best_score[update] = corr_s[update]
                    best_z_idx[update] = z0 + z_loc
            else:
                X2, Y2 = torch.meshgrid(self.x_vals, self.y_vals, indexing='ij')
                XY_flat = torch.stack([X2.ravel(), Y2.ravel()], dim=1)
                Z_col = z_block.repeat_interleave(Nx * Ny).unsqueeze(1)
                XY_exp = XY_flat.unsqueeze(0).expand(cur_Nz, -1, -1).reshape(-1, 2)
                grid_f = torch.cat([XY_exp, Z_col], dim=1)

                uv_L, m_L = self.transform_gcs2ccs(grid_f, 'left',  image_shape=self.left_images.shape[1:])
                uv_R, m_R = self.transform_gcs2ccs(grid_f, 'right', image_shape=self.right_images.shape[1:])

                iL = self.interpolate_images(self.left_images,  uv_L, uv_mask=m_L, method=method)
                iR = self.interpolate_images(self.right_images, uv_R, uv_mask=m_R, method=method)

                iL = iL.view(cur_Nz, Nx, Ny, self.left_images.shape[0]).permute(3, 0, 1, 2)
                iR = iR.view(cur_Nz, Nx, Ny, self.left_images.shape[0]).permute(3, 0, 1, 2)
                L_u = F.unfold(iL.permute(1, 0, 2, 3).reshape(cur_Nz, self.left_images.shape[0], Nx, Ny), kernel_size=(Kx, Ky), stride=(stride, stride))
                R_u = F.unfold(iR.permute(1, 0, 2, 3).reshape(cur_Nz, self.left_images.shape[0], Nx, Ny), kernel_size=(Kx, Ky), stride=(stride, stride))

                L_p = L_u.permute(2, 1, 0)
                R_p = R_u.permute(2, 1, 0)

                if method == 'fringe':
                    corr_all = torch.nan_to_num(torch.abs(L_p[:, (Kx*Ky)//2, :] - R_p[:, (Kx*Ky)//2, :]), nan=float('inf'))
                    fill_val = float('inf')
                else:
                    corr_all = torch.nan_to_num(self._zncc_batch(L_p, R_p), nan=float('-inf'))
                    fill_val = float('-inf')

                corr_all = torch.where(in_win, corr_all, torch.tensor(fill_val, device=self.device, dtype=torch.float32))

                if method == 'fringe':
                    blk_best, blk_idx = corr_all.min(dim=1)
                    update = blk_best < best_score
                else:
                    blk_best, blk_idx = corr_all.max(dim=1)
                    update = blk_best > best_score

                best_score[update] = blk_best[update]
                best_z_idx[update] = z0 + blk_idx[update].long()

        torch.cuda.empty_cache()

        zm1 = self.z_vals[(best_z_idx - 1).clamp(0, Nz_total - 1)]
        zp1 = self.z_vals[(best_z_idx + 1).clamp(0, Nz_total - 1)]
        s_m1 = self._score_at_z_per_point(x_pts, y_pts, zm1, Kx, Ky, method, sz_x, sz_y)
        s_p1 = self._score_at_z_per_point(x_pts, y_pts, zp1, Kx, Ky, method, sz_x, sz_y)
        z_refined = self._parabolic_refine(best_z_idx, best_score, s_m1, s_p1, method)

        xyz_final = torch.stack([x_pts, y_pts, z_refined], dim=1)
        print(f"[Hybrid C2F] Complete in {time.time()-t_start:.2f}s")
        return xyz_final, best_score, best_z_idx

    def process_adaptive_epipolar_to_fine(self, K_list=[7, 5, 3], window_list=[3.0, 1.5, 0.7],
                                          stride: int = 1, method: str = 'correl',
                                          surface_following: bool = True,
                                          corr_threshold: float = 0.5):
        """
        Runs an Adaptive Window Search.
        First does Epipolar Coarse to get the initial Z-seed.
        Then cascades through decreasing Kernel sizes and narrowing Z-windows.
        Results are merged: the smallest kernel result is kept if its correlation is above threshold.
        Otherwise, it falls back to the larger kernel result.
        """
        import time
        t_start = time.time()

        if len(K_list) != len(window_list):
            raise ValueError("K_list and window_list must have the same length")

        if not hasattr(self, 'rect'):
            self.set_rectification()

        x_lim = [self.x_vals[0].item(), self.x_vals[-1].item()]
        y_lim = [self.y_vals[0].item(), self.y_vals[-1].item()]
        z_lim = [self.z_vals[0].item(), self.z_vals[-1].item()]

        c = torch.tensor([
            [x_lim[0], y_lim[0], z_lim[0]], [x_lim[1], y_lim[0], z_lim[0]],
            [x_lim[0], y_lim[1], z_lim[0]], [x_lim[1], y_lim[1], z_lim[0]],
            [x_lim[0], y_lim[0], z_lim[1]], [x_lim[1], y_lim[0], z_lim[1]],
            [x_lim[0], y_lim[1], z_lim[1]], [x_lim[1], y_lim[1], z_lim[1]]
        ], dtype=torch.float32, device=self.device)

        u, v, d, _ = self.world_to_rect(c)
        margin_px = 10
        u0 = max(0, int(torch.floor(u.min()).item()) - margin_px)
        u1 = min(self.left_rect.shape[2], int(torch.ceil(u.max()).item()) + margin_px)
        v0 = max(0, int(torch.floor(v.min()).item()) - margin_px)
        v1 = min(self.left_rect.shape[1], int(torch.ceil(v.max()).item()) + margin_px)

        d_min = int(torch.floor(d.min()).item()) - 2
        d_max = int(torch.ceil(d.max()).item()) + 2

        print(f"[Adaptive C2F] Epipolar Coarse ROI: u({u0}-{u1}) v({v0}-{v1}), disparity: {d_min} to {d_max}")
        xyz_coarse, corr_coarse = self.process_epipolar_coarse((u0, u1, v0, v1), (d_min, d_max), k=K_list[0])
        z_seed_flat = self.resample_to_lattice(xyz_coarse, corr_coarse, min_pts=1)
        z_seed_flat = torch.nan_to_num(z_seed_flat, nan=z_lim[0] + (z_lim[1] - z_lim[0]) / 2)

        Nx, Ny = len(self.x_vals), len(self.y_vals)
        pad_x, pad_y = K_list[0] // 2, K_list[0] // 2
        ix_c = torch.arange(pad_x, Nx - pad_x, stride, device=self.device)
        iy_c = torch.arange(pad_y, Ny - pad_y, stride, device=self.device)
        IX_c, IY_c = torch.meshgrid(ix_c, iy_c, indexing='ij')
        IX_c, IY_c = IX_c.ravel(), IY_c.ravel()
        Nc = IX_c.shape[0]
        x_pts, y_pts = self.x_vals[IX_c], self.y_vals[IY_c]

        z_seed = z_seed_flat[IX_c * Ny + IY_c]
        dz_fine = (self.z_vals[1] - self.z_vals[0]).item() if len(self.z_vals) > 1 else 1.0
        Nz_total = len(self.z_vals)

        results = [] # Store (best_score, best_z_idx) for each level

        for k, fine_window_mm in zip(K_list, window_list):
            print(f"[Adaptive C2F] Fine pass — Kernel: {k}x{k}, window=±{fine_window_mm:.1f} mm")
            sz_x = sz_y = None
            if surface_following:
                sz_x, sz_y = self._eval_surface_gradient(IX_c, IY_c, z_seed)

            Nz_win = max(3, int(round(2 * fine_window_mm / dz_fine)) + 1)
            Nz_block_fine = max(1, Nz_win // 3)

            best_score = torch.full((Nc,), float('inf') if method == 'fringe' else float('-inf'), device=self.device, dtype=torch.float32)
            best_z_idx = torch.zeros(Nc, dtype=torch.long, device=self.device)

            for z0 in range(0, Nz_total, Nz_block_fine):
                z1 = min(z0 + Nz_block_fine, Nz_total)
                z_block = self.z_vals[z0:z1]
                cur_Nz = z_block.shape[0]

                in_win = ((z_block.unsqueeze(0) - z_seed.unsqueeze(1)).abs() <= fine_window_mm)
                if not in_win.any():
                    continue

                if surface_following:
                    for z_loc in range(cur_Nz):
                        active = in_win[:, z_loc]
                        if not active.any():
                            continue

                        # Fix OOM: only process active points, and batch them!
                        active_idx = torch.where(active)[0]
                        active_corr = torch.empty(active_idx.shape[0], device=self.device, dtype=torch.float32)

                        batch_sz = 20000 # Memory safe limit
                        for b_start in range(0, active_idx.shape[0], batch_sz):
                            b_end = min(b_start + batch_sz, active_idx.shape[0])
                            idx_b = active_idx[b_start:b_end]

                            ax = x_pts[idx_b]
                            ay = y_pts[idx_b]
                            az = torch.full((idx_b.shape[0],), z_block[z_loc].item(), device=self.device, dtype=torch.float32)
                            asx = sz_x[idx_b] if sz_x is not None else None
                            asy = sz_y[idx_b] if sz_y is not None else None

                            active_corr[b_start:b_end] = self._score_at_z_per_point(ax, ay, az, k, k, method, asx, asy)

                        fill = float('inf') if method == 'fringe' else float('-inf')
                        corr_s = torch.full((Nc,), fill, device=self.device, dtype=torch.float32)
                        corr_s[active] = active_corr

                        update = (corr_s < best_score) if method == 'fringe' else (corr_s > best_score)
                        best_score[update] = corr_s[update]
                        best_z_idx[update] = z0 + z_loc
                else:
                    X2, Y2 = torch.meshgrid(self.x_vals, self.y_vals, indexing='ij')
                    XY_flat = torch.stack([X2.ravel(), Y2.ravel()], dim=1)
                    Z_col = z_block.repeat_interleave(Nx * Ny).unsqueeze(1)
                    XY_exp = XY_flat.unsqueeze(0).expand(cur_Nz, -1, -1).reshape(-1, 2)
                    grid_f = torch.cat([XY_exp, Z_col], dim=1)

                    uv_L, m_L = self.transform_gcs2ccs(grid_f, 'left',  image_shape=self.left_images.shape[1:])
                    uv_R, m_R = self.transform_gcs2ccs(grid_f, 'right', image_shape=self.right_images.shape[1:])

                    iL = self.interpolate_images(self.left_images,  uv_L, uv_mask=m_L, method=method)
                    iR = self.interpolate_images(self.right_images, uv_R, uv_mask=m_R, method=method)

                    iL = iL.view(cur_Nz, Nx, Ny, self.left_images.shape[0]).permute(3, 0, 1, 2)
                    iR = iR.view(cur_Nz, Nx, Ny, self.left_images.shape[0]).permute(3, 0, 1, 2)
                    L_u = torch.nn.functional.unfold(iL.permute(1, 0, 2, 3).reshape(cur_Nz, self.left_images.shape[0], Nx, Ny), kernel_size=(k, k), stride=(stride, stride))
                    R_u = torch.nn.functional.unfold(iR.permute(1, 0, 2, 3).reshape(cur_Nz, self.left_images.shape[0], Nx, Ny), kernel_size=(k, k), stride=(stride, stride))

                    L_p = L_u.permute(2, 1, 0)
                    R_p = R_u.permute(2, 1, 0)

                    if method == 'fringe':
                        corr_all = torch.nan_to_num(torch.abs(L_p[:, (k*k)//2, :] - R_p[:, (k*k)//2, :]), nan=float('inf'))
                        fill_val = float('inf')
                    else:
                        corr_all = torch.nan_to_num(self._zncc_batch(L_p, R_p), nan=float('-inf'))
                        fill_val = float('-inf')

                    corr_all = torch.where(in_win, corr_all, torch.tensor(fill_val, device=self.device, dtype=torch.float32))

                    if method == 'fringe':
                        blk_best, blk_idx = corr_all.min(dim=1)
                        update = blk_best < best_score
                    else:
                        blk_best, blk_idx = corr_all.max(dim=1)
                        update = blk_best > best_score

                    best_score[update] = blk_best[update]
                    best_z_idx[update] = z0 + blk_idx[update].long()

            torch.cuda.empty_cache()

            # Update z_seed for next cascade pass
            z_seed = self.z_vals[best_z_idx]
            results.append((best_score.clone(), best_z_idx.clone()))

        print("[Adaptive C2F] Merging results...")

        # Merge strategy: start from smallest reliable kernel and fallback to larger ones
        # results are ordered from largest (K=9) to smallest (K=5)
        # We start with the result of the largest kernel as the base.
        final_z_idx = results[0][1].clone()
        final_score = results[0][0].clone()

        for i in range(1, len(results)):
            score, z_idx = results[i]
            # reliable mask: for fringe (MSE), lower is better. for correl (ZNCC), higher is better.
            if method == 'fringe':
                reliable = (score < corr_threshold) & torch.isfinite(score)
            else:
                reliable = (score > corr_threshold) & torch.isfinite(score)

            # Update final maps where this smaller kernel is reliable
            final_z_idx[reliable] = z_idx[reliable]
            final_score[reliable] = score[reliable]

            kept_pct = (reliable.sum().item() / Nc) * 100
            print(f"  > Kernel {K_list[i]} kept for {kept_pct:.1f}% of nodes.")

        final_z = self.z_vals[final_z_idx]
        xyz = torch.stack((x_pts, y_pts, final_z), dim=1)

        t_end = time.time()
        print(f"[Adaptive C2F] Process completed in {t_end - t_start:.2f}s")

        return xyz, final_score, final_score

    def filter_sparse_points(self, xyz_gpu: torch.Tensor, corr_gpu: torch.Tensor, min_neighbors: int = 5, radius: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filtra pontos 3D esparsos com base na densidade de vizinhos.

        Args:
            xyz_gpu (torch.Tensor): Tensor com as coordenadas (N, 3) dos pontos.
            corr_gpu (torch.Tensor): Tensor com os valores de correlação (N,).
            min_neighbors (int): Número mínimo de vizinhos em um raio para um ponto ser mantido.
            radius (float): O raio para a busca de vizinhos.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Um par de tensores (xyz, corr) contendo apenas os pontos densos.
        """
        if xyz_gpu.numel() == 0:
            return xyz_gpu, corr_gpu

        xyz_cpu = xyz_gpu.cpu().numpy()
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz_cpu)

        neighbor_counts = tree.query_ball_point(xyz_cpu, r=radius, return_length=True)
        dense_mask = neighbor_counts >= min_neighbors

        return xyz_gpu[dense_mask], corr_gpu[dense_mask]

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D'):
        import matplotlib.pyplot as plt
        def to_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy()
            return tensor

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(title)

        scatter = ax.scatter(to_numpy(x), to_numpy(y), to_numpy(z), c=to_numpy(color), cmap='viridis', marker='o')
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
