import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import cKDTree
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

        self.epsilon = 1e-10
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

        for cam in ['left', 'right']:
            camera_params[cam]['kk'] = torch.tensor(params[f'camera_matrix_{cam}'], dtype=torch.float32, device=self.device)
            camera_params[cam]['kc'] = torch.tensor(params[f'dist_coeffs_{cam}'], dtype=torch.float32, device=self.device)
            camera_params[cam]['r'] = torch.tensor(params[f'rot_matrix_{cam}'], dtype=torch.float32, device=self.device)
            camera_params[cam]['t'] = torch.tensor(params[f't_{cam}'], dtype=torch.float32, device=self.device).view(3, 1)
    
        camera_params['stereo']['R'] = torch.tensor(params['R'], dtype=torch.float32, device=self.device)
        camera_params['stereo']['T'] = torch.tensor(params['T'], dtype=torch.float32, device=self.device).view(3, 1)

        return camera_params

    def convert_images(self, left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True, tile=1, climp=2.0):
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=climp, tileGridSize=(tile, tile))

        def process_image(img, cam_params):
            if apply_clahe:
                img = clahe.apply(img)
            if undist:
                img = cv2.undistort(img, cam_params['kk'].cpu().numpy(), cam_params['kc'].cpu().numpy())
            return img

        processed_left_imgs = [process_image(img, self.camera_params['left']) for img in left_imgs_cpu]
        processed_right_imgs = [process_image(img, self.camera_params['right']) for img in right_imgs_cpu]

        self.left_images = torch.from_numpy(np.stack(processed_left_imgs, axis=0)).to(self.device, dtype=torch.float32)
        self.right_images = torch.from_numpy(np.stack(processed_right_imgs, axis=0)).to(self.device, dtype=torch.float32)

    def remove_img_distortion(self, img, cam_name):
        """Remove a distorção de uma imagem usando os parâmetros da câmera."""
        k = self.camera_params[cam_name]['kk'].cpu().numpy()
        kc = self.camera_params[cam_name]['kc'].cpu().numpy()
        return cv2.undistort(img, k, kc)
    
    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        self.x_vals = torch.arange(x_lim[0], x_lim[1] + xy_step, xy_step, dtype=torch.float32, device=self.device)
        self.y_vals = torch.arange(y_lim[0], y_lim[1] + xy_step, xy_step, dtype=torch.float32, device=self.device)
        self.z_vals = torch.arange(z_lim[0], z_lim[1] + z_step, z_step, dtype=torch.float32, device=self.device)
        
        X, Y, Z = torch.meshgrid(self.x_vals, self.y_vals, self.z_vals, indexing='ij')
        self.grid = torch.stack((X, Y, Z), axis=-1)

    def transform_gcs2ccs(self, points_3d, cam_name, image_shape=None):
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
        torch.cat([rt_matrix, torch.tensor([[0, 0, 0, 1]], device=self.device)], dim=0)
        xyz_ccs = torch.matmul(rt_matrix, xyz_gcs_1.T.to(torch.float32)).T
        
        zc = xyz_ccs[:, 2]
        valid_mask = zc > self.epsilon
        uv_points = torch.full((num_points, 2), -1.0, device=self.device, dtype=torch.float32)
        
        if torch.any(valid_mask):
            xn = xyz_ccs[valid_mask, 0] / zc[valid_mask]
            yn = xyz_ccs[valid_mask, 1] / zc[valid_mask]

            xyz_ccs = torch.matmul(k, torch.stack([xn, yn, torch.ones_like(xn)], dim=0))
            
            uv_points[valid_mask] = xyz_ccs[:2, :].T
           

        if image_shape is not None:
            H, W = image_shape
            # Mask for points inside image boundaries
            inside_mask = (
                (uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) &
                (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H)
            )
            return uv_points, inside_mask
        else:
            return uv_points

    def interpolate_images(self, images, uv_points, uv_mask=None):
        if uv_points.numel() == 0:
            return torch.empty((0, images.shape[0]), device=self.device)
        
        T, H, W = images.shape
        N = uv_points.shape[0]

        u_norm = (uv_points[:, 0] / (W - 1)) * 2 - 1
        v_norm = (uv_points[:, 1] / (H - 1)) * 2 - 1
        
        grid = torch.stack([u_norm, v_norm], dim=1).view(1, N, 1, 2)
        images_batch = images.unsqueeze(0)

        interpolated = F.grid_sample(images_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        interpolated = interpolated.view(T, N).T

        if uv_mask is not None:
                interpolated[~uv_mask] = torch.nan


        return interpolated

    def phase_map_difference(self, L_patches, R_patches):
        #Compute absolute difference between left and right patches

        return torch.abs(L_patches - R_patches)[:,0]#, dim=1, keepdim=True)  # Keep the channel dimension for consistency

    def zncc_correlation(self, L_patches, R_patches):
        L_mean = torch.mean(L_patches, dim=1, keepdim=True)
        R_mean = torch.mean(R_patches, dim=1, keepdim=True)
        L_centered = L_patches - L_mean
        R_centered = R_patches - R_mean

        numerator = torch.sum(L_centered * R_centered, dim=1)
        denom_L = torch.sum(L_centered**2, dim=1)
        denom_R = torch.sum(R_centered**2, dim=1)
        denominator = torch.sqrt(denom_L) * torch.sqrt(denom_R)
        
        return numerator / torch.max(denominator, torch.tensor(1e-10))

    def process_segmented_z(self, Kx, Ky, stride=1, Nz_block_voxels=40, method='correl'):
        Nx, Ny, Nz_total = self.grid.shape[:3]
        T = self.left_images.shape[0]
        
        pad_x, pad_y = Kx // 2, Ky // 2
        ix_centers = torch.arange(pad_x, Nx - pad_x, stride, device=self.device)
        iy_centers = torch.arange(pad_y, Ny - pad_y, stride, device=self.device)

        if len(ix_centers) == 0 or len(iy_centers) == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

        IX_centers, IY_centers = torch.meshgrid(ix_centers, iy_centers, indexing='ij')
        IX_centers, IY_centers = IX_centers.ravel(), IY_centers.ravel()
        Nc_for_xy_plane = IX_centers.shape[0]

        corr_map_overall_z = torch.full((Nc_for_xy_plane, Nz_total), -torch.inf, device=self.device, dtype=torch.float32)

        for z0_idx in range(0, Nz_total, Nz_block_voxels):
            z1_idx = min(z0_idx + Nz_block_voxels, Nz_total)
            # print(f"[Z-SEGMENT] Processando Z-slice: índices {z0_idx} a {z1_idx-1}")
            
            grid_slice = self.grid[:, :, z0_idx:z1_idx, :].to(torch.float32)
            current_Nz_in_slice = grid_slice.shape[2]

            grid_flat_xy = grid_slice.permute(2,0,1,3).reshape(current_Nz_in_slice, Nx*Ny, 3)
            
            uv_left, uv_left_mask = self.transform_gcs2ccs(grid_flat_xy.reshape(-1, 3), 'left', image_shape=self.left_images.shape[1:])
            uv_right, uv_right_mask = self.transform_gcs2ccs(grid_flat_xy.reshape(-1, 3), 'right', image_shape=self.right_images.shape[1:])

            interp_L = self.interpolate_images(self.left_images, uv_left, uv_mask=uv_left_mask)
            interp_R = self.interpolate_images(self.right_images, uv_right, uv_mask=uv_right_mask)
            del uv_left, uv_right, uv_left_mask, uv_right_mask
            torch.cuda.empty_cache()

            interp_L = interp_L.view(current_Nz_in_slice, Nx, Ny, T).permute(3,0,1,2)
            interp_R = interp_R.view(current_Nz_in_slice, Nx, Ny, T).permute(3,0,1,2)

            L_unfold = F.unfold(interp_L.permute(1,0,2,3).reshape(current_Nz_in_slice, T, Nx, Ny), kernel_size=(Kx, Ky), stride=(stride, stride))
            R_unfold = F.unfold(interp_R.permute(1,0,2,3).reshape(current_Nz_in_slice, T, Nx, Ny), kernel_size=(Kx, Ky), stride=(stride, stride))

            L_patches = L_unfold.permute(2, 1, 0).reshape(Nc_for_xy_plane, -1, current_Nz_in_slice)
            R_patches = R_unfold.permute(2, 1, 0).reshape(Nc_for_xy_plane, -1, current_Nz_in_slice)

            del interp_L, interp_R, L_unfold, R_unfold
            torch.cuda.empty_cache()


            for z_local_idx in range(current_Nz_in_slice):
                if method == 'fringe':
                    corr_slice = self.phase_map_difference(L_patches[:,:,z_local_idx], R_patches[:,:,z_local_idx])
                else:
                    corr_slice = self.zncc_correlation(L_patches[:,:,z_local_idx], R_patches[:,:,z_local_idx])

                corr_map_overall_z[:, z0_idx + z_local_idx] = corr_slice

        if method == 'fringe':
            corr_overall, z_best_indices_overall = torch.min(torch.nan_to_num(corr_map_overall_z, nan=1), dim=1)
            z_best_values_overall = self.z_vals[z_best_indices_overall]
        else:
            corr_overall, z_best_indices_overall = torch.max(torch.nan_to_num(corr_map_overall_z, nan=0), dim=1)
            z_best_values_overall = self.z_vals[z_best_indices_overall]
        

        x_coords_final = self.x_vals[IX_centers]
        y_coords_final = self.y_vals[IY_centers]
        
        xyz_final = torch.stack([x_coords_final, y_coords_final, z_best_values_overall], dim=1).to(torch.float32)

        # Project final points to both image planes and check bounds

        return xyz_final, corr_overall, z_best_indices_overall

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

        uv_left_final, _ = self.transform_gcs2ccs(xyz_gpu, 'left', image_shape=self.left_images.shape[1:])
        uv_right_final, _ = self.transform_gcs2ccs(xyz_gpu, 'right', image_shape=self.right_images.shape[1:])
        L_interp, R_interp = self.interpolate_images(self.left_images, uv_left_final), self.interpolate_images(self.right_images, uv_right_final)

        if method == 'fringe':
            std_mask = (L_interp[:,1]> bounds) & (R_interp[:,1] > bounds)
            L_masked = L_interp[:, 1]
        else:
            L_std, R_std = L_interp.std(dim=1), R_interp.std(dim=1)
            std_mask = (bounds < L_std) & (bounds < R_std)
            L_masked = L_std

        combined_mask = std_mask
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

            # Opcional: Limpa a memória para liberar a GPU
            del dist_batch, mask_dist, neighbors_count, mask_neighbors
            torch.cuda.empty_cache()
            
        if corr_gpu is None and interp is None:
            return xyz_gpu[final_mask], None, None
        
        elif corr_gpu is None and interp is not None:
            return xyz_gpu[final_mask], None,interp[final_mask]
        
        elif corr_gpu is not None and interp is None:
            return xyz_gpu[final_mask], corr_gpu[final_mask], None
        else:
            # Aplica a máscara final aos tensores originais
            return xyz_gpu[final_mask], corr_gpu[final_mask], interp[final_mask]

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
        tree = cKDTree(xyz_cpu)
    
        neighbor_counts = tree.query_ball_point(xyz_cpu, r=radius, return_length=True)
        dense_mask = neighbor_counts >= min_neighbors

        return xyz_gpu[dense_mask], corr_gpu[dense_mask]