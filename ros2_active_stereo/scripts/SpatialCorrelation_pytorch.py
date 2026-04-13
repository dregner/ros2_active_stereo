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
        # print(f"PyTorch a ser executado no dispositivo: {self.device}")

        self.left_images: torch.Tensor | None = None
        self.right_images: torch.Tensor | None = None
        self.grid: torch.Tensor | None = None
        self.x_vals: torch.Tensor | None = None
        self.y_vals: torch.Tensor | None = None
        self.z_vals: torch.Tensor | None = None

        self.epsilon = 1e-10
        self.camera_params = self.read_yaml_file(yaml_file)

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
            camera_params[cam]['kk'] = torch.tensor(params[f'camera_matrix_{cam}'], dtype=torch.double, device=self.device)
            camera_params[cam]['kc'] = torch.tensor(params[f'dist_coeffs_{cam}'], dtype=torch.double, device=self.device)
            camera_params[cam]['r'] = torch.tensor(params[f'rot_matrix_{cam}'], dtype=torch.double, device=self.device)
            camera_params[cam]['t'] = torch.tensor(params[f't_{cam}'], dtype=torch.double, device=self.device).view(3, 1)
    
        camera_params['stereo']['R'] = torch.tensor(params['R'], dtype=torch.double, device=self.device)
        camera_params['stereo']['T'] = torch.tensor(params['T'], dtype=torch.double, device=self.device).view(3, 1)

        return camera_params

    def convert_images(self, left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))

        def process_image(img, cam_params):
            if apply_clahe:
                img = clahe.apply(img)
            if undist:
                k = cam_params['kk'].cpu().numpy()
                kc = cam_params['kc'].cpu().numpy()
                img = cv2.undistort(img, k, kc)
            return img

        processed_left_imgs = [process_image(img, self.camera_params['left']) for img in left_imgs_cpu]
        processed_right_imgs = [process_image(img, self.camera_params['right']) for img in right_imgs_cpu]

        self.left_images = torch.from_numpy(np.stack(processed_left_imgs, axis=0)).to(self.device, dtype=torch.double)
        self.right_images = torch.from_numpy(np.stack(processed_right_imgs, axis=0)).to(self.device, dtype=torch.double)

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        self.x_vals = torch.arange(x_lim[0], x_lim[1] + xy_step, xy_step, dtype=torch.float16, device=self.device)
        self.y_vals = torch.arange(y_lim[0], y_lim[1] + xy_step, xy_step, dtype=torch.float16, device=self.device)
        self.z_vals = torch.arange(z_lim[0], z_lim[1] + z_step, z_step, dtype=torch.float16, device=self.device)
        
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
        xyz_ccs = torch.matmul(rt_matrix, xyz_gcs_1.T.to(torch.double)).T
        
        zc = xyz_ccs[:, 2]
        valid_mask = zc > self.epsilon
        uv_points = torch.full((num_points, 2), -1.0, device=self.device, dtype=torch.double)
        
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

        corr_map_overall_z = torch.full((Nc_for_xy_plane, Nz_total), -torch.inf, device=self.device, dtype=torch.double)

        for z0_idx in range(0, Nz_total, Nz_block_voxels):
            z1_idx = min(z0_idx + Nz_block_voxels, Nz_total)
            # print(f"[Z-SEGMENT] Processando Z-slice: índices {z0_idx} a {z1_idx-1}")
            
            grid_slice = self.grid[:, :, z0_idx:z1_idx, :].to(torch.double)
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
            corr_map_overall_z= torch.nan_to_num(corr_map_overall_z, nan=100)
            corr_overall, z_best_indices_overall = torch.min(corr_map_overall_z, dim=1)
        else:
            corr_map_overall_z = torch.nan_to_num(corr_map_overall_z, nan=0)
            corr_overall, z_best_indices_overall = torch.max(corr_map_overall_z, dim=1)

        z_best_values_overall = self.z_vals[z_best_indices_overall]
        

        x_coords_final = self.x_vals[IX_centers]
        y_coords_final = self.y_vals[IY_centers]
        
        xyz_final = torch.stack([x_coords_final, y_coords_final, z_best_values_overall], dim=1).to(torch.double)

        return xyz_final, corr_overall, z_best_indices_overall

    def mask_points(self, xyz_gpu: torch.Tensor, corr_gpu: torch.Tensor, bounds, method='correl') -> Tuple[torch.Tensor, torch.Tensor]:

        uv_left_final, uv_left_final_mask = self.transform_gcs2ccs(xyz_gpu, 'left', image_shape=self.left_images.shape[1:])
        uv_right_final, uv_right_final_mask = self.transform_gcs2ccs(xyz_gpu, 'right', image_shape=self.right_images.shape[1:])
        L_interp, R_interp = self.interpolate_images(self.left_images, uv_left_final), self.interpolate_images(self.right_images, uv_right_final)
        print('inter shape: {}, uv shape: {}'.format(L_interp.shape, uv_left_final.shape))
        if method == 'fringe':
            std_mask = (L_interp[:,1]> bounds) & (R_interp[:,1] > bounds)
        else:
            L_std, R_std = L_interp.std(dim=1), R_interp.std(dim=1)
            std_mask = (bounds < L_std) & (bounds < R_std)

        print('mask: {}, std: {}'.format(uv_left_final_mask.shape, std_mask.shape))
        combined_mask = std_mask
        xyz_masked = xyz_gpu[combined_mask]
        corr_masked = corr_gpu[combined_mask]


        return xyz_masked, corr_masked
    
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

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D'):
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