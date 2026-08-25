import gc
import cupy as cp
import matplotlib.pyplot as plt
import logging
import numpy as np
import open3d as o3d
import cv2
import yaml
from scipy.spatial import cKDTree
from cupyx.fallback_mode.fallback import ndarray

class InverseTriangulation:
    # def __init__(self, left_camera_info, right_camera_info):
    def __init__(self, yaml_file):

        self.left_images = cp.array([])
        self.right_images = cp.array([])
        self.left_mask = cp.array([])
        self.right_mask = cp.array([])

        self.z_scan_step = None
        self.num_points = None

        # self.left_camera_info = left_camera_info
        # self.right_camera_info = right_camera_info

        # Initialize all camera parameters in a single nested dictionary
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        # self.set_camera_info()
        self.read_yaml_file(yaml_file)

        self.max_gpu_usage = self.set_datalimit() // 3

        logging.basicConfig(level=logging.INFO)

    def read_yaml_file(self, yaml_file):
        """
        Read YAML file to extract cameras parameters
        """
        # Load the YAML file
        with open(yaml_file) as file:  # Replace with your file path
            params = yaml.safe_load(file)

            # Parse the matrices
        self.camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
        self.camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
        self.camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
        self.camera_params['left']['t'] = np.array(params['t_left'], dtype=np.float64)

        self.camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
        self.camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
        self.camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
        self.camera_params['right']['t'] = np.array(params['t_right'], dtype=np.float64)

        self.camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
        self.camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)

    def read_images(self, left_imgs, right_imgs, left_mask, right_mask):
        if len(left_imgs) != len(right_imgs):
            raise Exception("Number of images do not match")
        self.left_images = cp.asarray(left_imgs)
        self.right_images = cp.asarray(right_imgs)
        self.left_mask = cp.asarray(left_mask)
        self.right_mask = cp.asarray(right_mask)

    def set_camera_info(self):
        """
        Configura os parâmetros das câmeras usando as mensagens CameraInfo.
        """
        # Parâmetros da câmera esquerda
        self.camera_params['left'] = {
            'kk': np.array(self.left_camera_info.k).reshape(3, 3),  # Matriz intrínseca
            'kc': np.array(self.left_camera_info.d),  # Coeficientes de distorção
            'r': np.array(self.left_camera_info.r).reshape(3, 3),  # Matriz de retificação
            't': np.array(self.left_camera_info.p).reshape(3, 4)[:, 3] # Translação para uma única câmera
        }

        # Parâmetros da câmera direita
        self.camera_params['right'] = {
            'kk': np.array(self.right_camera_info.k).reshape(3, 3),  # Matriz intrínseca
            'kc': np.array(self.right_camera_info.d),  # Coeficientes de distorção
            'r': np.array(self.right_camera_info.r).reshape(3, 3),  # Matriz de retificação
            't': np.array(self.right_camera_info.p).reshape(3, 4)[:, 3]  # Translação para uma única câmera
        }

    def points3d_zstep(self, x_lim=(-5, 5), y_lim=(-5, 5), xy_step=1.0, z_lin=np.arange(0, 100, 0.1)):
        """
            Create a 3D space of combination from linear arrays of X Y Z
            Parameters:
                x_lim: Begin and end of linear space of X
                y_lim: Begin and end of linear space of Y
                z_lin: numpy array of z to be tested
                xy_step: Step size between X and Y
            Returns:
                cube_points: combination of X Y and Z
        """
        x_lin = np.arange(x_lim[0], x_lim[1], xy_step)
        y_lin = np.arange(y_lim[0], y_lim[1], xy_step)

        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

        c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        self.num_points = c_points.shape[0]
        self.z_scan_step = np.unique(c_points[:, 2]).shape[0]

        return c_points

    def points3D_arrays(self, x_lin: ndarray, y_lin: ndarray, z_lin: ndarray) -> ndarray:
        """
        Crete 3D meshgrid of points based on input vectors of x, y and z
        :param x_lin: linear space of x points
        :param y_lin: linear space of y points
        :param z_lin: linear space of z points
        :return: 3D meshgrid points size (N,3) where N = len(x)*len(y)*len(z)
        """
        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
        points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        self.num_points = points.shape[0]
        self.z_scan_step = np.unique(points[:, 2]).shape[0]

        return cp.asarray(points)

    def points3d(self, xlim, ylim, zlim, xy_step, z_step):
        """
        Build full 3D points grid (no memory explosion, sliding kernels will select parts).
        """
        x_lin = np.arange(xlim[0], xlim[1] + xy_step, xy_step)
        y_lin = np.arange(ylim[0], ylim[1] + xy_step, xy_step)
        z_lin = np.arange(zlim[0], zlim[1] + z_step, z_step)

        X, Y, Z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
        points = np.stack((X, Y, Z), axis=-1)  # shape (Nx, Ny, Nz, 3)

        # self.x_vals = cp.asarray(x_lin)
        # self.y_vals = cp.asarray(y_lin)
        # self.z_vals = cp.asarray(z_lin)

        return cp.asarray(points, dtype=cp.float16)  # shape (Nx, Ny, Nz, 3)

    def set_datalimit(self):
        """
        Identify gpu limit
        """
        # Create a device object for the first GPU (device ID 0)
        device_id = 0
        cp.cuda.Device(device_id).use()  # Set the current device
        # Get the total memory in bytes using runtime API
        total_memory = cp.cuda.runtime.getDeviceProperties(device_id)['totalGlobalMem']
        # Convert bytes to GB
        return total_memory / (1024 ** 3)

    def remove_img_distortion(self, image, camera):
        return cv2.undistort(image, self.camera_params[camera]['kk'], self.camera_params[camera]['kc'])

    def transform_gcs2ccs(self, points_3d, cam_name):
        """
        Transform Global Coordinate System (xg, yg, zg)
        to Camera's Coordinate System (xc, yc, zc) and transform to Image's plane (uv)
        Returns:
            uv_image_points: (2,N) reprojected points to image's plane
        """
        
        # Convert all inputs to CuPy arrays for GPU computation
        xyz_gcs = cp.asarray(points_3d)
        k = cp.asarray(self.camera_params[cam_name]['kk'])
        dist = cp.asarray(self.camera_params[cam_name]['kc'])
        rot = cp.asarray(self.camera_params[cam_name]['r'])
        tran = cp.asarray(self.camera_params[cam_name]['t'])

        # Estimate memory required for processing
        bytes_per_float32 = 8
        memory_per_point = (4 * 3 * bytes_per_float32) + (3 * bytes_per_float32)
        total_memory_required =  int(points_3d.shape[0]) * int(memory_per_point)

        # Adjust batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(
                (self.max_gpu_usage * 1024 ** 3 // memory_per_point) // 10)  # Reduce batch size more aggressively
            # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
        else:
            points_per_batch = self.num_points  # Process all points at once

        # Initialize a list to store results on the GPU
        uv_points_list = cp.empty((2, xyz_gcs.shape[0]), dtype=np.float32)

        # Process points in batches
        for i in range(0, self.num_points, points_per_batch):
            end = min(i + points_per_batch, self.num_points)
            xyz_gcs_batch = xyz_gcs[i:end]

            # Add one extra line of ones to the global coordinates
            ones = cp.ones((xyz_gcs_batch.shape[0], 1), dtype=cp.float32)
            xyz_gcs_1 = cp.hstack((xyz_gcs_batch, ones))

            # Create the rotation and translation matrix
            rt_matrix = cp.vstack(
                (cp.hstack((rot, tran[:, None])), cp.array([0, 0, 0, 1], dtype=cp.float32))
            )

            # Multiply the RT matrix with global points [X; Y; Z; 1]
            xyz_ccs = cp.dot(rt_matrix, xyz_gcs_1.T)
            del xyz_gcs_1

            # Normalize by dividing by Z to get normalized image coordinates
            epsilon = 1e-10  # Small value to prevent division by zero
            xyz_ccs_norm = cp.hstack(
                (xyz_ccs[:2, :].T / cp.maximum(xyz_ccs[2, :, cp.newaxis], epsilon),
                 cp.ones((xyz_ccs.shape[1], 1), dtype=cp.float32))
            ).T
            del xyz_ccs


            # Compute image points using the intrinsic matrix K
            uv_points_batch = cp.dot(k, xyz_ccs_norm).astype(cp.float32)
            del xyz_ccs_norm  # Free memory

            # Transfer results back to CPU after processing each batch
            uv_points_list[:, i:end] = uv_points_batch[:2, :]

            # Free GPU memory after processing each batch
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Transfer final result to CPU in a single operation
        return uv_points_list

    def bi_interpolation(self, images, modulation_map, uv_points):
        """
        Perform bilinear interpolation on a stack of images at specified uv_points on the GPU.

        Parameters:
        ----------
        images : (height, width, num_images) array or (height, width) for a single image.
        uv_points : (2, N) array of UV points where N is the number of points.

        Returns:
        -------
        interpolated_cpu : np.ndarray
            Interpolated pixel values for each point.
        std_cpu : np.ndarray
            Standard deviation of the corner pixels used for interpolation.
        """
        images = cp.asarray(images, dtype=cp.float32)
        uv_points = cp.asarray(uv_points, dtype=cp.float32)
        mod_threshold = 0.01 * cp.max(modulation_map)
        mod_threshold = cp.float64(mod_threshold)

        if len(images.shape) == 2:  # Convert single image to a stack with one image
            images = images[:, :, cp.newaxis]

        height, width, num_images = images.shape

        # Estimate memory usage per point
        memory_per_point = 4 * num_images * 4
        points_per_batch = max(1, int(self.max_gpu_usage * 1024 ** 3 // memory_per_point))

        # Output arrays on GPU
        interpolated = cp.zeros((uv_points.shape[1], num_images), dtype=cp.float32)
        std = cp.zeros((uv_points.shape[1], num_images), dtype=cp.float32)

        for i in range(0, uv_points.shape[1], points_per_batch):
            end = min(i + points_per_batch, uv_points.shape[1])
            uv_batch = uv_points[:, i:end]

            # Compute integer and fractional parts of UV coordinates
            x = uv_batch[0].astype(cp.float32)
            y = uv_batch[1].astype(cp.float32)

            x1 = cp.clip(cp.floor(x).astype(cp.int32), 0, width - 1)
            y1 = cp.clip(cp.floor(y).astype(cp.int32), 0, height - 1)
            x2 = cp.clip(x1 + 1, 0, width - 1)
            y2 = cp.clip(y1 + 1, 0, height - 1)

            x_diff = x - x1
            y_diff = y - y1
            for k in range(num_images):
                # Vectorized extraction of corner pixels
                p11 = images[y1, x1, k]  # Top-left
                p12 = images[y2, x1, k]  # Bottom-left
                p21 = images[y1, x2, k]  # Top-right
                p22 = images[y2, x2, k]  # Bottom-right

                # Modulation map values
                mod_p11 = modulation_map[y1, x1]
                mod_p12 = modulation_map[y2, x1]
                mod_p21 = modulation_map[y1, x2]
                mod_p22 = modulation_map[y2, x2]

                 # Check if all corner modulations are above the threshold - Remove points with less than 1% of the modulation map value
                if cp.all(cp.array([mod_p11, mod_p12, mod_p21, mod_p22]) < mod_threshold):
                    # If any modulation is below the threshold, discard or adjust interpolation
                    interpolated[i:end, k] = cp.nan # You can replace with NaN or other value
                    std[i:end, k] = cp.nan  # Reset the standard deviation too
                else:
                    # Bilinear interpolation
                    interpolated_batch = (
                            p11 * (1 - x_diff) * (1 - y_diff) +
                            p21 * x_diff * (1 - y_diff) +
                            p12 * (1 - x_diff) * y_diff +
                            p22 * x_diff * y_diff
                    )

                    std_batch = cp.std(cp.vstack([p11, p12, p21, p22]), axis=0)

                    # Store results in GPU arrays
                    interpolated[i:end, k] = interpolated_batch
                    std[i:end, k] = std_batch

                    del p11, p12, p21, p22, std_batch, interpolated_batch

                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

        return interpolated, std

    def phase_map(self, interp_left, interp_right):
        """
        Identify minimum phase map value.
        Parameters:
            interp_left: left interpolated points (1D array, cupy.ndarray)
            interp_right: right interpolated points (1D array, cupy.ndarray)
        Returns:
            phi_min_id: indices of minimum phase map values (cupy.ndarray).
        """

        # Se forem (Nc, Nz), o diff já tem o shape certo
        diff_phi = cp.abs(interp_left - interp_right)  # shape: (Nc, Nz)

        # Reshape the array for efficient block processing
        diff_phi_blocks = diff_phi.reshape(-1, self.z_scan_step)

        # Find the indices of minimum values within each block
        block_min_indices = cp.argmin(diff_phi_blocks, axis=1)

        # Adjust indices to account for the block position
        phi_min_id = block_min_indices + cp.arange(len(block_min_indices)) * self.z_scan_step


        return phi_min_id
    
    def fringe_masks(self, uv_l, uv_r, std_l, std_r, phi_id, min_thresh=0, max_thresh=0.12, mod_thresh=0.01):
        """
        Mask from fringe process to remove outbounds points.
        Parameters:
            std_l: STD interpolation image's points
            std_r: STD interpolation image's points
            phi_id: Indices for min phase difference
            min_thresh: max threshold for STD
            max_thresh: min threshold for STD
        Returns:
             valid_mask: Valid 3D points on image's plane
        """
        # converte as coordenadas em um array cupy
        uv_l = cp.asarray(uv_l, dtype=cp.float32)
        uv_r = cp.asarray(uv_r, dtype=cp.float32)

        # Verifica se as coordenadas estão dentro dos limites das máscaras
        valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < self.left_mask.shape[1])
        valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < self.left_mask.shape[0])
        valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < self.right_mask.shape[1])
        valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < self.right_mask.shape[0])

        # Aplica as verificações de validade nas coordenadas UV para evitar indexação fora do limite
        valid_uv_l = valid_u_l & valid_v_l
        valid_uv_r = valid_u_r & valid_v_r

        # Verifica os pontos válidos nas máscaras (aplica as coordenadas para obter as máscaras)
        valid_uv_l &= (self.left_mask[uv_l[1, :].clip(0, self.left_mask.shape[0] - 1).astype(int), uv_l[0, :].clip(0, self.left_mask.shape[1] - 1).astype(int)] > (0.07 * cp.max(self.left_mask)))
        valid_uv_r &= (self.right_mask[uv_r[1, :].clip(0, self.right_mask.shape[0] - 1).astype(int), uv_r[0, :].clip(0, self.right_mask.shape[1] - 1).astype(int)] > (0.07 * cp.max(self.right_mask)))

        # Combine as verificações dos limites
        valid_uv = valid_uv_r & valid_uv_l

        # Máscara para `phi_id`
        phi_mask = cp.zeros(uv_l.shape[1], dtype=bool)
        phi_mask[phi_id] = True

        # Verificação dos thresholds de `std` para pontos válidos
        valid_l = (min_thresh < std_l) & (std_l < max_thresh)
        valid_r = (min_thresh < std_r) & (std_r < max_thresh)
        valid_std = valid_r[:, 0] & valid_l[:, 0]

        std_mod_l = cp.std(self.left_mask)
        std_mod_r = cp.std(self.right_mask)

        mod_threshold_l = mod_thresh * cp.max(self.left_mask)
        mod_threshold_r = mod_thresh * cp.max(self.right_mask)

        valid_mod_l = (0 < std_mod_l) & (std_mod_l < mod_threshold_l)
        valid_mod_r = (0 < std_mod_r) & (std_mod_r < mod_threshold_r)
        valid_mod = valid_mod_l & valid_mod_r

        # Retorne a máscara final considerando os pontos válidos em `uv`, `phi` e `std`
        mask = valid_uv & phi_mask & valid_std 

        return mask

    def fringe_process(self, points_3d: ndarray, mod_thresh, batch=1) -> ndarray:
        """
        Zscan for stereo fringe process over points in shape (Nc, Nz, 3).
        Parameters:
            points_3d: ndarray of shape (Nc, Nz, 3)
            mod_thresh: modulation threshold
        Returns:
            measured_pts: Valid 3D points (N_valid, 3)
        """
        # Estimar consumo de memória (conservador)
        bytes_per_point = 3 * 8  # float64, 3 coords
        Nz = points_3d.shape[1]
        bytes_per_batch = bytes_per_point * Nz

        batch_size = int((self.max_gpu_usage * 1024 ** 3) // bytes_per_batch // batch)  # margem de segurança

        Nc = points_3d.shape[0]
        measured_pts_list = []

        for i in range(0, Nc, batch_size):
            end = min(i + batch_size, Nc)

            # Seleciona o batch e achata para (Nb*Nz, 3)
            batch = points_3d[i:end]
            self.z_scan_step = Nz                  # (Nb, Nz, 3)
            batch_points = cp.asarray(batch.reshape(-1, 3))  # (Nb * Nz, 3)
            self.num_points = batch_points.shape[0]

            # Projeções
            uv_left = self.transform_gcs2ccs(batch_points, cam_name='left')
            uv_right = self.transform_gcs2ccs(batch_points, cam_name='right')

            # Interpolações bilineares
            inter_left, std_left = self.bi_interpolation(self.left_images, self.left_mask, uv_left)
            inter_right, std_right = self.bi_interpolation(self.right_images, self.right_mask, uv_right)

            # Cálculo de fase
            phi_min_id = self.phase_map(inter_left, inter_right)

            # Máscara de validade
            mask = self.fringe_masks(uv_l = uv_left, uv_r = uv_right, std_l = std_left, std_r = std_right, phi_id = phi_min_id, mod_thresh=mod_thresh)

            # Seleciona pontos válidos
            valid_points = batch_points[mask]  # (N_valid, 3)
            measured_pts_list.append(valid_points)

            # Libera memória
            del uv_left, uv_right, inter_left, inter_right
            del std_left, std_right, phi_min_id, mask, batch_points
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # # Concatena resultados
        # if measured_pts_list:
        measured_pts = np.concatenate(measured_pts_list, axis=0)
        # else:
        #     measured_pts = np.empty((0, 3))

        return measured_pts

    def filter_sparse_points(self, xyz, min_neighbors=5, radius=10):
        """
        Remove sparse points from a 3D point cloud based on spatial density.

        Parameters:
        ----------
        xyz : np.ndarray
            3D points of shape (N, 3).
        corr : np.ndarray
            Correlation values of shape (N,).
        min_neighbors : int
            Minimum number of neighbors required to keep a point.
        radius : float
            Radius within which to count neighbors.

        Returns:
        -------
        filtered_xyz : np.ndarray
            Filtered 3D points.
        filtered_corr : np.ndarray
            Correlation values corresponding to the filtered points.
        """

        # Converte para NumPy, se necessário
        if isinstance(xyz, cp.ndarray):
            xyz = cp.asnumpy(xyz)

        # Build a KD-tree for fast neighbor search
        tree = cKDTree(xyz)

        # Query the number of neighbors within the radius for each point
        neighbor_counts = tree.query_ball_point(xyz, r=radius)
        neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_counts])

        # Create a mask for points with sufficient neighbors
        dense_mask = neighbor_counts >= min_neighbors

        # Filter points
        filtered_xyz = xyz[dense_mask]

        return filtered_xyz
    
    def process_blocks(self, x_range, y_range, z_range, num_splits):
            """Divide o espaço em blocos e processa cada um."""
            x_split = cp.array_split(x_range, num_splits)
            y_split = cp.array_split(y_range, num_splits)
            points_result = []

            # Processa cada bloco
            for x_arr in x_split:
                for y_arr in y_split:
                    points_3d = self.points3D_arrays(x_arr, y_arr, z_range)
                    z_zcan_points = self.fringe_process(points_3d=points_3d)
                    points_result.append(z_zcan_points)
                    
            points_result = cp.concatenate(points_result, axis=0)

            return points_result