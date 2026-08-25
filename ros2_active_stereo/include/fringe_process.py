import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple

# try:
from fringe_pattern import FringePattern
from gray_code import GrayCode
# except (ImportError, ValueError):
#     from fringe_pattern import FringePattern
#     from gray_code import GrayCode


class FringeProcess(GrayCode, FringePattern):

    def __init__(self, img_resolution=(1920, 1080), camera_resolution=(2448, 2048), px_f=12, steps=12, device: Optional[Union[torch.device, str]] = None):
        """
                   Inicializa uma instância da classe com parâmetros específicos de resolução e configuração.
                   Este método inicializa as variáveis necessárias para o processamento de imagens, bem como imagens capturadas pela câmera.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_min_bits = self.min_bits_gc(math.floor(img_resolution[0] / px_f)) + 2
        total_channels = int(steps + self.n_min_bits)

        self.images_left = torch.zeros((camera_resolution[1], camera_resolution[0], total_channels),
                                       dtype=torch.float32, device=self.device)
        self.images_right = torch.zeros((camera_resolution[1], camera_resolution[0], total_channels),
                                        dtype=torch.float32, device=self.device)

        self.steps = steps

        # Armazenamento de resultados intermediários e finais
        self.phi_image_left = None
        self.phi_image_right = None
        self.modulation_map_left = None
        self.modulation_map_right = None
        self.qsi_image_left = None
        self.qsi_image_right = None
        self.remaped_qsi_image_left = None
        self.remaped_qsi_image_right = None
        self.abs_phi_image_left = None
        self.abs_phi_image_right = None

        FringePattern.__init__(self, resolution=img_resolution, px_f=px_f, steps=steps)
        GrayCode.__init__(self, resolution=img_resolution, n_bits=self.min_bits_gc(math.floor(img_resolution[0] / px_f)),
                          px_f=px_f)

    def to(self, device: Union[torch.device, str]):
        """
            Transfere os tensores da classe para o dispositivo especificado (e.g., 'cuda' ou 'cpu').
        """
        self.device = torch.device(device)
        self.images_left = self._to_tensor(self.images_left)
        self.images_right = self._to_tensor(self.images_right)
        if self.phi_image_left is not None:
            self.phi_image_left = self._to_tensor(self.phi_image_left)
        if self.phi_image_right is not None:
            self.phi_image_right = self._to_tensor(self.phi_image_right)
        if self.modulation_map_left is not None:
            self.modulation_map_left = self._to_tensor(self.modulation_map_left)
        if self.modulation_map_right is not None:
            self.modulation_map_right = self._to_tensor(self.modulation_map_right)
        if self.qsi_image_left is not None:
            self.qsi_image_left = self._to_tensor(self.qsi_image_left)
        if self.qsi_image_right is not None:
            self.qsi_image_right = self._to_tensor(self.qsi_image_right)
        if self.remaped_qsi_image_left is not None:
            self.remaped_qsi_image_left = self._to_tensor(self.remaped_qsi_image_left)
        if self.remaped_qsi_image_right is not None:
            self.remaped_qsi_image_right = self._to_tensor(self.remaped_qsi_image_right)
        if self.abs_phi_image_left is not None:
            self.abs_phi_image_left = self._to_tensor(self.abs_phi_image_left)
        if self.abs_phi_image_right is not None:
            self.abs_phi_image_right = self._to_tensor(self.abs_phi_image_right)
        return self

    def _to_tensor(self, x, dtype=None) -> Optional[torch.Tensor]:
        """
            Converte a entrada (numpy.ndarray, torch.Tensor, list, etc.) para torch.Tensor no dispositivo configurado.
        """
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            if dtype is not None and x.dtype != dtype:
                return x.to(device=self.device, dtype=dtype)
            return x.to(device=self.device)
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x).to(device=self.device)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype=dtype)
            return t
        else:
            return torch.tensor(x, device=self.device, dtype=dtype)

    def min_bits_gc(self, x):
        """
                    Calcula o número mínimo de bits necessários para representar um número em código Gray.
                    Parameters:
                    -----------
                    x : int
                        Número positivo para o qual se deseja calcular o número mínimo de bits necessários.
                    Returns:
                    --------
                    int
                        Número mínimo de bits necessários para representar o número `x` em código Gray.
        """
        if x <= 0:
            raise ValueError("Input must be a positive integer.")
        return math.ceil(math.log2(x) + 1)

    def normalize_white(self, mask_left, mask_right):
        """
            Calcula a média dos valores máximos dos pixels brancos nas imagens esquerda e direita usando máscaras.

            Parameters:
            -----------
            mask_left : numpy.ndarray | torch.Tensor
                Máscara binária aplicada à imagem esquerda, onde os pixels de interesse são marcados com o valor 255.

            mask_right : numpy.ndarray | torch.Tensor
                Máscara binária aplicada à imagem direita, onde os pixels de interesse são marcados com o valor 255.

            Returns:
            --------
            media_branco_max_left : float
                Média dos valores máximos dos pixels brancos na imagem esquerda, calculada a partir da máscara.

            media_branco_max_right : float
                Média dos valores máximos dos pixels brancos na imagem direita, calculada a partir da máscara.
        """
        img_l = self._to_tensor(self.images_left, dtype=torch.float32)
        img_r = self._to_tensor(self.images_right, dtype=torch.float32)
        m_left = self._to_tensor(mask_left)
        m_right = self._to_tensor(mask_right)

        white_pixels_left = img_l[:, :, self.steps][m_left == 255]
        white_pixels_right = img_r[:, :, self.steps][m_right == 255]

        media_branco_max_left = torch.mean(white_pixels_left).item() if white_pixels_left.numel() > 0 else 0.0
        media_branco_max_right = torch.mean(white_pixels_right).item() if white_pixels_right.numel() > 0 else 0.0

        # print("media dos brancos right:", media_branco_max_right)

        return media_branco_max_left, media_branco_max_right

    def set_images(self, image_left, image_right, counter):
        """
            Atribui imagens para os índices especificados nas matrizes de imagens esquerda e direita.
        """
        target_dtype = self.images_left.dtype if isinstance(self.images_left, torch.Tensor) else torch.float32
        img_l = self._to_tensor(image_left, dtype=target_dtype)
        img_r = self._to_tensor(image_right, dtype=target_dtype)

        if not isinstance(self.images_left, torch.Tensor):
            self.images_left = self._to_tensor(self.images_left)
        if not isinstance(self.images_right, torch.Tensor):
            self.images_right = self._to_tensor(self.images_right)

        self.images_left[:, :, counter] = img_l
        self.images_right[:, :, counter] = img_r

    def calculate_phi(self, image, name='Plot', visualize=True):
        """
            Calcula a imagem de fase (phi) a partir de uma imagem de múltiplos canais utilizando transformações senoidais e cossenoidais.

            Esta função processa uma imagem composta por múltiplos canais (por exemplo, imagens obtidas através de projeções de padrões de fase)
            e calcula a fase correspondente para cada pixel. O cálculo é realizado aplicando funções seno e cosseno aos canais da imagem
            e combinando esses valores para obter a fase através da função arctan2.

            Parameters:
            -----------
            image : np.ndarray | torch.Tensor
                Uma matriz tridimensional (altura, largura, canais), onde cada canal representa uma amostra
                de fase em diferentes momentos ou ângulos. A função espera que os canais sejam organizados em uma
                sequência de fases.

            Returns:
            --------
            modulation_map : torch.Tensor
                Mapa de modulação calculado a partir das componentes de seno e cosseno.
            phi_image : torch.Tensor
                Uma matriz bidimensional representando a imagem Phi. Cada valor de pixel na imagem corresponde
                ao ângulo Phi calculado para aquele pixel com base nas contribuições de seno e cosseno dos canais.
        """
        image_t = self._to_tensor(image, dtype=torch.float32)
        num_channels = image_t.shape[2]

        indices = torch.arange(1, num_channels + 1, dtype=torch.float32, device=image_t.device)
        angle = 2.0 * math.pi * indices / float(num_channels)

        sin_values = torch.sin(angle)
        cos_values = torch.cos(angle)

        sin_contributions = torch.sum(image_t * sin_values, dim=2)
        cos_contributions = torch.sum(image_t * cos_values, dim=2)

        # Calcular Phi para cada pixel
        phi_image = torch.atan2(-sin_contributions, cos_contributions).to(torch.float64)

        # Calcular o mapa de modulação
        modulation_map = torch.sqrt(sin_contributions ** 2 + cos_contributions ** 2) / num_channels
        modulation_map = torch.clamp(modulation_map, min=0, max=255).to(torch.uint8)
        if visualize:
            img_l_t = self._to_tensor(self.images_left, dtype=torch.float32)
            img_r_t = self._to_tensor(self.images_right, dtype=torch.float32)
            steps = int(FringePattern.get_steps(self))

            modulation_map_left, phi_image_left = self.calculate_phi(img_l_t[:, :, :steps], visualize=False)
            modulation_map_right, phi_image_right = self.calculate_phi(img_r_t[:, :, :steps], visualize=False)

            qsi_image_left = self.calculate_qsi(img_l_t[:, :, steps:], visualize=False)
            qsi_image_right = self.calculate_qsi(img_r_t[:, :, steps:], visualize=False)

            gc_order = GrayCode.get_gc_order_v(self)
            remaped_qsi_image_left = self.remap_qsi_image(qsi_image_left, gc_order)
            remaped_qsi_image_right = self.remap_qsi_image(qsi_image_right, gc_order)

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            middle_index_left = int(img_l_t.shape[0] / 2)
            middle_index_right = int(img_r_t.shape[0] / 2)

            self.plot_1d_phase(axes[0, 0], phi_image_left[middle_index_left, :],
                               remaped_qsi_image_left[middle_index_left, :], 'Phi Image left', 'Phi Image left')

            self.plot_1d_phase(axes[0, 1], phi_image_right[middle_index_right, :],
                               remaped_qsi_image_right[middle_index_right, :], 'Phi Image right',
                               'Phi Image right')

            self.plot_2d_image(axes[1, 0], phi_image_left, 'Phi Image left 2D')
            self.plot_2d_image(axes[1, 1], phi_image_right, 'Phi Image right 2D')

            fig.suptitle('Fase franjas - {}'.format(name))
            plt.tight_layout()
            plt.show()

        return modulation_map, phi_image

    def calculate_qsi(self, graycode_image, name='Plot', visualize=True):
        """
            Calcula a imagem QSI (Quantitative Structure Image) a partir de uma imagem codificada em graycode.

            A função converte uma imagem de código gray em uma imagem QSI. A imagem de entrada deve ter várias camadas,
            onde a primeira camada é a referência de branco e as camadas subsequentes contêm os bits do código gray.
            A função normaliza os valores de bits com relação à camada de referência de branco e, em seguida, converte
            cada conjunto de bits em um único número inteiro, gerando a imagem QSI.

            Parameters:
            -----------
            graycode_image : np.ndarray | torch.Tensor
                Uma matriz de três dimensões (altura, largura, camadas) representando a imagem de código gray.
                A primeira camada(0) contém os valores de referência de branco, enquanto as camadas subsequentes contêm
                os bits do código gray.

            Returns:
            --------
            qsi_image : torch.Tensor
                Uma matriz bidimensional representando a imagem QSI calculada. Cada pixel na imagem QSI corresponde
                a um valor inteiro derivado dos bits de código gray.
        """
        gc_img_t = self._to_tensor(graycode_image, dtype=torch.float32)

        # Obter o valor de branco para cada pixel (shape (H, W))
        white_value = gc_img_t[:, :, 0]
        white_value = torch.clamp(white_value, min=1e-6)

        # Comparar os bits relevantes com o branco correspondente
        bit_values = gc_img_t[:, :, 2:] / white_value.unsqueeze(-1)
        bit_values = (bit_values > 0.5).to(torch.int64)

        # Converter cada pixel de bits em um único número inteiro
        num_bits = bit_values.shape[-1]
        powers = 2 ** torch.arange(num_bits - 1, -1, -1, device=gc_img_t.device, dtype=torch.int64)
        qsi_image = torch.sum(bit_values * powers, dim=-1)

        if visualize:
            img_l_t = self._to_tensor(self.images_left, dtype=torch.float32)
            img_r_t = self._to_tensor(self.images_right, dtype=torch.float32)
            steps = int(FringePattern.get_steps(self))

            qsi_image_left = self.calculate_qsi(img_l_t[:, :, steps:], visualize=False)
            qsi_image_right = self.calculate_qsi(img_r_t[:, :, steps:], visualize=False)

            gc_order = GrayCode.get_gc_order_v(self)
            remaped_qsi_image_left = self.remap_qsi_image(qsi_image_left, gc_order)
            remaped_qsi_image_right = self.remap_qsi_image(qsi_image_right, gc_order)

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            self.plot_2d_image(axes[0, 0], qsi_image_left, 'Qsi Image left 2D')
            self.plot_2d_image(axes[0, 1], qsi_image_right, 'Qsi Image right 2D')
            self.plot_2d_image(axes[1, 0], remaped_qsi_image_left, 'Remaped Qsi Image left 2D')
            self.plot_2d_image(axes[1, 1], remaped_qsi_image_right, 'Remaped Qsi Image right 2D')

            fig.suptitle('Qsi & Remaped QSI {}'.format(name))
            plt.tight_layout()
            plt.show()

        return qsi_image

    def remap_qsi_image(self, qsi_image, real_qsi_order):
        """
            Remapeia os valores de uma imagem QSI de acordo com uma nova ordem QSI real.

            Esta função remapeia os valores da imagem QSI fornecida, utilizando uma ordem QSI real específica.
            O mapeamento é realizado criando uma tabela de lookup (LUT) que associa os valores originais aos novos índices,
            de acordo com a ordem fornecida.

            Parameters:
            -----------
            qsi_image : np.ndarray | torch.Tensor
                Uma matriz bidimensional representando a imagem QSI original cujos valores precisam ser remapeados.

            real_qsi_order : list | np.ndarray | torch.Tensor
                Uma lista ou array de inteiros representando a ordem real dos valores QSI. Cada valor nesta lista corresponde
                a um valor original da imagem QSI, e a posição desse valor na lista determina o novo índice a ser
                aplicado.

            Returns:
            --------
            remapped_qsi_image : torch.Tensor
                Uma matriz bidimensional com os valores remapeados de acordo com a nova ordem QSI.
                O resultado mantém a mesma forma da `qsi_image` original, mas com os valores ajustados conforme
                a nova ordem especificada.
        """
        qsi_t = self._to_tensor(qsi_image, dtype=torch.int64)

        if isinstance(real_qsi_order, (list, tuple)):
            order_tensor = torch.tensor(real_qsi_order, dtype=torch.int64, device=qsi_t.device)
        elif isinstance(real_qsi_order, np.ndarray):
            order_tensor = torch.from_numpy(real_qsi_order).to(device=qsi_t.device, dtype=torch.int64)
        elif isinstance(real_qsi_order, torch.Tensor):
            order_tensor = real_qsi_order.to(device=qsi_t.device, dtype=torch.int64)
        else:
            order_tensor = torch.as_tensor(real_qsi_order, device=qsi_t.device, dtype=torch.int64)

        max_order_val = int(order_tensor.max().item()) if order_tensor.numel() > 0 else 0
        max_qsi_val = int(qsi_t.max().item()) if qsi_t.numel() > 0 else 0
        max_val = max(max_order_val, max_qsi_val)

        lut = torch.zeros(max_val + 1, dtype=torch.int64, device=qsi_t.device)
        lut[order_tensor] = torch.arange(len(order_tensor), dtype=torch.int64, device=qsi_t.device)

        valid_mask = (qsi_t >= 0) & (qsi_t <= max_val)
        safe_qsi = torch.where(valid_mask, qsi_t, torch.zeros_like(qsi_t))
        remapped_qsi_image = torch.where(valid_mask, lut[safe_qsi], torch.zeros_like(qsi_t))

        return remapped_qsi_image

    def calculate_abs_phi_images(self, name='Plot', visualize=False, save=False):
        """
            Calcula as imagens de fase absoluta (phi) para os conjuntos de imagens esquerda e direita.

            Esta função gera as imagens de fase absoluta `abs_phi_image_left` e `abs_phi_image_right` a partir das imagens
            de fase `phi_image_left` e `phi_image_right`, bem como das imagens de QSI remapeadas correspondentes. As imagens
            de fase absoluta são calculadas considerando as diferentes condições de fase em relação a -π/2 e π/2, aplicando
            correções baseadas nos valores remapeados de QSI.

            O método é utilizado para garantir que as fases calculadas estejam em um intervalo contínuo e coerente para
            processamento subsequente, como na análise de padrões de fase ou reconstrução 3D.

            Parameters:
            -----------
            name : str
                Nome/título para gráficos e identificação.
            visualize : bool
                Se verdadeiro, exibe gráficos dos resultados.
            save : bool
                Se verdadeiro, salva o gráfico gerado.

            Returns:
            --------
            abs_phi_image_left : torch.Tensor
                Uma matriz bidimensional representando a imagem de fase absoluta correspondente à imagem
                `phi_image_left`. Os valores da fase estão em radianos.

            abs_phi_image_right : torch.Tensor
                Uma matriz bidimensional representando a imagem de fase absoluta correspondente à imagem
                `phi_image_right`. Os valores da fase estão em radianos.

            modulation_map_l : torch.Tensor
                Mapa de modulação da imagem esquerda.

            modulation_map_r : torch.Tensor
                Mapa de modulação da imagem direita.
        """
        t0 = time.time()

        img_l_t = self._to_tensor(self.images_left, dtype=torch.float32)
        img_r_t = self._to_tensor(self.images_right, dtype=torch.float32)

        modulation_map_l, phi_image_left = self.calculate_phi(img_l_t[:, :, self.n_min_bits:], visualize=False)
        modulation_map_r, phi_image_right = self.calculate_phi(img_r_t[:, :, self.n_min_bits:], visualize=False)

        qsi_image_left = self.calculate_qsi(img_l_t[:, :, :self.n_min_bits], visualize=False)
        qsi_image_right = self.calculate_qsi(img_r_t[:, :, :self.n_min_bits], visualize=False)

        gc_order = GrayCode.get_gc_order_v(self)
        remaped_qsi_image_left = self.remap_qsi_image(qsi_image_left, gc_order)
        remaped_qsi_image_right = self.remap_qsi_image(qsi_image_right, gc_order)

        pi = torch.tensor(math.pi, device=self.device, dtype=torch.float32)

        # Condição para a imagem esquerda
        mask_left1 = phi_image_left <= -pi / 2.0
        mask_left2 = (phi_image_left > -pi / 2.0) & (phi_image_left < pi / 2.0)
        mask_left3 = phi_image_left >= pi / 2.0

        abs_phi_image_left = torch.zeros_like(phi_image_left,dtype=torch.float64)
        remap_l_float = remaped_qsi_image_left.to(torch.float32)

        abs_phi_image_left[mask_left1] = phi_image_left[mask_left1] + 2.0 * pi * torch.floor(
            (remap_l_float[mask_left1] + 1.0) / 2.0) + pi

        abs_phi_image_left[mask_left2] = phi_image_left[mask_left2] + 2.0 * pi * torch.floor(
            remap_l_float[mask_left2] / 2.0) + pi

        abs_phi_image_left[mask_left3] = phi_image_left[mask_left3] + 2.0 * pi * (
                torch.floor((remap_l_float[mask_left3] + 1.0) / 2.0) - 1.0) + pi

        # Condição para a imagem direita
        mask_right1 = phi_image_right <= -pi / 2.0
        mask_right2 = (phi_image_right > -pi / 2.0) & (phi_image_right < pi / 2.0)
        mask_right3 = phi_image_right >= pi / 2.0

        abs_phi_image_right = torch.zeros_like(phi_image_right, dtype=torch.float64)
        remap_r_float = remaped_qsi_image_right.to(torch.float32)

        abs_phi_image_right[mask_right1] = phi_image_right[mask_right1] + 2.0 * pi * torch.floor(
            (remap_r_float[mask_right1] + 1.0) / 2.0) + pi

        abs_phi_image_right[mask_right2] = phi_image_right[mask_right2] + 2.0 * pi * torch.floor(
            remap_r_float[mask_right2] / 2.0) + pi

        abs_phi_image_right[mask_right3] = phi_image_right[mask_right3] + 2.0 * pi * (
                torch.floor((remap_r_float[mask_right3] + 1.0) / 2.0) - 1.0) + pi

        # Salvar nos atributos
        self.phi_image_left = phi_image_left
        self.phi_image_right = phi_image_right
        self.modulation_map_left = modulation_map_l
        self.modulation_map_right = modulation_map_r
        self.qsi_image_left = qsi_image_left
        self.qsi_image_right = qsi_image_right
        self.remaped_qsi_image_left = remaped_qsi_image_left
        self.remaped_qsi_image_right = remaped_qsi_image_right
        self.abs_phi_image_left = abs_phi_image_left
        self.abs_phi_image_right = abs_phi_image_right

        if visualize:
            fig, axes = plt.subplots(3, 2, figsize=(10, 8))

            middle_index_left = int(img_l_t.shape[0] / 2)
            middle_index_right = int(img_r_t.shape[0] / 2)

            self.plot_1d_phase(axes[0, 0], abs_phi_image_left[middle_index_left, :],
                               remaped_qsi_image_left[middle_index_left, :], 'Abs Phi Image left 1D',
                               'Abs Phi Image left')

            self.plot_1d_phase(axes[0, 1], abs_phi_image_right[middle_index_right, :],
                               remaped_qsi_image_right[middle_index_right, :], 'Abs Phi Image right 1D',
                               'Abs Phi Image right')

            self.plot_2d_image(axes[1, 0], abs_phi_image_left, 'Abs Phi Image left 2D')
            self.plot_2d_image(axes[1, 1], abs_phi_image_right, 'Abs Phi Image right 2D')

            self.plot_2d_image(axes[2, 0], modulation_map_l, 'Modulation Map left', cmap='jet')
            self.plot_2d_image(axes[2, 1], modulation_map_r, 'Modulation Map right', cmap='jet')
            if save:
                plt.savefig("gráfico_mapa_de_fase.png", dpi=300, bbox_inches='tight')

            fig.suptitle('Fase absoluta {}'.format(name))

            plt.tight_layout()
            plt.show()

        print('Process abs phase: {} dt'.format(round(time.time() - t0, 2)))
        return abs_phi_image_left.cpu().numpy(), abs_phi_image_right.cpu().numpy(), modulation_map_l.cpu().numpy(), modulation_map_r.cpu().numpy()

    def calculate_phi_images(self, visualize=False):
        """
            Calcula as imagens de fase phi para esquerda e direita e armazena nos atributos.
        """
        img_l_t = self._to_tensor(self.images_left, dtype=torch.float32)
        img_r_t = self._to_tensor(self.images_right, dtype=torch.float32)
        self.modulation_map_left, self.phi_image_left = self.calculate_phi(img_l_t[:, :, self.n_min_bits:], visualize=visualize)
        self.modulation_map_right, self.phi_image_right = self.calculate_phi(img_r_t[:, :, self.n_min_bits:], visualize=visualize)
        return self.phi_image_left, self.phi_image_right

    def calculate_qsi_images(self, visualize=False):
        """
            Calcula as imagens QSI para esquerda e direita e armazena nos atributos.
        """
        img_l_t = self._to_tensor(self.images_left, dtype=torch.float32)
        img_r_t = self._to_tensor(self.images_right, dtype=torch.float32)
        self.qsi_image_left = self.calculate_qsi(img_l_t[:, :, :self.n_min_bits], visualize=visualize)
        self.qsi_image_right = self.calculate_qsi(img_r_t[:, :, :self.n_min_bits], visualize=visualize)
        return self.qsi_image_left, self.qsi_image_right

    def calculate_remaped_qsi_images(self, visualize=False):
        """
            Remapeia as imagens QSI para esquerda e direita e armazena nos atributos.
        """
        if self.qsi_image_left is None or self.qsi_image_right is None:
            self.calculate_qsi_images(visualize=False)
        gc_order = GrayCode.get_gc_order_v(self)
        self.remaped_qsi_image_left = self.remap_qsi_image(self.qsi_image_left, gc_order)
        self.remaped_qsi_image_right = self.remap_qsi_image(self.qsi_image_right, gc_order)
        return self.remaped_qsi_image_left, self.remaped_qsi_image_right

    def plot_abs_phase_map(self, name='Plot', save=False):
        """Exibe o mapa de fase absoluta."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        self.plot_2d_image(axes[0], self.abs_phi_image_left, 'Abs Phi Left')
        self.plot_2d_image(axes[1], self.abs_phi_image_right, 'Abs Phi Right')
        fig.suptitle(name)
        plt.tight_layout()
        if save:
            plt.savefig(f"{name}_abs_phase.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_phase_map(self, name='Plot', save=False):
        """Exibe o mapa de fase envolvida."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        self.plot_2d_image(axes[0], self.phi_image_left, 'Phi Left')
        self.plot_2d_image(axes[1], self.phi_image_right, 'Phi Right')
        fig.suptitle(name)
        plt.tight_layout()
        if save:
            plt.savefig(f"{name}_phase.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_qsi_map(self, name='Plot', save=False):
        """Exibe o mapa de QSI."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        self.plot_2d_image(axes[0], self.remaped_qsi_image_left, 'Remaped QSI Left')
        self.plot_2d_image(axes[1], self.remaped_qsi_image_right, 'Remaped QSI Right')
        fig.suptitle(name)
        plt.tight_layout()
        if save:
            plt.savefig(f"{name}_qsi.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_1d_phase(self, ax, phi_image, remaped_qsi_image, title, ylabel):
        """
            Esta função cria dois gráficos sobrepostos no mesmo eixo. O primeiro gráfico mostra a imagem de
            fase (`phi_image`) e o segundo gráfico mostra a imagem QSI remapeada (`remaped_qsi_image`). O gráfico
            da imagem QSI remapeada é plotado em um eixo y secundário para permitir uma visualização clara das
            duas séries de dados com diferentes escalas.
        """
        if isinstance(phi_image, torch.Tensor):
            phi_image = phi_image.detach().cpu().numpy()
        if isinstance(remaped_qsi_image, torch.Tensor):
            remaped_qsi_image = remaped_qsi_image.detach().cpu().numpy()

        ax.plot(phi_image, color='gray')
        ax.set_ylabel(ylabel, color='gray')
        ax.tick_params(axis='y', labelcolor='gray')
        ax.set_title(title)
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(remaped_qsi_image, color='red')
        ax2.set_ylabel('Remaped QSI Image', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    def plot_2d_image(self, ax, image, title, cmap='gray'):
        """
            Esta função exibe uma imagem 2D em um eixo específico, com a opção de definir o título e o mapa de
            cores (colormap). Também adiciona uma barra de cores ao lado da imagem para indicar a escala dos valores.
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
