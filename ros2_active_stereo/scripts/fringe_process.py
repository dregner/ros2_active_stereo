import math
import numpy as np
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt


class Fringe_Process():
    def __init__(self, img_resolution=(1920, 1080), camera_resolution=(2448, 2048), px_f=12, steps=12):
        """
            Inicializa uma instância da classe com parâmetros específicos de resolução e configuração.

            Este metodo inicializa as variáveis necessárias para o processamento de imagens, incluindo
            imagens remapeadas e de fase para a esquerda e direita, bem como imagens capturadas pela câmera.
        """
        self.images_left = np.zeros(
            (camera_resolution[1], camera_resolution[0],
             int(steps + self.min_bits_gc(np.floor(img_resolution[0] / px_f)) + 2)), np.uint8)
        self.images_right = np.zeros(
            (camera_resolution[1], camera_resolution[0],
             int(steps + self.min_bits_gc(np.floor(img_resolution[0] / px_f)) + 2)), np.uint8)
        self.n_min_bits = self.min_bits_gc(np.floor(img_resolution[0] / px_f)) + 2

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
        return math.ceil(math.log2(x)+1)

    def normalize_white(self, mask_left, mask_right):
        """
            Calcula a média dos valores máximos dos pixels brancos nas imagens esquerda e direita usando máscaras.

            Parameters:
            -----------
            mask_left : numpy.ndarray
                Máscara binária aplicada à imagem esquerda, onde os pixels de interesse são marcados com o valor 255.

            mask_right : numpy.ndarray
                Máscara binária aplicada à imagem direita, onde os pixels de interesse são marcados com o valor 255.

            Returns:
            --------
            media_branco_max_left : float
                Média dos valores máximos dos pixels brancos na imagem esquerda, calculada a partir da máscara.

            media_branco_max_right : float
                Média dos valores máximos dos pixels brancos na imagem direita, calculada a partir da máscara.
        """
        # Assuming self.mask_left and self.mask_right are the masks for left and right images
        # They should have the same dimensions as the respective images

        media_branco_max_left = np.mean(self.images_left[:, :, self.steps][mask_left == 255])
        media_branco_max_right = np.mean(self.images_right[:, :, self.steps][mask_right == 255])

        return media_branco_max_left, media_branco_max_right

    def set_images(self, image_left, image_right, count):
        """
            Atribui imagens para os índices especificados nas matrizes de imagens esquerda e direita.
        """
        self.images_left[:, :, count] = image_left
        self.images_right[:, :, count] = image_right

    def calculate_phi(self, image):
        """
            Calcula a imagem de fase (phi) a partir de uma imagem de múltiplos canais utilizando transformações senoidais e cossenoidais.

            Esta função processa uma imagem composta por múltiplos canais (por exemplo, imagens obtidas através de projeções de padrões de fase)
            e calcula a fase correspondente para cada pixel. O cálculo é realizado aplicando funções seno e cosseno aos canais da imagem
            e combinando esses valores para obter a fase através da função arctan2.

            Parameters:
            -----------
            image : np.ndarray
                Uma matriz NumPy tridimensional (altura, largura, canais), onde cada canal representa uma amostra
                de fase em diferentes momentos ou ângulos. A função espera que os canais sejam organizados em uma
                sequência de fases.

            Returns:
            --------
            phi_image : np.ndarray
                Uma matriz NumPy bidimensional representando a imagem Phi. Cada valor de pixel na imagem corresponde
                ao ângulo Phi calculado para aquele pixel com base nas contribuições de seno e cosseno dos canais.
        """
        # Calcular os valores de seno e cosseno para todos os canais de uma vez
        indices = np.arange(1, image.shape[2] + 1)
        angle = 2 * np.pi * indices / image.shape[2]

        sin_values = np.sin(angle)
        cos_values = np.cos(angle)

        # Multiplicar a imagem pelos valores de seno e cosseno
        sin_contributions = np.sum(image * sin_values, axis=2)
        cos_contributions = np.sum(image * cos_values, axis=2)

        # Calcular Phi para cada pixel
        phi_image = np.arctan2(-sin_contributions, cos_contributions)

        # Calcular o mapa de modulação
        # modulation_map = np.sqrt(sin_contributions ** 2 + cos_contributions ** 2)
        modulation_map = np.sqrt(sin_contributions ** 2 + cos_contributions ** 2)/image.shape[2]

        # Converter para mensagem ROS
        # bridge = CvBridge()
        # modulation_map_32 = modulation_map.astype(np.float32)
        # modulation_mask_msg = bridge.cv2_to_imgmsg(modulation_map_32, encoding="32FC1")

        return modulation_map, phi_image

    def calculate_qsi(self, graycode_image):
        """
            Calcula a imagem QSI (Quantitative Structure Image) a partir de uma imagem codificada em graycode.

            A função converte uma imagem de código gray em uma imagem QSI. A imagem de entrada deve ter várias camadas,
            onde a primeira camada é a referência de branco e as camadas subsequentes contêm os bits do código gray.
            A função normaliza os valores de bits com relação à camada de referência de branco e, em seguida, converte
            cada conjunto de bits em um único número inteiro, gerando a imagem QSI.

            Parameters:
            -----------
            graycode_image : np.ndarray
                Uma matriz NumPy de três dimensões (altura, largura, camadas) representando a imagem de código gray.
                A primeira camada(0) contém os valores de referência de branco, enquanto as camadas subsequentes contêm
                os bits do código gray.

            Returns:
            --------
            qsi_image : np.ndarray
                Uma matriz NumPy bidimensional representando a imagem QSI calculada. Cada pixel na imagem QSI corresponde
                a um valor inteiro derivado dos bits de código gray.
            """

        # Obter o valor de branco para cada pixel (shape (X, Y))
        white_value = graycode_image[:, :, 0].astype(float)
        white_value = np.maximum(white_value, 1e-6)

        # Comparar os bits relevantes com o branco correspondente
        bit_values = graycode_image[:, :, 2:] / white_value[:, :, None]
        bit_values = (bit_values > 0.4).astype(int)

        # Converter cada linha de bits em um único número inteiro
        qsi_image = np.dot(bit_values, 2 ** np.arange(bit_values.shape[-1])[::-1])

        return qsi_image

    def remap_qsi_image(self, qsi_image, real_qsi_order):
        """
        Remapeia os valores de uma imagem QSI de acordo com uma nova ordem QSI real.

        Esta função remapeia os valores da imagem QSI fornecida, utilizando uma ordem QSI real específica.
        O mapeamento é realizado criando um dicionário que associa os valores originais aos novos índices,
        de acordo com a ordem fornecida.

        Parameters:
        -----------
        qsi_image : np.ndarray
            Uma matriz NumPy bidimensional representando a imagem QSI original cujos valores precisam ser remapeados.

        real_qsi_order : list
            Uma lista de inteiros representando a ordem real dos valores QSI. Cada valor nesta lista corresponde
            a um valor original da imagem QSI, e a posição desse valor na lista determina o novo índice a ser
            aplicado.

        Returns:
        --------
        remapped_qsi_image : np.ndarray
            Uma matriz NumPy bidimensional com os valores remapeados de acordo com a nova ordem QSI.
            O resultado mantém a mesma forma da `qsi_image` original, mas com os valores ajustados conforme
            a nova ordem especificada.
        """
        # Cria um mapeamento dos valores originais para os novos índices
        value_to_new_index_map = {value: new_index for new_index, value in enumerate(real_qsi_order)}

        # Mapeia os valores da qsi_image usando numpy para operações vetorizadas
        remapped_qsi_image = np.vectorize(lambda x: value_to_new_index_map.get(x, 0))(qsi_image)

        return remapped_qsi_image

    def calculate_abs_phi_images(self, save=False):
        # Obtem imagens necessarias para o calculo de abs phi
        modulation_map_l, phi_image_left = self.calculate_phi(self.images_left[:, :, self.n_min_bits:])
        modulation_map_r, phi_image_right = self.calculate_phi(self.images_right[:, :, self.n_min_bits:])

        qsi_image_left = self.calculate_qsi(self.images_left[:, :, :self.n_min_bits])
        qsi_image_right = self.calculate_qsi(self.images_right[:, :, :self.n_min_bits])

        remaped_qsi_image_left = self.remap_qsi_image(qsi_image_left, GrayCode.get_gc_order_v(self))
        remaped_qsi_image_right = self.remap_qsi_image(qsi_image_right, GrayCode.get_gc_order_v(self))

        # Condição para a imagem esquerda
        mask_left1 = phi_image_left <= -np.pi / 2
        mask_left2 = (phi_image_left > -np.pi / 2) & (phi_image_left < np.pi / 2)
        mask_left3 = phi_image_left >= np.pi / 2

        abs_phi_image_left = np.zeros_like(phi_image_left)

        abs_phi_image_left[mask_left1] = phi_image_left[mask_left1] + 2 * np.pi * np.floor(
            (remaped_qsi_image_left[mask_left1] + 1) / 2) + np.pi

        abs_phi_image_left[mask_left2] = phi_image_left[mask_left2] + 2 * np.pi * np.floor(
            remaped_qsi_image_left[mask_left2] / 2) + np.pi

        abs_phi_image_left[mask_left3] = phi_image_left[mask_left3] + 2 * np.pi * (
                np.floor((remaped_qsi_image_left[mask_left3] + 1) / 2) - 1) + np.pi

        # Condição para a imagem direita
        mask_right1 = phi_image_right <= -np.pi / 2
        mask_right2 = (phi_image_right > -np.pi / 2) & (phi_image_right < np.pi / 2)
        mask_right3 = phi_image_right >= np.pi / 2


        abs_phi_image_right = np.zeros_like(phi_image_right)

        abs_phi_image_right[mask_right1] = phi_image_right[mask_right1] + 2 * np.pi * np.floor(
            (remaped_qsi_image_right[mask_right1] + 1) / 2) + np.pi

        abs_phi_image_right[mask_right2] = phi_image_right[mask_right2] + 2 * np.pi * np.floor(
            remaped_qsi_image_right[mask_right2] / 2) + np.pi

        abs_phi_image_right[mask_right3] = phi_image_right[mask_right3] + 2 * np.pi * (
                np.floor((remaped_qsi_image_right[mask_right3] + 1) / 2) - 1) + np.pi
        
        if save:
            fig, axes = plt.subplots(3, 2, figsize=(10, 8))

            middle_index_left = int(self.images_left.shape[1] / 2)
            middle_index_right = int(self.images_right.shape[1] / 2)

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
            plt.savefig("gráfico_mapa_de_fase.png", dpi=300, bbox_inches='tight')


        return abs_phi_image_left, abs_phi_image_right, modulation_map_l, modulation_map_r

    def plot_1d_phase(self, ax, phi_image, remaped_qsi_image, title, ylabel):
        """
            Esta função cria dois gráficos sobrepostos no mesmo eixo. O primeiro gráfico mostra a imagem de
            fase (`phi_image`) e o segundo gráfico mostra a imagem QSI remapeada (`remaped_qsi_image`). O gráfico
            da imagem QSI remapeada é plotado em um eixo y secundário para permitir uma visualização clara das
            duas séries de dados com diferentes escalas.
        """

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

        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
