import numpy as np
import cv2


class GrayCode:
    def __init__(self, resolution=(512, 512), n_bits=4, axis=0, px_f=16):
        """
            Inicializa uma instância para a criação de imagens de Gray Code.

            Este metodo configura os parâmetros necessários para gerar imagens de Gray Code e inicializa
            a matriz de imagens com zeros. Ele também cria as imagens de Gray Code com base na resolução
            e no número de bits especificados.
        """
        self.width = resolution[0]
        self.height = resolution[1]
        self.px_f = px_f
        self.n_bits = n_bits  # Number of bits for the Gray code, adjusted to start from 0
        self.gc_images = np.zeros((self.height, self.width, self.n_bits + 2), np.uint8)  # Combined image sequence
        # self.create_images(axis=axis)  # Generate the Gray code images based on the specified axis
        self.create_graycode_images()

    def get_gc_images(self):
        """
            Retorna as imagens de Gray Code geradas.
        """
        return self.gc_images

    def show_image(self):
        """
            Exibe cada padrão de imagem na matriz `gc_images` em uma janela de visualização.
        """
        for i in range(self.gc_images.shape[2]):
            print(i)
            cv2.imshow('Image', self.gc_images[:, :, i])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_graycode_images(self):
        """
            Gera imagens de código de Gray para codificação de padrões projetados em uma cena.

            Esta função cria uma sequência de imagens codificadas em Gray (Gray code) usadas em técnicas como
            projeção de padrões estruturados. Essas imagens são utilizadas para calcular a profundidade ou mapear
            a geometria da cena.

            Parameters:
            -----------
            Não possui parâmetros de entrada.

            Returns:
            --------
            Nenhum valor é retornado explicitamente. A função modifica o atributo `gc_images` da classe, que é
            uma matriz tridimensional de forma (altura, largura, canais) onde os padrões de código de Gray são
            armazenados.
        """
        n_bits = np.ceil(np.log2(self.width / self.px_f / 2))
        width = [element for element in np.arange(2 ** self.n_bits, dtype=np.uint8) for _ in range(int(self.px_f / 2))]
        width_b = self.list_to_graycode_binary(width, self.n_bits)
        self.gc_images[:, :, 0] = 255

        for j in range(self.width):
            for i in range(self.n_bits):
                self.gc_images[:, j, i + 2] = int(list(width_b[j])[i]) * 255

    def list_to_graycode_binary(self, int_list, bit_length):
        """
            Converte uma lista de inteiros em suas representações binárias em código de Gray.

            Esta função recebe uma lista de inteiros e converte cada inteiro em seu código de Gray
            correspondente. O código de Gray é uma forma de codificação binária onde dois números
            consecutivos diferem em apenas um bit.

            Parameters:
            -----------
            int_list : list of int
                Uma lista de inteiros que serão convertidos para código de Gray.

            bit_length : int
                O comprimento em bits que cada código de Gray deve ter. Este valor determina o
                número de dígitos binários na representação final, incluindo os zeros à esquerda.

            Returns:
            --------
            graycode_list : list of str
                Uma lista de strings, onde cada string é a representação binária do código de Gray
                correspondente a um inteiro da lista de entrada, com zeros à esquerda conforme
                especificado pelo `bit_length`.
        """
        graycode_list = []
        for n in int_list:
            graycode = n ^ (n >> 1)
            # Convert to binary string with leading zeros based on the bit length
            graycode_binary = format(graycode, f'0{bit_length}b')
            graycode_list.append(graycode_binary)
        return graycode_list

    def get_gc_order_v(self):
        """
            Obtém a ordem dos valores de código de Gray na imagem horizontal.

            Esta função analisa a porção relevante de uma imagem que contém padrões de código de Gray,
            converte esses padrões em valores binários, e então os converte em inteiros. Os valores únicos
            são identificados, indexados, e ordenados para determinar a sequência de código de Gray.

            Parameters:
            -----------
            Não possui parâmetros de entrada.

            Returns:
            --------
            sorted_qsi_val : numpy.ndarray
                Um array contendo os valores únicos de código de Gray, ordenados de acordo com sua
                posição original na imagem.
        """
        # Converter a porção relevante da imagem para valores binários
        bit_values = (self.gc_images[0, :, 2:] / 255).astype(int)
        # Converter os valores binários em inteiros
        int_values = np.dot(bit_values, 2 ** np.arange(bit_values.shape[-1])[::-1])
        # Obter valores únicos e seus índices
        qsi_val, indices = np.unique(int_values, return_index=True)

        # Combinar os valores únicos e índices em uma lista de tuplas e ordenar
        sorted_indices = np.argsort(indices)
        sorted_qsi_val = qsi_val[sorted_indices]

        return sorted_qsi_val