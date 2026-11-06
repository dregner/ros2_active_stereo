import numpy as np
import cv2
import matplotlib.pyplot as plt
class FringePattern:

    def __init__(self, resolution=(1920, 1080), px_f=16, steps=11):
        """
            Inicializa uma instância para a criação de imagens de franja.

            Este metodo configura os parâmetros necessários para gerar imagens de franja e inicializa
            a matriz de imagens com zeros. Ele também gera as imagens de franja com base na resolução,
            no fator de pixel e no número de etapas fornecidos.
        """
        self.width = resolution[0]
        self.height = resolution[1]
        self.n_fringes = np.floor(self.width / px_f)  # function sine
        self.sin_values = []
        self.steps = steps # number of time steps
        self.fr_images = np.zeros((int(resolution[1]), int(resolution[0]), self.steps), dtype=np.uint8) # vector image
        self.create_fringe_image()

    def get_steps(self):
        """
            Retorna o número de etapas ou passos de franja.
        """
        return self.steps

    def show_image(self): # reading last shape of vector image
        """
           Exibe cada padrão de franja na matriz `fr_images` em uma janela de visualização.
        """
        for i in range(self.fr_images.shape[2]):
            cv2.imshow('Image', self.fr_images[:, :, i])
            cv2.waitKey(0)

    def print_image(self): # reading shape 0 e 1
        """
            Imprime a matriz de imagem `fr_images` linha por linha.

            Esta função percorre a matriz tridimensional `fr_images`, que representa uma imagem em formato
            de matriz, e imprime os valores de cada pixel na saída padrão. Cada linha da imagem é impressa
            em uma linha separada na saída, e uma mensagem "finished" é exibida após a conclusão da impressão.

            Parameters:
            -----------
            Não possui parâmetros de entrada.

            Returns:
            --------
            Nenhum valor é retornado. A função imprime os valores da imagem na saída padrão.
        """
        for i in range(self.fr_images.shape[0]):
            for j in range(self.fr_images.shape[1]):
                print(self.fr_images[i, j], end='')
            print('\n')
        print("finshed")

    def create_fringe_image(self):
        """
            Cria uma imagem de franjas usando uma série de deslocamentos de fase.

            Esta função gera uma imagem de franjas projetadas com base em funções seno. A função itera
            sobre um número definido de deslocamentos de fase (`steps`) e calcula valores senoidais
            correspondentes para cada deslocamento. Esses valores são usados para preencher a matriz de
            imagem `fr_images` com padrões de franjas.

            Parameters:
            -----------
            Não possui parâmetros de entrada.

            Returns:
            --------
            Nenhum valor é retornado. A função modifica o atributo `fr_images` da classe, preenchendo-o
            com valores de franjas projetadas.
        """
        x = np.arange(self.fr_images.shape[1])
        for n in range(self.steps): # phase shift of n=4
            phase_shift = n * 2 * np.pi/self.steps
            y = np.sin(2 * np.pi * float(self.n_fringes) * x / self.fr_images.shape[1] + phase_shift) + 1
            self.sin_values.append(y)
        for k in range(len(self.sin_values)):
            for i in range(self.fr_images.shape[0]):
                self.fr_images[i, :, k] = (self.sin_values[k]) * 255 / 2

    def get_fr_image(self):
        """
            Retorna as imagens de franja geradas.
        """
        return self.fr_images