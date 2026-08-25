import cv2
import numpy as np
import screeninfo

class ProjectorControl:
    def __init__(self, img_resolution=(1920,1080), monitor_name='DP-1', index=0):
        self.img_resolution = img_resolution
        self.current_image_index = index
        self.fringe_images = []
        self.graycode_images = []

        # Detecta o monitor e configura posição
        self.move = self._detect_monitor(monitor_name)

        # Combina padrões de projeção
        self.n_img = 0
        self.n_img_max = 0

    def set_images(self, fringe, graycode):
        """Define imagens de fringe e Gray Code."""
        self.fringe_images = fringe
        self.graycode_images = graycode
        self.n_img = self._combine_patterns()
        self.n_img_max = self.n_img.shape[2] - 1

        
    def _detect_monitor(self, monitor_name):
        """Detecta o monitor pelo nome e retorna sua posição."""
        for m in screeninfo.get_monitors():
            if monitor_name in m.name:
                self.img_resolution = (m.width, m.height)
                return (m.x, m.y)
        raise ValueError(f"Monitor '{monitor_name}' não encontrado. Monitores disponiveis: {m.name}")

    def _combine_patterns(self):
        """Combina imagens de fringe e Gray Code."""
        return np.concatenate((self.graycode_images, self.fringe_images), axis=2)

    def setup_projector_window(self):
        """Configura a janela de projeção em tela cheia no monitor selecionado."""
        cv2.namedWindow('projector', cv2.WINDOW_NORMAL)
        cv2.moveWindow('projector', self.move[0], self.move[1])
        cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def get_next_image(self):
        """Returns the current index and projects the next pattern image."""
        self.current_image_index = (self.current_image_index + 1) % (self.n_img_max + 1)
        pattern_image = self.n_img[:, :, self.current_image_index]
        pattern_image = self.convert_to_BGR(pattern_image)

        cv2.imshow('projector', pattern_image)
        # Use minimum waitKey so the OS compositor flips the framebuffer
        # without artificially delaying the caller. Settling time is handled
        # externally (settle_ms parameter / state machine timer).
        cv2.waitKey(1)
        return self.current_image_index, self.n_img_max

    def project_image(self, image):
        """Projects a specific image (e.g. black frame between scans)."""
        cv2.imshow('projector', image)
        cv2.waitKey(1)
    
    def convert_to_BGR(self, image):
        """Converte uma imagem para o formato BGR."""
        bgr_image = np.zeros((*image.shape, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = image  # Blue channel
        # Green and Red channels remain 0 (black)
        return bgr_image
